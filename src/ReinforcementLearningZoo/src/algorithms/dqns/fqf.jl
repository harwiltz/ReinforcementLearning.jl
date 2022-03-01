export FQFLearner, FullyParameterizedQuantileNet

"""
    FQFLearner(;kwargs)

See [paper](https://arxiv.org/abs/1911.02140)

# Keyword arguments
- `approximator`, a [`ImplicitQuantileNet`](@ref)
- `target_approximator`, a [`ImplicitQuantileNet`](@ref), must have the same structure as `approximator`
- `κ = 1.0f0`,
- `N = 32`,
- `N′ = 32`,
- `Nₑₘ = 64`,
- `K = 32`,
- `γ = 0.99f0`,
- `stack_size = 4`,
- `batch_size = 32`,
- `update_horizon = 1`,
- `min_replay_history = 20000`,
- `update_freq = 4`,
- `target_update_freq = 8000`,
- `update_step = 0`,
- `default_priority = 1.0f2`,
- `β_priority = 0.5f0`,
- `rng = Random.GLOBAL_RNG`,
- `device_seed = nothing`,
"""
mutable struct FQFLearner{A,T,R,D} <: AbstractImplicitQuantileLearner
    approximator::A
    target_approximator::T
    sampler::NStepBatchSampler
    κ::Float32
    N::Int
    N′::Int
    Nₑₘ::Int
    K::Int
    min_replay_history::Int
    update_freq::Int
    target_update_freq::Int
    update_step::Int
    default_priority::Float32
    β_priority::Float32
    rng::R
    device_rng::D
    loss::Float32
end

function FQFLearner(;
    approximator,
    target_approximator,
    κ = 1.0f0,
    N = 32,
    N′ = 32,
    Nₑₘ = 64,
    K = 32,
    γ = 0.99f0,
    stack_size = 4,
    batch_size = 32,
    update_horizon = 1,
    min_replay_history = 20000,
    update_freq = 4,
    target_update_freq = 8000,
    update_step = 0,
    default_priority = 1.0f2,
    β_priority = 0.5f0,
    rng = Random.GLOBAL_RNG,
    device_rng = CUDA.CURAND.RNG(),
    traces = SARTS,
    loss = 0.0f0,
)
    copyto!(approximator, target_approximator)  # force sync
    if device(approximator) !== device(device_rng)
        throw(
            ArgumentError(
                "device of `approximator` doesn't match the device of `device_rng`: $(device(approximator)) !== $(device_rng)",
            ),
        )
    end
    sampler = NStepBatchSampler{traces}(;
        γ = γ,
        n = update_horizon,
        stack_size = stack_size,
        batch_size = batch_size,
    )
    FQFLearner(
        approximator,
        target_approximator,
        sampler,
        κ,
        N,
        N′,
        Nₑₘ,
        K,
        min_replay_history,
        update_freq,
        target_update_freq,
        update_step,
        default_priority,
        β_priority,
        rng,
        device_rng,
        loss,
    )
end

function (learner::FQFLearner)(env)
    s = send_to_device(device(learner), state(env))
    s = Flux.unsqueeze(s, ndims(s) + 1)
    τ = rand(learner.device_rng, Float32, learner.K, 1)
    τₑₘ = embed(τ, learner.Nₑₘ)
    quantiles = learner.approximator(s, τₑₘ)
    vec(sum(quantiles .* weights(τ); dims = 2)) |> send_to_host
end

function quantile_weights(τ::Float32)
    if ndims(τ) == 1
        τ = Flux.unsqueeze(τ, 2)
    end
    bdim = size(τ)[end]
    c₁ = Flux.unsqueeze(ones(bdim), 1)
    c₀ = Flux.unsqueeze(zeros(bdim), 1)
    τᵣ = vcat(τ, c₁)
    τₗ = vcat(c₀, τ)
    τᵣ - τₗ
end

function RLBase.update!(learner::FQFLearner, batch::NamedTuple)
    Z = learner.approximator
    Zₜ = learner.target_approximator
    N = learner.N
    N′ = learner.N′
    Nₑₘ = learner.Nₑₘ
    κ = learner.κ
    β = learner.β_priority
    batch_size = learner.sampler.batch_size

    D = device(Z)
    s, r, t, s′ =
        (send_to_device(D, batch[x]) for x in (:state, :reward, :terminal, :next_state))

    τ′ = rand(learner.device_rng, Float32, N′, batch_size)  # TODO: support β distribution
    τₑₘ′ = embed(τ′, Nₑₘ)
    zₜ = Zₜ(s′, τₑₘ′)
    w = weights(τ')
    avg_zₜ = sum(zₜ .* w, dims = 2)

    if haskey(batch, :next_legal_actions_mask)
        masked_value = fill(typemin(Float32), size(batch.next_legal_actions_mask))
        masked_value[batch.next_legal_actions_mask] .= 0
        avg_zₜ .+= send_to_device(D, masked_value)
    end

    aₜ = argmax(avg_zₜ, dims = 1)
    aₜ = aₜ .+ typeof(aₜ)(CartesianIndices((0:0, 0:N′-1, 0:0)))
    qₜ = reshape(zₜ[aₜ], :, batch_size)
    target =
        reshape(r, 1, batch_size) .+
        learner.sampler.γ * reshape(1 .- t, 1, batch_size) .* qₜ  # reshape to allow broadcast

    τ = rand(learner.device_rng, Float32, N, batch_size)
    τₑₘ = embed(τ, Nₑₘ)
    a = CartesianIndex.(repeat(batch.action, inner = N), 1:(N*batch_size))

    is_use_PER = haskey(batch, :priority)  # is use Prioritized Experience Replay
    if is_use_PER
        updated_priorities = Vector{Float32}(undef, batch_size)
        weights = 1.0f0 ./ ((batch.priority .+ 1f-10) .^ β)
        weights ./= maximum(weights)
        weights = send_to_device(D, weights)
    end

    gs = Zygote.gradient(Flux.params(Z)) do
        z = flatten_batch(Z(s, τₑₘ))
        q = z[a]

        TD_error = reshape(target, N′, 1, batch_size) .- reshape(q, 1, N, batch_size)
        # can't apply huber_loss in RLCore directly here
        abs_error = abs.(TD_error)
        quadratic = min.(abs_error, κ)
        linear = abs_error .- quadratic
        huber_loss = 0.5f0 .* quadratic .* quadratic .+ κ .* linear

        # dropgrad
        raw_loss =
            abs.(reshape(τ, 1, N, batch_size) .- Zygote.dropgrad(TD_error .< 0)) .*
            huber_loss ./ κ
        loss_per_quantile = reshape(sum(raw_loss; dims = 1), N, batch_size)
        loss_per_element = mean(loss_per_quantile; dims = 1)  # use as priorities
        loss =
            is_use_PER ? dot(vec(weights), vec(loss_per_element)) * 1 // batch_size :
            mean(loss_per_element)
        ignore() do
            # @assert all(loss_per_element .>= 0)
            is_use_PER && (
                updated_priorities .=
                    send_to_host(vec((loss_per_element .+ 1f-10) .^ β))
            )
            learner.loss = loss
        end
        loss
    end

    update!(Z, gs)

    is_use_PER ? updated_priorities : nothing
end
