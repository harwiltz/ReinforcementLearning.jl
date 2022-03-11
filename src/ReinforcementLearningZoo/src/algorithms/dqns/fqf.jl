export FQFLearner 

using Random

"""
    FQFLearner(;kwargs)

See [paper](https://arxiv.org/abs/1911.02140)

# Keyword arguments
- `approximator`, a [`ImplicitQuantileNet`](@ref)
- `target_approximator`, a [`ImplicitQuantileNet`](@ref), must have the same structure as `approximator`
- `κ = 1.0f0`,
- `N = 32`,
- `Nₑₘ = 64`,
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
mutable struct FQFLearner{A,T,F,R,D} <: AbstractImplicitQuantileLearner
    approximator::A
    target_approximator::T
    fraction_proposer::F
    sampler::NStepBatchSampler
    κ::Float32
    N::Int
    Nₑₘ::Int
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

function FQFLearner(
    env;
    state_embedder,
    fraction_embedder,
    optimizer,
    fraction_proposal_optimizer = nothing,
    N = 32,
    rng = Random.GLOBAL_RNG,
    init_fn = glorot_uniform,
    kwargs...
)
    na = length(action_space(env))
    latent_size = size(Flux.modules(state_embedder)[end].weight)[1]
    header = Dense(latent_size, na, init = init_fn(rng))
    fraction_proposer = Chain(Dense(latent_size, N + 1, sigmoid, init = init_fn(rng)),
                              x -> cumsum(x; dims = 1),
                              # x -> x) |> gpu
                              x -> x[1:N,:] ./ x[end:end,:]) |> gpu
    approximator = NeuralNetworkApproximator(model = ImplicitQuantileNet(state_embedder,
                                                                         fraction_embedder,
                                                                         header) |> gpu,
                                             optimizer = optimizer)
    fpo = isnothing(fraction_proposal_optimizer) ? optimizer : fraction_proposal_optimizer
    fraction_proposer_approximator = NeuralNetworkApproximator(model = fraction_proposer,
                                                               optimizer = fpo)

    FQFLearner(;
        approximator = approximator,
        target_approximator = approximator,
        fraction_proposer = fraction_proposer_approximator,
        rng = rng,
        N = N,
        kwargs...
    )
end

function FQFLearner(;
    approximator,
    target_approximator,
    fraction_proposer,
    κ = 1.0f0,
    N = 32,
    Nₑₘ = 64,
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
        fraction_proposer,
        sampler,
        κ,
        N,
        Nₑₘ,
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
    state_embedding = learner.approximator(s)
    τ = learner.fraction_proposer(state_embedding)
    τₑₘ = embed(learner, τ)
    quantiles = learner.approximator(s, τₑₘ; features = state_embedding)
    weights = quantile_weights(learner, τ)
    @ein q_values[i,k] := quantiles[i,j,k] * weights[j,k];
    vec(q_values) |> send_to_host
end

function quantile_weights(learner::FQFLearner, τ::Vector{Float32})
    if ndims(τ) == 1
        τ = Flux.unsqueeze(τ, 2)
    end
    bdim = size(τ)[end]
    c₁ = Flux.unsqueeze(ones(Float32, bdim), 1)
    c₀ = Flux.unsqueeze(zeros(Float32, bdim), 1)
    τᵣ = vcat(τ[2:end], c₁)
    τₗ = vcat(c₀, τ[1:end - 1])
    τᵣ - τₗ
end

function RLBase.update!(learner::FQFLearner, batch::NamedTuple)
    Z = learner.approximator
    Zₜ = learner.target_approximator
    P = learner.fraction_proposer
    N = learner.N
    Nₑₘ = learner.Nₑₘ
    κ = learner.κ
    β = learner.β_priority
    batch_size = learner.sampler.batch_size

    D = device(Z)
    s, r, t, s′ =
        (send_to_device(D, batch[x]) for x in (:state, :reward, :terminal, :next_state))

    ξ′ = Zₜ(s′)
    τ′ = P(ξ′)  # TODO: support β distribution
    τₑₘ′ = embed(learner, τ′)
    zₜ = Zₜ(s′, τₑₘ′; features = ξ′)
    w = quantile_weights(learner, τ′)
    @ein avg_zₜ[i,k] := zₜ[i,j,k] * w[j,k];

    if haskey(batch, :next_legal_actions_mask)
        masked_value = fill(typemin(Float32), size(batch.next_legal_actions_mask))
        masked_value[batch.next_legal_actions_mask] .= 0
        avg_zₜ .+= send_to_device(D, masked_value)
    end

    aₜ = argmax(Flux.unsqueeze(avg_zₜ, 2), dims = 1)
    aₜ = aₜ .+ typeof(aₜ)(CartesianIndices((0:0, 0:N-1, 0:0)))
    qₜ = reshape(zₜ[aₜ], :, batch_size)
    target =
        reshape(r, 1, batch_size) .+
        learner.sampler.γ * reshape(1 .- t, 1, batch_size) .* qₜ  # reshape to allow broadcast

    is_use_PER = haskey(batch, :priority)  # is use Prioritized Experience Replay
    if is_use_PER
        updated_priorities = Vector{Float32}(undef, batch_size)
        weights = 1.0f0 ./ ((batch.priority .+ 1f-10) .^ β)
        weights ./= maximum(weights)
        weights = send_to_device(D, weights)
    end
    
    a = CartesianIndex.(repeat(batch.action, inner = N), 1:(N*batch_size))

    gs = Zygote.gradient(Flux.params(Z)) do
        ξ = Z(s)
        τ = P(ξ)
        τₑₘ = embed(learner, τ)

        z = flatten_batch(Z(s, τₑₘ; features = ξ))
        q = z[a]

        TD_error = reshape(target, N, 1, batch_size) .- reshape(q, 1, N, batch_size)
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

    τ = s |> Z |> P

    τ̂ = let
        bdim = size(τ)[end]
        c₁ = Flux.unsqueeze(ones(bdim), 1)
        c₀ = Flux.unsqueeze(zeros(bdim), 1)
        τᵣ = vcat(τ, c₁)
        τₗ = vcat(c₀, τ)
        (τₗ + τᵣ) / 2f0
    end

    τₑₘ = embed(learner, τ)
    τ̂ₑₘₗ = embed(learner, τ̂[1:end-1,:])
    τ̂ₑₘᵣ = embed(learner, τ̂[2:end,:])
    ξ = Z(s)
    z = flatten_batch(Z(s, τₑₘ; features = ξ))
    zₗ = flatten_batch(Z(s, τ̂ₑₘₗ; features = ξ))
    zᵣ = flatten_batch(Z(s, τ̂ₑₘᵣ; features = ξ))
    quantile_grads = @. 2 * z[a] - zₗ[a] - zᵣ[a]
    vjp(x) = ein"qb,qb->b"(P(x), reshape(quantile_grads, (learner.N, :)))
    js = gradient(() -> mean(vjp(ξ)), Flux.params(P))
    update!(P, js)

    is_use_PER ? updated_priorities : nothing
end
