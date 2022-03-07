export AbstractImplicitQuantileLearner, ImplicitQuantileNet

using OMEinsum

abstract type AbstractImplicitQuantileLearner <: AbstractLearner end

Flux.functor(x::AbstractImplicitQuantileLearner) =
    (Z = x.approximator, Zₜ = x.target_approximator, device_rng = x.device_rng),
    y -> begin
        x = @set x.approximator = y.Z
        x = @set x.target_approximator = y.Zₜ
        x = @set x.device_rng = y.device_rng
        x
    end

function (learner::AbstractImplicitQuantileLearner)(env)
    s = send_to_device(device(learner), state(env))
    s = Flux.unsqueeze(s, ndims(s) + 1)
    τ = rand(learner.device_rng, Float32, learner.K, 1)
    τₑₘ = embed(learner, τ)
    quantiles = learner.approximator(s, τₑₘ)
    weights = quantile_weights(learner, τ)
    q_values = ein"ijk,jk->ik"(quantiles, weights)
    vec(q_values) |> send_to_host
end

function quantile_weights(learner::AbstractImplicitQuantileLearner,
                          τ::Union{Vector{Float32}, Matrix{Float32}})
    ones(Float32, size(τ)) / size(τ)[1]
end

function embed(learner::AbstractImplicitQuantileLearner, τ)
    cos.(Float32(π) .* (1:learner.Nₑₘ) .* reshape(τ, 1, :))
end

"""
    ImplicitQuantileNet(;ψ, ϕ, header)

```
         quantiles (n_action, n_quantiles, batch_size)
            ↑
          header
            ↑
feature ↱  ⨀  ↰ transformed embedding
       ψ       ϕ
       ↑       ↑
       s       τ
```
"""
Base.@kwdef struct ImplicitQuantileNet{A,B,C}
    ψ::A
    ϕ::B
    header::C
end

Flux.@functor ImplicitQuantileNet

function (net::ImplicitQuantileNet)(s)
    net.ψ(s)
end

function (net::ImplicitQuantileNet)(s, emb; features = nothing)
    features = isnothing(features) ? net(s) : features # (n_feature, batch_size)
    emb_aligned = net.ϕ(emb)  # (n_feature, N * batch_size)
    merged =
        Flux.unsqueeze(features, 2) .*
        reshape(emb_aligned, size(features, 1), :, size(features, 2))  # (n_feature, N, batch_size)
    quantiles = net.header(flatten_batch(merged))
    reshape(quantiles, :, size(merged, 2), size(merged, 3))  # (n_action, N, batch_size)
end
