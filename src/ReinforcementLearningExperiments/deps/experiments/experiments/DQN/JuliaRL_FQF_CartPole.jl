
# ---
# title: JuliaRL\_FQF\_CartPole
# cover: assets/JuliaRL_FQF_CartPole.png
# description: FQF applied to CartPole
# date: 2022-03-06
# author: "[Harley Wiltzer](https://github.com/harwiltz)"
# ---


using ReinforcementLearning
using StableRNGs
using Flux
using Flux.Losses
using CUDA

function RL.Experiment(
    ::Val{:JuliaRL},
    ::Val{:FQF},
    ::Val{:CartPole},
    ::Nothing;
    seed = 123,
)
    rng = StableRNG(seed)
    device_rng = CUDA.functional() ? CUDA.CURAND.RNG() : rng
    env = CartPoleEnv(; T = Float32, rng = rng)
    ns, na = length(state(env)), length(action_space(env))
    init = glorot_uniform(rng)
    Nₑₘ = 16
    n_hidden = 64
    κ = 1.0f0

    n_atoms = 32

    agent = Agent(
        policy = QBasedPolicy(
            learner = FQFLearner(
                env;
                state_embedder = Dense(ns, n_hidden, relu; init = init),
                fraction_embedder = Dense(Nₑₘ, n_hidden, relu; init = init),
                optimizer = ADAM(0.001),
                κ = κ,
                N = 8,
                N′ = 8,
                Nₑₘ = Nₑₘ,
                K = n_atoms,
                γ = 0.99f0,
                stack_size = nothing,
                batch_size = 32,
                update_horizon = 1,
                min_replay_history = 100,
                update_freq = 1,
                target_update_freq = 100,
                default_priority = 1.0f2,
                rng = rng,
                device_rng = device_rng,
            ),
            explorer = EpsilonGreedyExplorer(
                kind = :exp,
                ϵ_stable = 0.01,
                decay_steps = 500,
                rng = rng,
            ),
        ),
        trajectory = CircularArrayPSARTTrajectory(
            capacity = 1000,
            state = Vector{Float32} => (ns,),
        ),
    )

    stop_condition = StopAfterStep(10_000, is_show_progress=!haskey(ENV, "CI"))
    hook = TotalRewardPerEpisode()
    Experiment(agent, env, stop_condition, hook, "")
end


#+ tangle=false
using Plots
pyplot() #hide
ex = E`JuliaRL_FQF_CartPole`
run(ex)
plot(ex.hook.rewards)
savefig("assets/JuliaRL_FQF_CartPole.png") #hide

# ![](assets/JuliaRL_FQF_CartPole.png)
