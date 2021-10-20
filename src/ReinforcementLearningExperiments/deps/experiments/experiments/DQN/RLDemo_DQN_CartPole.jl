# ---
# title: RLDemo\_DQN\_CartPole
# description: A simple demo that renders rollouts
# date: 2021-10-19
# author: "[Harley Wiltzer](https://github.com/harwiltz)"
# ---

#+ tangle=true
using ReinforcementLearning
using StableRNGs
using Flux
using Flux.Losses

function RL.Experiment(
    ::Val{:RLDemo},
    ::Val{:DQN},
    ::Val{:CartPole},
    ::Nothing;
    seed = 123,
)
    layer_width = 128
    rng = StableRNG(seed)
    rng_test = StableRNG(seed + 1)
    env_fn(r) = CartPoleEnv(; T = Float32, rng = r)
    env = env_fn(rng)
    ns, na = length(state(env)), length(action_space(env))

    policy = Agent(
        policy = QBasedPolicy(
            learner = BasicDQNLearner(
                approximator = NeuralNetworkApproximator(
                    model = Chain(
                        Dense(ns, layer_width, relu; init = glorot_uniform(rng)),
                        Dense(layer_width, layer_width, relu; init = glorot_uniform(rng)),
                        Dense(layer_width, na; init = glorot_uniform(rng)),
                    ) |> cpu,
                    optimizer = ADAM(),
                ),
                batch_size = 32,
                min_replay_history = 100,
                loss_func = huber_loss,
                rng = rng,
            ),
            explorer = EpsilonGreedyExplorer(
                kind = :exp,
                ϵ_stable = 0.01,
                decay_steps = 500,
                rng = rng,
            ),
        ),
        trajectory = CircularArraySARTTrajectory(
            capacity = 1000,
            state = Vector{Float32} => (ns,),
        ),
    )
    stop_condition = StopAfterStep(10_000, is_show_progress=!haskey(ENV, "CI"))
    hook = TotalRewardPerEpisode()
    Experiment(policy, env, stop_condition, hook, "# DQN <-> CartPole")
end

#+ tangle=false
using Plots
using ReinforcementLearning
using ReinforcementLearningEnvironments

pyplot() #hide
ex = E`RLDemo_DQN_CartPole`
run(ex)
plot(ex.hook.rewards)
savefig("assets/RLDemo_DQN_CartPole.png") #hide

demo = Experiment(ex.policy,
                  CartPoleEnv(),
                  StopWhenDone(),
                  RolloutHook(plot, closeall),
                  "DQN <-> Demo")

run(demo)

# ![](assets/JuliaRL_BasicDQN_CartPole.png)
