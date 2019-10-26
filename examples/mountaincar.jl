using ReinforcementLearningEnvironmentClassicControl, ReinforcementLearning

# Creating a new environment of type MountainCar (implementing AbstractEnv)
# maxsteps defines the maximum steps the learner can make (within an episode?)
env = MountainCar(maxsteps = 10^4)

# StateAggregator lets us define discrete 'state bins' from a non-discrete
# state space. The below example creates....
high = [.5, .07]
low = [-1.2, -.07]
nbins = [8, 8]
p0 = StateAggregator(low, high, nbins)

preprocessor = TilingStateAggregator(p0, 8)
rlsetup = RLSetup(Sarsa(ns = 8*8^2, na = 3, α = 1/8, λ = .96, γ = 1.), 
                  env, ConstantNumberSteps(400),
                  preprocessor = preprocessor,
                  callbacks = [Visualize(wait = .02)])
rlsetup.policy.ϵ = 0
@info("Before learning.") 
run!(rlsetup)
rlsetup.callbacks = [EvaluationPerEpisode(TimeSteps())]
rlsetup.stoppingcriterion = ConstantNumberSteps(10^5)
@time learn!(rlsetup)
getvalue(rlsetup.callbacks[1])
@info("After learning.")
rlsetup.callbacks = [Visualize(wait = .02)]
rlsetup.stoppingcriterion = ConstantNumberSteps(400)
run!(rlsetup)
