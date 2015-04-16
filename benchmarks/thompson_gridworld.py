"""
Solves grid world using three different parameter settings for Thompson
sampling. This empirically shows the convergence of Thompson sampling regardless
of the prior misspecification.
"""

from bayesrl.environments import GridWorld
from bayesrl.agents.thompsonsampagent import ThompsonSampAgent
from bayesrl.trial import Trial
from bayesrl.plot import Plot

# Define environment.
task = GridWorld(
    GridWorld.samples['larger'],
    action_error_prob=.1,
    rewards={'*': 50, 'moved': -1, 'hit-wall': -1})

num_trials = 1

################################################################################
# Thompson Sampling
################################################################################
# Dirichlet params = 1, Reward params = 50
agent = ThompsonSampAgent(
    num_states=task.num_states, num_actions=task.num_actions,
    discount_factor=0.95, T=50, dirichlet_param=1, reward_param=50)
trial_thompson1 = Trial(agent, task)
trial_thompson1.run_multiple(num_trials)

# Dirichlet params = 1, Reward params = 10
agent.dirichlet_param = 1
agent.reward_param = 10
trial_thompson2 = Trial(agent, task)
trial_thompson2.run_multiple(num_trials)

# Dirichlet params = 10, Reward params = 50
agent.dirichlet_param = 10
agent.reward_param = 50
trial_thompson3 = Trial(agent, task)
trial_thompson3.run_multiple(num_trials)

################################################################################
# Plots!
################################################################################
plot = Plot({"Thompson sampling": [trial_thompson1, trial_thompson2, trial_thompson3]
            })
# Plot cumulative rewards by iteration
plot.cum_rewards_by_iteration()
# Plot rewards by episode
plot.rewards_by_episode()
