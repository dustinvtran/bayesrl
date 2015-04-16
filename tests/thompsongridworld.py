"""
Solves grid world using Thompson sampling.
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

# Define agent.
# Dirichlet params = 1, Reward params = 50
agent = ThompsonSampAgent(
    num_states=task.num_states, num_actions=task.num_actions,
    discount_factor=0.95, T=50, dirichlet_param=1, reward_param=50)
trial_thompson1 = Trial(agent, task)
trial_thompson1.run_multiple(num_trials)

# Plots!
plot = Plot({"Thompson sampling": [trial_thompson1]})
plot.rewards_by_episode()
