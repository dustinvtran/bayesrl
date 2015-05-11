import numpy as np
import itertools


class Agent(object):
    """
    Base class for all reinforcement learning agents to inherit from.

    Parameters
    ----------
    grid: the environment the agent acts on.
    gamma: float in (0,1]
        The discount factor per iteration.
    target_reward: reward for getting a target item
	reward is -1 on non-target states.
    """
    def __init__(self, grid, gamma=.9, target_reward=100):
	self.discount_factor = gamma
	self.target_reward = target_reward
	self.grid = grid
	self.states = [(r,c) for r in range(self.grid.height) for c in range(self.grid.width)]
        self.num_actions = len(self.grid.actions)
        self.value_table = np.zeros((self.grid.height, self.grid.width, self.num_actions))


    def _value_iteration(self):
	value = np.zeros(self.value_table.shape)
	reward = self.get_reward_state()
	most_likely_state =
	self.states[self._argmax_breaking_ties_randomly(np.ravel(self.grid.belief))]
        k = 0
        while True:
            diff = 0
            for s in self.states:
                old = value[s]
                value[s] = np.max([np.sum([p*(reward[s_]+self.discount_factor*value[s_][a]) for (s_,p)
		in self.grid.transition_probs(s,a).items()]) for a in self.grid.actions])
                diff = max(diff, abs(old - value[s]))
            k += 1
            if diff < 1e-2:
                break
            if k > 1e6:
                raise Exception("Value iteration not converging. Stopped at 1e6 iterations.")
	for s in self.states:
	    self.value_table[s] = [np.sum([p*(reward[s_]+self.discount_factor*value[s_]) for (s_,p)
	    in self.grid.transition_probs(s,a).items()]) for a in self.grid.actions]
        next_action =
	self._argmax_breaking_ties_randomly(self.value_table[most_likely_state])
	return self.grid.actions[next_action]

    def get_reward_state(self):
	targets = self.grid.targets
	target_states = []
	rewards = np.ones((self.grid.height, self.grid.width))*-1.
	aisles_belief = self.grid.aisles_belief
	content_belief = self.grid.content_belief
	categories = aisles_belief[1].keys()
	aisles_configs = []
	aisles_probs = []
	for config in itertools.permutations(categories):
	    aisles_configs.append(config)
	    aisle_prob = np.product([aisles_belief[i+1][config[i]] for i in
	    range(len(config))])
	    aisles_probs.append(aisle_prob)
	aisles_probs = np.array(aisles_probs)/sum(aisles_probs)
	multinomial = np.random.multinomial(1, aisles_probs)
	aisles_config = aisles_configs[list(multinomial).index(1)]
	items_configs = []
	for category in aisles_config:
	    shelf_configs = []
	    shelf_probs = []
	    items = content_belief[category][1].keys()
	    for config in itertools.permutations(items):
		shelf_configs.append(config)
		shelf_prob = np.product([content_belief[category][i][config[i]] for i in
		range(len(config))])
		shelf_probs.append(shelf_prob)
	    shelf_probs = np.array(shelf_probs)/sum(shelf_probs)
	    multinomial = np.random.multinomial(1, shelf_probs)
	    items_config = shelf_configs[list(multinomial).index(1)]
	    items_configs.append(items_config)
	    for t in targets:
		if t in items_config:
		    state =
		    self.grid.aisles[aisles_config.index(category)][items_config.index(t)]
		    target_states.append(state)
		    targets.discard(t)
	for s in target_states:
	    for dr, dc in self.grid.actions:
		neighbor = (s[0]+dr, s[1]+dc)
		if not self.grid.blocked(neighbor):
		    if rewards[neighbor] == -1:
			rewards[neighbor] = 0
		    rewards[neighbor] += self.target_reward
	return rewards


    # Make sure inherited classes have interact() function.
    def interact(self, reward, next_state, next_state_is_terminal):
	return


    def _argmax_breaking_ties_randomly(self, x):
        """Taken from Ken."""
        max_value = np.max(x)
        indices_with_max_value = np.flatnonzero(x == max_value)
        return np.random.choice(indices_with_max_value)
