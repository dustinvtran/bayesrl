from modelbasedagent import ModelBasedAgent
import numpy as np

class ThompsonSampAgent(ModelBasedAgent):
    def __init__(self, dirichlet_param, reward_param, **kwargs):
        super(ThompsonSampAgent, self).__init__(**kwargs)
        self.dirichlet_param = dirichlet_param
        self.reward_param = reward_param

        self.reward = np.full((self.num_states, self.num_actions, self.num_states), self.reward_param)

    def reset(self):
        super(ThompsonSampAgent, self).reset()
        self.reward.fill(self.reward_param)

    def interact(self, reward, next_state, next_state_is_terminal, idx):
        # Handle start of episode.
        if reward is None:
            # Return random action since there is no information.
            next_action = np.random.randint(self.num_actions)
            self.last_state = next_state
            self.last_action = next_action
            return self.last_action

        # Handle completion of episode.
        if next_state_is_terminal:
            # Proceed as normal.
            pass

        # Update the reward associated with (s,a,s') if first time.
        if self.reward[self.last_state, self.last_action, next_state] == self.reward_param:
            self.reward[self.last_state, self.last_action, next_state] = reward

        # Update set of states reached by playing a.
        self.transition_observations[self.last_state, self.last_action, next_state] += 1

        # Update transition probabilities after every T steps
        if self.policy_step == self.T:
            self.__compute_policy()

        # Choose next action according to policy.
        next_action = self._argmax_breaking_ties_randomly(self.value_table[next_state])

        self.policy_step += 1
        self.last_state = next_state
        self.last_action = next_action

        return self.last_action

    def __compute_policy(self):
        """Compute an optimal T-step policy for the current state."""
        self.policy_step = 0
        transition_probs = np.zeros((self.num_states, self.num_actions, self.num_states))
        for s in xrange(self.num_states):
            for a in xrange(self.num_actions):
                transition_probs[s,a] = np.random.dirichlet(self.transition_observations[s,a] +\
                                                            self.dirichlet_param, size=1)
        self._value_iteration(transition_probs)
