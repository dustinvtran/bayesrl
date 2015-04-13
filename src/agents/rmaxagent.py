from modelbasedagent import ModelBasedAgent
import numpy as np

class RMAXAgent(ModelBasedAgent):
    """Runs R-MAX only for an MDP, i.e., not a stochastic game, in order to simplify data structures."""
    def __init__(self, min_visit_count, **kwargs):
        super(RMAXAgent, self).__init__(**kwargs)
        self.min_visit_count = min_visit_count

        self.Rmax = 50 # arbitrarily set (!)
        self.reward = np.full((self.num_states+1, self.num_actions, self.num_states+1), self.Rmax)
        self.transition_observations = np.zeros((self.num_states+1, self.num_actions, self.num_states+1))
        self.value_table = np.zeros((self.num_states+1, self.num_actions))

    def reset(self):
        super(RMAXAgent, self).reset()
        self.reward.fill(self.Rmax)
        self.transition_observations.fill(0)
        self.value_table.fill(0)

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
        if self.reward[self.last_state+1, self.last_action, next_state+1] == self.Rmax:
            self.reward[self.last_state+1, self.last_action, next_state+1] = reward
            if self.Rmax < reward:
                self.reward[self.reward == self.Rmax] = reward
                self.Rmax = reward

        # Update set of states reached by playing a.
        self.transition_observations[self.last_state+1, self.last_action, next_state+1] += 1

        # Compute new optimal T-step policy if reach min_visit_count or finished executing previous one
        if self.transition_observations[self.last_state+1, self.last_action].sum() == self.min_visit_count or \
           self.policy_step == self.T:
                self.__compute_policy()

        # Choose next action according to policy.
        next_action = self._argmax_breaking_ties_randomly(self.value_table[next_state+1])

        self.policy_step += 1
        self.last_state = next_state
        self.last_action = next_action

        return next_action

    def __compute_policy(self):
        """Compute an optimal T-step policy for the current state."""
        self.policy_step = 0
        # Obtain transition probabilities (prevent dividing by zero).
        divisor = self.transition_observations.sum(axis=2, keepdims=True)
        divisor[divisor == 0] = 1
        transition_probs = self.transition_observations / divisor
        # Replace all state-action pairs with zero probability everywhere, i.e.,
        # no counts, with probability 1 to the fictitious game state.
        eps = 1e-5
        for s in xrange(self.num_states+1):
            for a in xrange(self.num_actions):
                if -eps < transition_probs[s,a].sum() < eps:
                    transition_probs[s, a, 0] = 1
        self._value_iteration(transition_probs)
