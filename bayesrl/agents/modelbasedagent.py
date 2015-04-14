from agent import Agent
import numpy as np

class ModelBasedAgent(Agent):
    """Runs R-MAX only for an MDP, i.e., not a stochastic game, in order to simplify data structures."""
    def __init__(self, T, **kwargs):
        super(ModelBasedAgent, self).__init__(**kwargs)
        self.T = T

        self.policy_step = self.T # To keep track of where in T-step policy the agent is in; initialized to recompute policy
        self.transition_observations = np.zeros((self.num_states, self.num_actions, self.num_states))
        self.value_table = np.zeros((self.num_states, self.num_actions))

    def reset(self):
        super(ModelBasedAgent, self).reset()
        self.policy_step = self.T # To keep track of where in T-step policy the agent is in; initialized to recompute policy
        self.transition_observations.fill(0)
        self.value_table.fill(0)

    def _value_iteration(self, transition_probs):
        """
        Run value iteration, using procedure described in Sutton and Barto
        (2012). The end result is an updated value_table, from which one can
        deduce the policy for state s by taking the argmax (breaking ties
        randomly).
        """
        value_dim = transition_probs.shape[0]
        value = np.zeros(value_dim)
        k = 0
        while True:
            diff = 0
            for s in xrange(value_dim):
                old = value[s]
                value[s] = np.max(np.sum(transition_probs[s]*(self.reward[s] +
                           self.discount_factor*np.array([value,]*self.num_actions)),
                           axis=1))
                diff = max(0, abs(old - value[s]))
            k += 1
            if diff < 1e-2:
                break
            if k > 1e6:
                raise Exception("Value iteration not converging. Stopped at 1e6 iterations.")
        for s in xrange(value_dim):
            self.value_table[s] = np.sum(transition_probs[s]*(self.reward[s] +
                   self.discount_factor*np.array([value,]*self.num_actions)),
                   axis=1)

    def _argmax_breaking_ties_randomly(self, x):
        """Taken from Ken."""
        max_value = np.max(x)
        indices_with_max_value = np.flatnonzero(x == max_value)
        return np.random.choice(indices_with_max_value)
