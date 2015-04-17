from thompsonsampagent import ThompsonSampAgent
import numpy as np

class ThompsonSampAgentPOMDP(ThompsonSampAgent):
    def __init__(self, observation_model, dirichlet_param, reward_param, **kwargs):
        super(ThompsonSampAgentPOMDP, self).__init__(dirichlet_param, reward_param, **kwargs)
        self.observation_model = observation_model
        self.reset_belief()
        self.__compute_policy()

    def reset_belief(self):
        self.belief = np.array([1./self.num_states for _ in range(self.num_states)])

    def reset(self):
        super(ThompsonSampAgentPOMDP, self).reset()
        self.reset_belief()

    def interact(self, reward, observation, next_state_is_terminal, idx):
        self.last_state = self._argmax_breaking_ties_randomly(self.belief)
        if self.last_action is not None:
            self.__update_belief(self.last_action,observation)
            next_state = self._argmax_breaking_ties_randomly(self.belief)
        else:
            next_state = self.last_state
        return super(ThompsonSampAgentPOMDP, self).interact(reward,next_state,next_state_is_terminal,idx)

    def __compute_policy(self):
        """Compute an optimal T-step policy for the current state."""
        self.policy_step = 0
        self.transition_probs = np.zeros((self.num_states, self.num_actions, self.num_states))
        for s in xrange(self.num_states):
            for a in xrange(self.num_actions):
                self.transition_probs[s,a] = np.random.dirichlet(self.transition_observations[s,a] +\
                                                            self.dirichlet_param, size=1)
        self._value_iteration(self.transition_probs)

    def __update_belief(self,action,observation):
        self.__transition(action)
        self.__observe(observation)

    def __transition(self,action):
        for s in range(self.num_states):
            self.belief[s] = sum(self.transition_probs[s_,action,s]*self.belief[s_] for s_ in range(self.num_states))

    def __observe(self,observation):
        self.belief = [self.belief[s]*self.observation_model[s][observation] for s in range(self.num_states)]
        Z = sum(self.belief)
        self.belief = np.array(self.belief)/float(Z)
