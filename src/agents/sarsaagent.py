from agent import Agent
import numpy as np

class SARSAAgent(Agent):
    def __init__(self, learning_rate, epsilon, value=0, **kwargs):
        super(SARSAAgent, self).__init__(**kwargs)
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.value = value

        self.value_table = np.full((self.num_states, self.num_actions), self.value)

    def reset(self):
        super(SARSAAgent, self).reset()
        self.value_table.fill(self.value)

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

        # Choose next action according to epsilon-greedy policy.
        if np.random.random() < self.epsilon:
            next_action = np.random.randint(self.num_actions)
        else:
            next_action = np.argmax(self.value_table[next_state])

        # Update value function.
        delta = reward + self.discount_factor*self.value_table[next_state, next_action] - \
                self.value_table[self.last_state, self.last_action]
        self.value_table[self.last_state, self.last_action] += self.learning_rate(idx) * delta

        self.last_state = next_state
        self.last_action = next_action

        return self.last_action
