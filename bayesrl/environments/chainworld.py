import numpy as np
from ..utils import check_random_state

class ChainWorld(object):
    def __init__(self, left_length, left_reward, right_length, right_reward, on_chain_reward, p_return_to_start, random_state=None):
        self.left_length = left_length
        self.left_reward = left_reward
        self.right_length = right_length
        self.right_reward = right_reward
        self.on_chain_reward = on_chain_reward
        self.p_return_to_start = p_return_to_start
        self.num_states = self.left_length + self.right_length + 1
        self.num_actions = 2
        self.random_state = check_random_state(random_state)
        self.reset()

    def reset(self):
        self.state = self.left_length

    def observe(self):
        return self.state

    def is_terminal(self, state):
        return state == 0 or state == self.num_states - 1

    def perform_action(self, action):
        if self.p_return_to_start and self.random_state.rand() < self.p_return_to_start:
            self.reset()
        elif action == 0:
            self.state -= 1
        else:
            self.state += 1

        if self.state == 0:
            reward = self.left_reward
        elif self.state == self.num_states - 1:
            reward = self.right_reward
        else:
            reward = self.on_chain_reward
        return self.observe(), reward

    def get_max_reward(self):
        return max(self.left_reward, self.right_reward)
