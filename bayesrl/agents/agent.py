import numpy as np

class Agent(object):
    """
    Base class for all reinforcement learning agents to inherit from.

    Parameters
    ----------
    num_states: int
        Number of states in the task.
    num_actions: int
        Number of actions in the task.
    discount_factor: float in (0,1]
        The discount factor per iteration.
    """
    def __init__(self, num_states, num_actions, discount_factor):
        self.num_states = num_states
        self.num_actions = num_actions
        self.discount_factor = discount_factor

        self.last_state = None
        self.last_action = None

    def reset(self):
        self.last_state = None
        self.last_action = None

    # Make sure inherited classes have interact() function.
    def interact(self, reward, next_state, next_state_is_terminal):
        raise NameError("interact() has not been implemented.")
