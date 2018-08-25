# Agent classes

import numpy as np

class RandomAgent(object):
    def __init__(self, state):
        self.current_state = state

    def get_observation(self, state):
        self.current_state = state

    def get_action(self):
        valid_action = False
        while not valid_action:
            action = np.random.randint(low=0, high=7)
            if np.min(self.current_state[:,action]) == 0:
                valid_action = True
        return action