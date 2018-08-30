# Agent classes

import numpy as np
import tensorflow as tf

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

class CNNAgent(object):
    options = {

        # hyper parameters
        'input_space': 42,
        'epsilon_rate': 0.1,
        'learning_rate': 0.001,
        'reward_function': lambda state: state,

    }

    def __init__(self, load_weights=False):
        pass

    def initialize(self):
        pass

    def load_weights(self):
        pass

    def save_weights(self):
        pass

    def initialize(self):
        pass

    def get_current_state(self):
        pass

    def get_action(self):
        pass

    def get_game_history(self):
        pass

    def train(self):
        pass

    def predict(self):
        pass