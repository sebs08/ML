# Agent classes

import numpy as np
import tensorflow as tf
from ConnectFour import reward_functions

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
    """
    https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/examples/tutorials/layers/cnn_mnist.py
    """
    options = {

        # hyper parameters
        'input_space': 42,
        'epsilon_rate': 0.1,
        'learning_rate': 0.001,

        # functions
        'reward_function': reward_functions.simple_reward,  # move to game-handler
        'loss_function': tf.losses.mean_squared_error,
        'optimizer': tf.train.GradientDescentOptimizer

    }

    def __init__(self, state, load_weights=False):
        self.logits = self.setup_tf_model()
        self.initialize_variables()
        self.current_state = state
        self.sess = tf.Session()

    def setup_tf_model(self):
        input_layer = tf.placeholder(dtype=tf.float32, shape=[None, 6, 7, 1], name='Input')

        reshape = tf.reshape(input_layer, [-1, 6, 7, 1])


        conv1 = tf.layers.conv2d(inputs=reshape,
                                 filters=6,
                                 kernel_size=[2,2],
                                 padding='same',
                                 activation=tf.nn.relu)

        conv2 = tf.layers.conv2d(inputs=conv1,
                                 filters=20,
                                 kernel_size=[3,3],
                                 padding='same',
                                 activation=tf.nn.relu)

        dense1 = tf.layers.dense(inputs=conv2, units=40, activation=tf.nn.relu)

        logits = tf.layers.dense(inputs=dense1, units=1, activation=tf.nn.relu)

        return logits

    def initialize_variables(self):
        tf.global_variables_initializer()

    def predict(self):
        pass


    def load(self):
        # load weights / inference
        pass

    def save(self):
        # possible to save weights & graph separately??
        pass

    def initialize(self):
        # initialize TensorFlor Variables
        # load stuff
        pass

    def get_current_state(self, state):
        pass

    def preprocess(self):
        # needed ??
        pass

    def get_action(self):
        return self.predict()

    def get_game_history(self):
        pass

    def train(self):
        pass

    def get_observation(self, state):
        self.current_state = state

