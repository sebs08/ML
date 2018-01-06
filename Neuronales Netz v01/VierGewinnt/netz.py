"""
Schlachtplan:

Architektur:
    s_1 = 7*6 = 42 inputs units(spielfeld)
    s_2 = 7 outputs units(reihe)
    s_3 = 7 hidden units

    L = # of layers = 3


    42 (+1) -> 7 (+1) -> 7

    Theta_1 = 7 x 43 matrix (with bias)
    Theta_2 = 8 x 7 matrix (with bias)

    Theta_1__j_i correseponds to the j -th row; i-th column

    activation function1 = >0 function
    activation function2 = sigmoid

    x = input vector
    a_1 = hidden layer vector
    a_2 = output vector

    a_1 = activation function1 (Theta1 * x)
    a_2 = activation function2 (Theta2 * a_1)

    Neural Network Cost function:
    lambda -
    m - number of training examples
    x_(i) - i -th training example
    x_(i)_j - j -th component of the i -th training example
    y_(i) - classification of the i -th training example
    y_(i)_k - k -th component of the i -th classification

    Error function:

    -1/m * (sum over training examples _i ( sum over all classes _k ( y_(i)_k * log[(evaluated x_(i))_k]
                                                          + (1 - y_(i)_k) * log[1 - (evaluated x_(i))_k]  ) )
    + lambda / 2m (sum over layers _l (sum over nodes in layer s_l _i(sum over nodes in layer s_(l+1) _j
                                                [Theta_l__j_i ^2 ] )))

    Gradient of the error function:

    calculate forward propagation and save every step

"""

import numpy as np

class Net():
    """
    Neural Net with 1 hidden layer
    """
    def __init__(self, input_nodes, hidden_nodes, output_nodes, load_weights = None):

        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        if load_weights is None: # load with path or np.array ?
            # adding bias
            # randomly initalize weights for matrizes
            epsilon_init_1 = np.sqrt(6.)/np.sqrt(hidden_nodes + input_nodes)
            epsilon_init_2 = np.sqrt(6.)/np.sqrt(output_nodes + hidden_nodes)

            self.Theta_1 = np.random.rand((hidden_nodes,1+input_nodes))*epsilon_init_1*2 - epsilon_init_1
            self.Theta_2 = np.random.rand((output_nodes,1+hidden_nodes))*epsilon_init_2*2 - epsilon_init_2
        else:
            # load weights
            pass

        self.lambda_value = 0

    def sigmoid(self, vector):
        """

        :param vector: np.array
        :return: np.array - evaluated sigmoid function
        """
        return 1/(1+np.exp(-vector))

    def for_prop_single(self):
        pass

    def back_pro_single(self):
        pass

    def train_multi(self):
        pass

    def train_single(self):
        pass

    def save_weights(self):
        pass

    def load_weights(self):
        pass

    def run_tests(self):
        pass

    def run_gradient_test(self):
        pass

    def plot_error(self):
        pass

    def cross_validation(self):
        pass