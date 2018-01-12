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
import pickle # for saving weights

class NeuroNet():
    """
    Neural Net with 1 hidden layer
    """
    def __init__(self, input_nodes=0, hidden_nodes=0, output_nodes=0, load_weights=False, save_weights=False):

        if load_weights is False: # load with path or np.array ?

            self.input_nodes = input_nodes
            self.hidden_nodes = hidden_nodes
            self.output_nodes = output_nodes

            # adding bias
            # randomly initalize weights for matrizes
            epsilon_init_1 = np.sqrt(6.)/np.sqrt(hidden_nodes + input_nodes)
            epsilon_init_2 = np.sqrt(6.)/np.sqrt(output_nodes + hidden_nodes)

            self.Theta_1 = np.random.rand(hidden_nodes,1+input_nodes)*epsilon_init_1*2 - epsilon_init_1
            self.Theta_2 = np.random.rand(output_nodes,1+hidden_nodes)*epsilon_init_2*2 - epsilon_init_2
        else:
            # load weights
            self.Theta_1, self.Theta_2 = self.load_weights()
            # derive input, hidden and output nodes from matrix shapes
            (self.hidden_nodes, self.input_nodes) = np.shape(self.Theta_1)
            self.input_nodes = self.input_nodes - 1 # TODO: at this point bias terms don't get considered
            (self.output_nodes, _) = np.shape(self.Theta_2)

        self.lambda_value = 0

        self.save = save_weights

    def sigmoid(self, vector):
        """

        :param vector: np.array
        :return: np.array - evaluated sigmoid function
        """
        return 1/(1+np.exp(-vector))

    def for_prop_single(self, x):
        """
        forward propagation for one sample

        calculates: output = sigmoid( Theta_2 * sigmoid ( Theta_1 * x ))
        stores:
            - z_2   = Theta_1 * x
            - a_2     = sigmoid(a_1)
            - z_3   = Theta_2 * h
            - a_3 = sigmoid(a_2)
            note: a_1 ~ x

        :param x: vector
        :return: vectors
        """
        x = np.array(x)
        if np.shape(x)[0] != self.input_nodes+1:
            x = np.insert(x, 0, 1) # insert bias term # doing this in train_multi
        x = x.reshape((self.input_nodes+1,1))

        z_2 = np.dot(self.Theta_1, x) # calculate pre activation hidden layer
        a_2 = np.insert(self.sigmoid(z_2), 0, 1) # calculate h (hidden layer); insert bias term
        z_3 = np.dot(self.Theta_2, a_2) # calculate pre activation output layer
        a_3 = self.sigmoid(z_3)  # return value of output neurons
        return a_3, z_3, a_2, z_2

    def back_pro_single(self, x, error=lambda x: None):
        Delta_1 = np.zeros(np.shape(self.Theta_1)) # initialize Delta matrices, store gradient information
        Delta_2 = np.zeros(np.shape(self.Theta_2))

        x = np.insert(x, 0, 1)
        x = x.reshape((self.input_nodes+1,1))

        # get forward prop data
        a_3, z_3, a_2, z_2 = self.for_prop_single(x)

        # get error
        # TODO: make more general
        delta_3 = error(a_3)
        delta_3 = delta_3.reshape((self.output_nodes,1))
        # error gets calculated differently in case y == None
        # see script details

        a_2 = a_2.reshape((8,1))
        delta_2 = np.multiply(np.dot(self.Theta_2.T, delta_3), a_2)

        Delta_1 = Delta_1 + np.dot(delta_2[1:,],x.T)
        Delta_2 = Delta_2 + np.dot(delta_3,a_2.T)

        return {'D1': Delta_1, 'D2': Delta_2}


    def train_multi(self, batch_x, batch_y=None):
        #print("batch backprob: started")
        # each row in batch_x/y corresponds to one training sample
        Grad_1 = np.zeros(np.shape(self.Theta_1))  # initialize Delta matrices, store gradient information
        Grad_2 = np.zeros(np.shape(self.Theta_2))

        # TODO: make more general
        # defining error function here; should be more generalized
        def error_function(output, run_outcome):
            if run_outcome == 1:
                return output
            else:
                return -output


        m = np.shape(batch_x)[0] # number of training samples

        # TODO: ~ include case y != None
        for i in range(m): # run trough each training example and compute gradient
            Deltas = self.back_pro_single(x=batch_x[i,:], error=lambda x: error_function(x, batch_y))
            # clarification: here we want y to be the 'game' outcome which is the same for one batch of x (states)
            # since the batch represents one game
            Grad_1 = Grad_1 + Deltas['D1']
            Grad_2 = Grad_2 + Deltas['D2']

        Grad_1 = (1/float(m))*Grad_1
        Grad_2 = (1/float(m))*Grad_2

        # TODO: include regularization / !!care about bias terms!!

        self.Theta_1 = self.Theta_1 + Grad_1
        self.Theta_2 = self.Theta_2 + Grad_2
        #print("batch backprop: done")

        if self.save:
            self.save_weights()


    def save_weights(self):
        file = open('save.p', 'wb')
        pickle.dump({'T1':self.Theta_1, 'T2':self.Theta_2}, file)

    def load_weights(self):
        model = pickle.load(open('save.p', 'rb'))
        return model['T1'], model['T2']

    def get_move(self,x):
        prob, _, _, _ = self.for_prop_single(x)
        return np.argmax(prob) # get decision with hightest value

    def run_tests(self):
        pass

    def run_gradient_test(self):
        pass

    def plot_error(self):
        pass

    def cross_validation(self):
        pass

if __name__ == "__main__":
    test_game_winner = 2
    test_game = \
    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 0, 0, 1, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 0, 0, 1, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 2, 0, 1, 0, 0, 0, 1, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 2, 2, 0, 1, 0, 0, 0, 1, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 2, 2, 0, 1, 0, 0, 0, 1, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 2, 2, 0, 1, 1, 0, 0, 1, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 2, 2, 0, 1, 1, 0, 0, 1, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 2, 2, 0, 1, 1, 0, 0, 1, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 2, 2, 0, 1, 1, 2, 0, 1, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 2, 2, 0, 1, 1, 2, 0, 1, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 2, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 2, 2, 0, 1, 1, 2, 0, 1, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 2, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 2, 2, 0, 1, 1, 2, 0, 1, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 2, 0, 0, 1, 0, 0, 1, 1, 0, 2, 1, 1, 0, 2, 2, 0, 1, 1, 2, 0, 1, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 2, 0, 0, 1, 0, 0, 1, 1, 0, 2, 1, 1, 0, 2, 2, 0, 1, 1, 2, 1, 1, 2],
    [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 2, 0, 0, 1, 0, 0, 1, 1, 0, 2, 1, 1, 0, 2, 2, 0, 1, 1, 2, 1, 1, 2],
    [0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 2, 0, 0, 1, 0, 0, 1, 1, 0, 2, 1, 1, 0, 2, 2, 0, 1, 1, 2, 1, 1, 2],
    [0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 2, 0, 0, 2, 0, 0, 2, 2, 0, 0, 1, 0, 0, 1, 1, 0, 2, 1, 1, 0, 2, 2, 0, 1, 1, 2, 1, 1, 2],
    [0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 2, 0, 0, 2, 0, 0, 2, 2, 0, 1, 1, 0, 0, 1, 1, 0, 2, 1, 1, 0, 2, 2, 0, 1, 1, 2, 1, 1, 2],
    [0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 2, 0, 2, 2, 0, 0, 2, 2, 0, 1, 1, 0, 0, 1, 1, 0, 2, 1, 1, 0, 2, 2, 0, 1, 1, 2, 1, 1, 2],
    [0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 1, 2, 0, 2, 2, 0, 0, 2, 2, 0, 1, 1, 0, 0, 1, 1, 0, 2, 1, 1, 0, 2, 2, 0, 1, 1, 2, 1, 1, 2],
    [0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 1, 2, 0, 2, 2, 0, 0, 2, 2, 0, 1, 1, 2, 0, 1, 1, 0, 2, 1, 1, 0, 2, 2, 0, 1, 1, 2, 1, 1, 2],
    [0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 1, 2, 0, 2, 2, 0, 0, 2, 2, 0, 1, 1, 2, 0, 1, 1, 0, 2, 1, 1, 1, 2, 2, 0, 1, 1, 2, 1, 1, 2],
    [0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 1, 2, 0, 2, 2, 2, 0, 2, 2, 0, 1, 1, 2, 0, 1, 1, 0, 2, 1, 1, 1, 2, 2, 0, 1, 1, 2, 1, 1, 2],
    [0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 1, 2, 0, 2, 2, 2, 0, 2, 2, 0, 1, 1, 2, 0, 1, 1, 0, 2, 1, 1, 1, 2, 2, 1, 1, 1, 2, 1, 1, 2],
    [0, 0, 0, 0, 0, 2, 2, 0, 0, 1, 0, 0, 1, 2, 0, 2, 2, 2, 0, 2, 2, 0, 1, 1, 2, 0, 1, 1, 0, 2, 1, 1, 1, 2, 2, 1, 1, 1, 2, 1, 1, 2],
    [0, 0, 0, 0, 0, 2, 2, 0, 1, 1, 0, 0, 1, 2, 0, 2, 2, 2, 0, 2, 2, 0, 1, 1, 2, 0, 1, 1, 0, 2, 1, 1, 1, 2, 2, 1, 1, 1, 2, 1, 1, 2],
    [0, 0, 0, 0, 0, 2, 2, 0, 1, 1, 0, 0, 1, 2, 0, 2, 2, 2, 0, 2, 2, 0, 1, 1, 2, 2, 1, 1, 0, 2, 1, 1, 1, 2, 2, 1, 1, 1, 2, 1, 1, 2]]

    test_game = np.array(test_game)

    #print(test_game[1,:].reshape((42,1)))

    # initialize Neuronal Net with 42 input (43 with bias),
    # 7 hidden (8 with bias) and 7 output neurons
    net = NeuroNet(42,7,7,save_weights=True,load_weights=True)
    net.train_multi(test_game,test_game_winner)


