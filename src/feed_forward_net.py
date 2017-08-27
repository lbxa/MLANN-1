'''
    Feed-Foward Artificial Neural Network
    -------------------------------------
    Feed-foward nets are the simplest NN's to master. They are comprised of
    an input layer, one or more hidden layers and an output layer. Since the
    dimensionality of out data is 2 to 1, there will be 2 input neruons and
    a single output neuron. For sake of simplicty this net will restrict 
    itself to a single hidden layer (deep belief networks can be for later).

    This model is based on estimating your SAT score based on the amount of
    hours you slept and the amount of hours you studied the night before
'''

import numpy as np

class FFN(object):
    
    def __init__(self):
        # define hyperparameters
        self.input_layer_size = 2
        self.hidden_layer_size = 3
        self.ouput_layer_size = 1

        # define parameters
        self.W1 = np.random.randn(self.input_layer_size, self.hidden_layer_size)
        self.W2 = np.random.randn(self.hidden_layer_size, self.ouput_layer_size)

    def forward(self, X):
        # forward propagation
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.relu(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        prediction = self.sigmoid(self.z3)
        return prediction

    # define activation functions

    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    def relu(self, z):
        return np.maximum(z, 0)

    def scale_data(self, x, y):
        # account for difference in units
        MAX_SCORE = 100
        x = x/np.amax(x, axis=0)
        y /= MAX_SCORE

if __name__ == "__main__":

    data_set = np.array(([3,5],[5,1],[10,2]), dtype=float)
    labels   = np.array(([75],[82],[93]), dtype=float)

    my_nn = FFN()
    y_hat = my_nn.forward(data_set)

    print(labels)
    print(y_hat)


