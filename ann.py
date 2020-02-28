from math import exp

import numpy as np

def sigmoid(x):
    return 1.0/(1.0 + exp(-x))

class ANN:
    def __init__(self, sizes):
            self.num_layers = len(sizes)
            self.sizes = sizes # Number of neurons in each layer
            # Initializing biases and weights randomly using Gaussin distribution
            # biases[0] -> 
            # w := weights[0] -> Array fo weights between first and second layer
            # w[j, k] -> weight for connection kth neuron on first layer and jth neuron on second layer.   k --> j
            self.biases = [np.random.rann(r, 1) for r in sizes[1:]] # Every layer has a bias except input layer
            self.weights = [np.random.randn(y, x)
                            for x, y in zip(sizes[:-1], sizes[1:]] # weight[0] stores connection between first and second layer
    
    def feed_forward(self, a):
        for b, w in zip(self.biases, self. weights):
            a = sigmoid(w @ a + b)
        return a

    def SGD(self, training_data, emini_batch_size, epochs, eta, verbose=True, test_data=None, exclusion_list=None):
        # training_data -> list of tuples (x, y)
        if test_data: n_test = len(test_data)
        # Looping over epochs
        for e in range(epochs):
            random.shuffle(training_data)
            # Generating mini_batches by spliting training_data
            mini_batches = [
                training_data[k: k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            # updating by gradient descend
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, exclusion_list)

        if verbose:
            print("Epoch {0}".format(e))

    def update_mini_batch(self, mini_batch, eta, exclustion_list):
        """
        :param mini_batch (list of tuples): mini_batch of test_data
        :param eta (int): rate of learning 
        :returns: null
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeors(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backpropagate(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                        for b, nb in zip(self.biases, nabla_b)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w])]
    
    def backpropagate(self, x, y):
        pass
