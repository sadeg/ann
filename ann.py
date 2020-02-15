from math import exp

import numpy as np

def sigmoid(x):
    return 1.0/(1.0 + exp(-x))

class ANN:
    def __init__(self, sizes):
            self.num_layers = len(sizes)
            self.sizes = sizes # Number of neurons in each layer
            # Initializing biases and weights randomly using Gaussin distribution
            self.biases = [np.random.rann(r, 1) for r in sizes[1:]] # Every layer has a bias except input layer
            self.weights = [np.random.randn(r, c)
                            for r, c in zip(sizes[:-1], sizes[1:]] # weight[0] stores connection between first and second layer
    
    def feed_forward(self, a):
        for b, w in zip(self.biases, self. weights):
            a = sigmoid(w @ a + b)
        return a

    
