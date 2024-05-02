import numpy as np
from numpy import ndarray
from random import gauss
from numpy.random import rand
from math import sqrt

class Layer:
    def __init__(self, index, sizes) -> None:
        self.size = sizes[index]
        self.size_next = sizes[index+1] if index < len(sizes) - 1 else None
        self.activations: ndarray = self.init_activations()
        self.weights: ndarray = self.new_weights() if self.size_next else None
        self.biases: ndarray = self.new_biases() if self.size_next else None
    
    def init_activations(self):
        return np.array([0 for _ in range(self.size)], dtype=np.float16).reshape(-1,1)
    
    def new_weights(self):
        return np.array(
            [self.gauss_list(self.size) for _ in range(self.size_next)],
            dtype=np.float16)
    
    def new_biases(self):
        return np.array(self.gauss_list(self.size_next)).reshape(-1,1)
    
    def gauss_list(self, size):
        return [gauss(0, 1) for _ in range(size)]