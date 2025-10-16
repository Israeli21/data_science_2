import numpy as np
from activation import Activation

class Tanh(Activation):
    ''' 
    Optional Tanh function if you'd like to try alternatives and see what happens.
    '''
    def __init__(self):
        def tanh(x):
            return np.tanh(x)

        def tanh_prime(x):
            return 1 - np.tanh(x) ** 2

        super().__init__(tanh, tanh_prime)

class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            # TODO: return the sigmoid of x
            pass

        def sigmoid_prime(x):
            # TODO: return the derivative
            pass

        super().__init__(sigmoid, sigmoid_prime)


