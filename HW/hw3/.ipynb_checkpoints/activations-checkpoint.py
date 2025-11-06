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
            # Sigmoid function: 1 / (1 + e^(-x))
            # Clip x to prevent overflow
            x_clipped = np.clip(x, -500, 500)
            return 1 / (1 + np.exp(-x_clipped))

        def sigmoid_prime(x):
            # TODO: return the derivative
            # Derivative of sigmoid: sigmoid(x) * (1 - sigmoid(x))
            # Use the sigmoid function we defined above
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)


