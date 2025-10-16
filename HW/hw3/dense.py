import numpy as np
from layer import Layer

class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def forward(self, input):
        # TODO: apply linear transformation to the input. see ppt for equation. 
        pass

    def backward(self, output_gradient, learning_rate):
        # TODO: update the weights and bias
        pass



    