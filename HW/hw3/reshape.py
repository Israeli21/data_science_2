import numpy as np
from layer import Layer

class Reshape(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input):
        # TODO: reshape the input to the output_shape and return it.
        pass

    def backward(self, output_gradient, learning_rate):
        # TODO: reshape the output to the input_shape and return it.
        pass
