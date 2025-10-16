import numpy as np
from scipy import signal
from layer import Layer

class Convolutional(Layer):
    def __init__(self, input_shape, kernel_size, depth):
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)

    def forward(self, input):
        self.input = input
        self.output = None
        # TODO: Implement the forward method using the formula provided in the powerpoint. 
        # You may add or remove any variables that you wish. 
        return self.output

    def backward(self, output_gradient, learning_rate):
        # TODO: initialize the kernels_gradient and input_gradient.
        kernels_gradient = None
        input_gradient = None

        # TODO: implement the back pass here. The equations in the ppt may help, but you're free to
        # add as much or as little code as you'd like. 

        # TODO: update the kernels and biases
        self.kernels -= None
        self.biases -= None

        return input_gradient
