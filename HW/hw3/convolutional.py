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
        # TODO: Implement the forward method using the formula provided in the powerpoint
        # You may add or remove any variables that you wish
        
        # Input shape: (input_depth, input_height, input_width)
        input_depth, input_height, input_width = self.input.shape
        output_depth, output_height, output_width = self.output_shape

        # Initialize output with biases
        output = np.copy(self.biases)

        # Compute cross-correlation for each output channel (depth)
        for d in range(output_depth):
            acc = np.zeros((output_height, output_width))
            for c in range(input_depth):
                # signal.correlate performs cross-correlation
                acc += signal.correlate(self.input[c], self.kernels[d, c], mode='valid')
            output[d] += acc

        self.output = output
        return self.output

    def backward(self, output_gradient, learning_rate): 

        # TODO: implement the back pass here. The equations in the ppt may help, but you're free to
        # add as much or as little code as you'd like.
        kernels_gradient = np.zeros_like(self.kernels)
        input_gradient = np.zeros_like(self.input)
        
        # Get shapes
        input_depth, input_height, input_width = self.input.shape
        output_depth, output_height, output_width = self.output_shape
        kernel_size = self.kernels.shape[3]  # Assuming square kernels
        
        # 1. Compute gradient with respect to biases: ∂E/∂b = output_gradient
        # (biases gradient is just the output gradient)
        
        # 2. Compute gradient with respect to kernels: ∂E/∂k
        # For each output channel and input channel
        for d in range(output_depth):
            for c in range(input_depth):
                # Cross-correlate input with output gradient to get kernel gradient
                kernels_gradient[d, c] = signal.correlate(self.input[c], output_gradient[d], mode='valid')
        
        # TODO: implement the back pass here. The equations in the ppt may help, but you're free to
        # add as much or as little code as you'd like.
        # 3. Compute gradient with respect to input: ∂E/∂X
        # For each input channel
        for c in range(input_depth):
            acc = np.zeros((input_height, input_width))
            for d in range(output_depth):
                # Cross-correlate output gradient with flipped kernel
                flipped_kernel = np.flip(np.flip(self.kernels[d, c], axis=0), axis=1)
                acc += signal.correlate(output_gradient[d], flipped_kernel, mode='full')
            input_gradient[c] = acc
        
        # TODO: update the kernels and biases
        # Update parameters using gradient descent
        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient

        return input_gradient
