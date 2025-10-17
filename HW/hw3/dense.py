import numpy as np
from layer import Layer

class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def forward(self, input):
        # TODO: apply linear transformation to the input. see ppt for equation. 
        self.input = input
        # Apply linear transformation: y = xA^T + b
        # input shape: (input_size, 1) or (batch_size, input_size)
        # weights shape: (output_size, input_size)
        # bias shape: (output_size, 1)
        self.output = np.dot(self.weights, input) + self.bias
        return self.output

    def backward(self, output_gradient, learning_rate):
        # TODO: update the weights and bias
        weights_gradient = np.dot(output_gradient, self.input.T)
        
        # ∂E/∂b = output_gradient (gradient w.r.t. bias is just the output gradient)
        bias_gradient = output_gradient
        
        # ∂E/∂x = W^T * output_gradient (gradient w.r.t. input)
        input_gradient = np.dot(self.weights.T, output_gradient)
        
        # Update weights and bias using gradient descent
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * bias_gradient
        
        return input_gradient