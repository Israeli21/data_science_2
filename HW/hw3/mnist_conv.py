import numpy as np
from keras.datasets import mnist
# Removed np_utils import - we'll implement to_categorical manually

from dense import Dense
from convolutional import Convolutional
from reshape import Reshape
from activations import Tanh, Sigmoid
from losses import binary_cross_entropy, binary_cross_entropy_prime

def to_categorical(y, num_classes=None):
    """Convert class vector to binary class matrix (one-hot encoding)"""
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=np.float32)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

def preprocess_data(x, y, limit):
    ''' 
    Will limit our data since using the whole thing will take forever on a cpu especially since we're
    implementing this from scratch.
    '''
    zero_index = np.where(y == 0)[0][:limit]
    one_index = np.where(y == 1)[0][:limit]
    all_indices = np.hstack((zero_index, one_index))
    all_indices = np.random.permutation(all_indices)
    x, y = x[all_indices], y[all_indices]
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255
    y = to_categorical(y)
    y = y.reshape(len(y), 2, 1)
    return x, y

# load MNIST from server, limit to 100 images per class since we're not training on GPU
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 100)
x_test, y_test = preprocess_data(x_test, y_test, 100)

# TODO: Add our layers and the flow of input into this list. 
# Network architecture: Conv → Sigmoid → Reshape → Dense → Sigmoid
# Input: (1, 28, 28) - single channel 28x28 image
# Conv: 3x3 kernel, 1 output channel -> output: (1, 26, 26)
# Reshape: (1, 26, 26) -> (676, 1) column vector
# Dense: 676 inputs -> 1 output (binary classification)
network = [
    Convolutional(input_shape=(1, 28, 28), kernel_size=3, depth=1),  # Conv layer
    Sigmoid(),                                                         # Activation
    Reshape(input_shape=(1, 26, 26), output_shape=(676, 1)),         # Reshape to column vector
    Dense(input_size=676, output_size=1),                             # Dense layer
    Sigmoid()                                                          # Final activation
]

epochs = 10
learning_rate = 0.1

# train
for e in range(epochs):
    error = 0
    for x, y in zip(x_train, y_train):
        # forward
        output = x
        for layer in network:
            output = layer.forward(output)

        # TODO: update our error
        # Calculate error using binary cross entropy
        # For binary classification, we only need the positive class (index 1)
        y_binary = y[1, 0]  # Extract the positive class label
        output_binary = output[0, 0]  # Extract the sigmoid output
        
        error += binary_cross_entropy(y_binary, output_binary)

        # TODO: perform back prop
        # Perform backpropagation
        # Create gradient with same shape as output
        gradient = np.zeros_like(output)
        gradient[0, 0] = binary_cross_entropy_prime(y_binary, output_binary)
        for layer in reversed(network):
            gradient = layer.backward(gradient, learning_rate) 

    error /= len(x_train)
    print(f"{e + 1}/{epochs}, error={error}")

# TODO: run the test data through and print out your predictions
print("\nTesting the network:")
correct = 0
total = 0

for x, y in zip(x_test, y_test):
    # Forward pass
    output = x
    for layer in network:
        output = layer.forward(output)
    
    # Get prediction (0 or 1)
    prediction = 1 if output[0, 0] > 0.5 else 0
    true_label = 1 if y[1, 0] > 0.5 else 0  # y is one-hot encoded
    
    if prediction == true_label:
        correct += 1
    total += 1
    
    print(f"Predicted: {prediction}, True: {true_label}, Confidence: {output[0, 0]:.3f}")

accuracy = correct / total
print(f"\nTest Accuracy: {accuracy:.3f} ({correct}/{total})")
