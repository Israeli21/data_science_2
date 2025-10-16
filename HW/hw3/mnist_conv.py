import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils

from dense import Dense
from convolutional import Convolutional
from reshape import Reshape
from activations import Tanh, Sigmoid
from losses import binary_cross_entropy, binary_cross_entropy_prime

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
    y = np_utils.to_categorical(y)
    y = y.reshape(len(y), 2, 1)
    return x, y

# load MNIST from server, limit to 100 images per class since we're not training on GPU
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 100)
x_test, y_test = preprocess_data(x_test, y_test, 100)

# TODO: Add our layers and the flow of input into this list. 
network = []

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
        error += None

        # TODO: perform back prop 

    error /= len(x_train)
    print(f"{e + 1}/{epochs}, error={error}")

# TODO: run the test data through and print out your predictions
for x, y in zip(x_test, y_test):
    pass
