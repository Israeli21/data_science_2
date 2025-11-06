import numpy as np

def binary_cross_entropy(y_true, y_pred):
    # TODO: return the binary_cross_entropy loss

    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return np.mean(loss)

def binary_cross_entropy_prime(y_true, y_pred):
    # TODO: return the binary_cross_entropy_prime. 
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return (y_pred - y_true) / (y_pred * (1 - y_pred))