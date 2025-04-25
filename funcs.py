import numpy as np

def sigmoid(x):
    """Computes the sigmoid of x
    """

    return 1 / (1+np.exp(-x))

def bce(*, ground_truths, predicted_probabilities):
    """Computes the binary cross-entropy loss
    """
    epsilon = 1e-8
    loss = -(ground_truths*np.log(predicted_probabilities + epsilon) + 
             (1-ground_truths)*np.log(1-predicted_probabilities + epsilon))

    return loss

def weighted_sum(fingerprints, weights, bias):
    """Computed the weighted sum of the fingerprint
    """
    fingerprint_matrix = np.stack(fingerprints)

    return np.dot(fingerprint_matrix, weights) + bias

def relu(x):
    """Computes the ReLU of x
    """
    return np.maximum(0, x)
