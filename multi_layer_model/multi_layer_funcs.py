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
    """Computes the weighted sum of the fingerprint
    """
    fingerprint_matrix = np.stack(fingerprints)

    return np.dot(fingerprint_matrix, weights) + bias

def relu(x):
    """Computes the ReLU of x
    """
    return np.maximum(0, x)

def relu_prime(x):
    """Computes derivative of ReLU
    """

    return x > 0

def weighted_bce(y_true, y_pred, pos_weight=1.0, neg_weight=1.0, epsilon=1e-8):
    """
    Computes eighted binary cross-entropy loss.
    """
    
    y_true = y_true.reshape(-1, 1)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    loss = -(pos_weight * y_true * np.log(y_pred) +
             neg_weight * (1 - y_true) * np.log(1 - y_pred))
    
    return np.mean(loss)

