import numpy as np

def sigmoid(x):
    """Computes the sigmoid of x
    """

    return 1 / (1+np.exp(-x))

def relu(x):
    """Computes the ReLU of x
    """
    return np.maximum(0, x)

def relu_prime(x):
    """Computes derivative of ReLU
    """

    return x > 0

def bce(*, ground_truths, predicted_probabilities):
    """Computes the binary cross-entropy loss
    """
    epsilon = 1e-8
    loss = -(ground_truths*np.log(predicted_probabilities + epsilon) + 
             (1-ground_truths)*np.log(1-predicted_probabilities + epsilon))

    return loss

def weighted_bce(y_true, y_pred, pos_weight=1.0, neg_weight=1.0, epsilon=1e-8):
    """
    Computes the weighted binary cross-entropy loss.
    """
    
    y_true = y_true.reshape(-1, 1)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    loss = -(pos_weight * y_true * np.log(y_pred) +
             neg_weight * (1 - y_true) * np.log(1 - y_pred))
    
    return np.mean(loss)

def clip_gradients(gradients, max_norm = 5):
    """Clip gradients if the norm is too high
    """

    total_norm = 0

    for gradient in gradients:
        total_norm += np.sum(gradient ** 2)
    total_norm = np.sqrt(total_norm)

    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-8)
        gradients = [grad * scale for gradient in gradients]

    return gradients

def weighted_sum(fingerprints, weights, bias):
    """Computed the weighted sum of the fingerprint
    """
    fingerprint_matrix = np.stack(fingerprints)

    return np.dot(fingerprint_matrix, weights) + bias

def forward_pass(X, weights_1, bias_1, weights_2, bias_2):
    
    z1 = np.dot(X, weights_1) + bias_1
    a1 = relu(z1)

    z2 = np.dot(a1, weights_2) + bias_2
    a2 = sigmoid(z2)

    return z1, a1, z2, a2

def back_prop(a2, z2, a1, z1, weights_2, X, y):

    # Output layer
    dL_da2 = (a2 - y.reshape(-1,1)) / len(y)
    dL_dz2 = dL_da2

    dL_dweights_2 = a1.T @ dL_dz2
    dL_dbias_2 = np.sum(dL_dz2, axis=0)

    # Hidden layer
    dL_da1 = dL_dz2 @ (weights_2.T)
    dL_dz1 = dL_da1 * relu_prime(z1)

    dL_dweights_1 = X.T @ dL_dz1
    dL_dbias_1 = np.sum(dL_dz1, axis=0)

    return dL_dweights_1, dL_dbias_1, dL_dweights_2, dL_dbias_2

