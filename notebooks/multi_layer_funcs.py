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

def weighted_bce(y_true, y_pred, pos_weight=1.0, neg_weight=1.0, epsilon=1e-8):
    """
    Computes eighted binary cross-entropy loss.
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

