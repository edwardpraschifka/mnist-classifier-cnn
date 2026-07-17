# softmax + cross-entropy

import numpy as np

def softmax(X: np.ndarray):
    """Computes softmax for each row of X"""

    e_X = np.exp(X)
    return e_X/np.sum(e_X, axis=1, keepdims=True)

def cross_ent(Y: np.ndarray, Y_true: np.ndarray):
    """Computes cross-entropy loss between Y and Y_true"""

    correct_cat = np.argmax(Y_true, axis=1)

    return [-np.log(row[i]) for (row,i) in zip(Y,correct_cat)]

def softmax_cross_ent_grad(Y: np.ndarray, Y_true: np.ndarray):
    """Returns the derivative of the loss with respect
    to Y"""

    return softmax(Y) - Y_true



