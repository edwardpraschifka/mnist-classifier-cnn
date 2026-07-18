# softmax + cross-entropy

import numpy as np

from utils import softmax

def loss(Y: np.ndarray, target: np.ndarray):
    """Computes softmax + cross-entropy loss between Y and Y_true"""

    Y_softmax = softmax(Y)
    target_softmax = softmax(target)

    correct_cat = np.argmax(target_softmax, axis=1)

    loss_by_row = [-np.log(row[i]) for (row,i) in zip(Y_softmax,correct_cat)]

    return np.mean(loss_by_row)

def loss_grad(Y: np.ndarray, target: np.ndarray):
    """Returns the derivative of the loss with respect
    to Y"""

    return softmax(Y) - target



