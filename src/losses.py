# softmax + cross-entropy

import numpy as np

from utils import softmax

def loss(Y: np.ndarray, target: np.ndarray):
    """Computes average softmax + cross-entropy loss between 
    Y and Y_true"""

    Y_softmax = softmax(Y)
    target_softmax = softmax(target)

    correct_cat = np.argmax(target_softmax, axis=1)

    # each row is a separate training example,
    # so we calculate its loss by doing \sum t_i log(y_i),
    # which is equivalent to \sum t_c log(y_c) where
    # c is the correct class

    loss_by_row = [-np.log(row[i]) for (row,i) in zip(Y_softmax,correct_cat)]

    return np.mean(loss_by_row)

def loss_grad(Y: np.ndarray, target: np.ndarray):
    """Returns the derivative of the loss with respect
    to Y"""

    # softmax(Y) - target returns a matrix whose
    # i-th row is dL_i/dy_ij

    # But in our tests, PyTorch returns a 
    # matrix whose i-th row is
    # dL_mean/dy_ij = d(L_1/n+L_2/n+...)/dy_ij
    #              = d(L_i/n)/dy_ij
    #              = 1/n * d(L_i)/dy_ij

    # we correct this by dividing the final result
    # by n_rows
    
    return (softmax(Y) - target)/Y.shape[0]



