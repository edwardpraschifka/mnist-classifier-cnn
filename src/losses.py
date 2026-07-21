# softmax + cross-entropy

"""Defines functions for i) computing the loss between the network's output
and a target output, and ii) the gradient of this loss."""

import numpy as np

from utils import softmax

def loss(Y: np.ndarray, target: np.ndarray):
    """Computes average softmax + cross-entropy loss between Y and target.

    Applies row-wise softmax to both Y and target, then for each row takes
    the negative log-probability that Y_softmax assigns to target's argmax
    class, and averages this over all rows. This matches the behavior of
    `torch.nn.CrossEntropyLoss` when given raw logits and either one-hot or
    class-probability targets.

    Args:
        Y: A 2D array of shape (batch_size, num_classes) containing raw
           logits, where each row is one example's per-class scores.
        target: A 2D array of shape (batch_size, num_classes) containing
           the target distribution for each example (e.g. one-hot labels).
           Only the argmax class of each row is used.

    Returns:
        A scalar (float) equal to the mean cross-entropy loss across all
        rows of Y.

    Example:
        >>> Y = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
        >>> target = np.array([[0, 0, 1], [1, 0, 0]])
        >>> loss(Y, target)
        0.7531091265562451
    """

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
    """Computes the gradient of the mean loss (see `loss`) with respect to Y.

    Args:
        Y: A 2D array of shape (batch_size, num_classes) containing raw
           logits, where each row is one example's per-class scores.
        target: A 2D array of shape (batch_size, num_classes) containing
           the target distribution for each example (e.g. one-hot labels).

    Returns:
        A 2D array of the same shape as Y, where entry (i, j) is
        d(mean loss)/dY[i, j].

    Example:
        >>> Y = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
        >>> target = np.array([[0, 0, 1], [1, 0, 0]])
        >>> loss_grad(Y, target)
        array([[ 0.04501529,  0.12236424, -0.16737952],
               [-0.33333333,  0.16666667,  0.16666667]])
    """

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



