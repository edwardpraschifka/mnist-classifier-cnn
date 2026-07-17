# softmax + cross-entropy

import numpy as np

def softmax(X: np.ndarray):
    """Computes softmax for each row of X"""

    e_X = np.exp(X)
    print(f"e_X = {e_X/np.sum(e_X)}")
    return e_X/np.sum(e_X, axis=1, keepdims=True)

def cross_ent():
    pass
