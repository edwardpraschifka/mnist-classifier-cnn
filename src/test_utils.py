import numpy as np

from utils import convolve_1c

def test_conv_1c():
    X = np.array([
        [1,0,1],
        [-1,1,0],
        [-1,-1,1]
    ])

    K = np.array([
        [2,1],
        [0,3]
    ])

    Y = convolve_1c(K,X)
    Y_actual = np.array([
        [5,1],
        [-4,5]
    ])

    assert np.array_equal(Y, Y_actual)
    
