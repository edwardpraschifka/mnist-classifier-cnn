import numpy as np

from utils import convolve_1c, convolve

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
    

def test_conv():
    X = np.zeros((2,3,3))
    X[0] = [[1,0,1],
           [-1,1,0],
           [-1,-1,1]]
    
    X[1] = [[0,0,1],
           [1,-1,1],
           [1,1,0]]

    W = np.zeros((3,2,2,2))

    W[0] = [[[1,2], [3,4]], [[4,3], [2,1]]]
    W[1] = [[[1,3], [2,4]], [[3,2], [1,4]]]
    W[2] = [[[2,1], [3,4]], [[1,2], [4,3]]]

    Y = convolve(W,X)
    Y_actual = np.zeros((3,2,2))

    Y_actual[0] = [[ 3,  7],
            [-2,  3]]

    Y_actual[1] = [[ 0, 10],
            [ 2,  3]]

    Y_actual[2] = [[ 4,  5],
            [-2,  8]]
    
    assert np.array_equal(Y, Y_actual)