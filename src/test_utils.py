import numpy as np
import torch
import torch.nn as nn

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
    np.random.seed(42)
    X = np.random.rand(2,3,3)
    W = np.random.randn(3,2,2,2)

    Y = convolve(W,X)
    
    torch_conv = nn.Conv2d(2, 3, 2, bias=False)
    torch_X = torch.tensor(X, dtype=torch.float32)
    torch_W = torch.tensor(W, dtype=torch.float32)

    with torch.no_grad():
            torch_conv.weight.copy_(torch_W)
    
    Y_actual = torch_conv.forward(torch_X)

    assert np.allclose(Y, Y_actual.detach().numpy())