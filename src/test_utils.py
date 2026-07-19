import numpy as np
import torch
import torch.nn as nn
import pytest

from utils import convolve_1c, convolve, softmax, quick_conv2d


def test_convolve_1c():
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

@pytest.mark.parametrize("output_channels", [1,2])
@pytest.mark.parametrize("input_channels", [1,2])
@pytest.mark.parametrize("padding", [0,1])
@pytest.mark.parametrize("stride", [1,2])
@pytest.mark.parametrize("kernel_size", [2,3])
@pytest.mark.parametrize("x_size", [4])
def test_convolve(output_channels: int, input_channels: int, padding: int, stride: int, kernel_size: int, x_size: int):
    np.random.seed(42)
    W = np.random.rand(output_channels,input_channels,kernel_size,kernel_size)
    X = np.random.rand(input_channels, x_size, x_size)
    B = np.zeros(output_channels)

    torch_conv = quick_conv2d(W, B, padding, stride)
    
    torch_Y = torch_conv.forward(torch.tensor(X, dtype=torch.float32))
    Y = convolve(W, X, padding, stride)

    assert np.allclose(Y, torch_Y.detach())


def test_softmax():
    X = torch.randn(3, 5)

    softmax_X = softmax(X.detach().numpy())
    torch_softmax_X = torch.softmax(X, dim=-1)

    assert np.allclose(softmax_X, torch_softmax_X)