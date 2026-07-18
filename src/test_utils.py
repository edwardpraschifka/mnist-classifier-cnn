import numpy as np
import torch
import torch.nn as nn
import pytest

from utils import convolve_1c, convolve, softmax


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

    
@pytest.mark.parametrize("output_channels", [1,2])
@pytest.mark.parametrize("input_channels", [1,2])
@pytest.mark.parametrize("padding", [0,1])
@pytest.mark.parametrize("stride", [1,2])
@pytest.mark.parametrize("kernel_size", [2,3])
def test_conv(output_channels: int, input_channels: int, padding: int, stride: int, kernel_size: int):
    np.random.seed(42)
    X = np.random.rand(input_channels,kernel_size+1, kernel_size+1)
    W = np.random.randn(output_channels,input_channels,kernel_size,kernel_size)

    Y = convolve(W, X, padding, stride)
    
    torch_conv = nn.Conv2d(input_channels, output_channels, kernel_size, 
                           padding=padding, stride=stride, bias=False)
    torch_X = torch.tensor(X, dtype=torch.float32)
    torch_W = torch.tensor(W, dtype=torch.float32)

    with torch.no_grad():
            torch_conv.weight.copy_(torch_W)
    
    Y_actual = torch_conv.forward(torch_X)

    assert np.allclose(Y, Y_actual.detach().numpy())


def test_softmax():
    X = torch.randn(3, 5)

    softmax_X = softmax(X.detach().numpy())
    torch_softmax_X = torch.softmax(X, dim=-1)

    assert np.allclose(softmax_X, torch_softmax_X)