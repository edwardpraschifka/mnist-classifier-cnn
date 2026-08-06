import numpy as np
import torch
import torch.nn as nn
import pytest

from utils import convolve_1c, convolve, softmax

@pytest.mark.parametrize("kernel_size", [2,3])
@pytest.mark.parametrize("stride", [1,2])
@pytest.mark.parametrize("x_size", [3,4])
def test_convolve_1c(kernel_size: int, stride: int, x_size: int):

    K = np.random.rand(kernel_size, kernel_size)
    X = np.random.rand(x_size, x_size)    
    Y = convolve_1c(K,X, stride)
    torch_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, stride=stride, bias=False)

    with torch.no_grad():
        torch_conv.weight.copy_(torch.tensor(K))

    torch_Y = torch_conv.forward(torch.tensor(X[None, :], dtype=torch.float32))
    assert np.allclose(torch_Y.detach().numpy(), Y)

@pytest.mark.parametrize("batch_size", [1,2])
@pytest.mark.parametrize("output_channels", [1,2])
@pytest.mark.parametrize("input_channels", [1,2])
@pytest.mark.parametrize("padding", [0,1])
@pytest.mark.parametrize("stride", [1])
@pytest.mark.parametrize("kernel_size", [2])
@pytest.mark.parametrize("x_size", [3])
def test_convolve(batch_size:int, output_channels: int, input_channels: int, padding: int, stride: int, kernel_size: int, x_size: int):

    W = np.random.rand(output_channels,input_channels,kernel_size,kernel_size)
    X = np.random.rand(batch_size, input_channels, x_size, x_size)
    Y = convolve(W, X, padding, stride)
    torch_conv = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, bias=False)
    
    with torch.no_grad():
        torch_conv.weight.copy_(torch.tensor(W))
    
    torch_Y = torch_conv.forward(torch.tensor(X, dtype=torch.float32))
    assert np.allclose(Y, torch_Y.detach())


@pytest.mark.parametrize("Y_height", [2,3])
@pytest.mark.parametrize("Y_width", [2,3])
def test_softmax(Y_height: int, Y_width: int):
    Y = np.random.rand(Y_height, Y_width)

    softmax_Y = softmax(Y)
    torch_softmax_Y = torch.softmax(torch.tensor(Y), dim=-1)

    assert np.allclose(softmax_Y, torch_softmax_Y)