import numpy as np
import torch
import torch.nn as nn
import pytest

from utils import convolve_1c, convolve, softmax, quick_conv2d

@pytest.mark.parametrize("kernel_size", [2,3])
@pytest.mark.parametrize("stride", [1,2])
@pytest.mark.parametrize("x_size", [2,3,4])
def test_convolve_1c(kernel_size: int, stride: int, x_size: int):

    # create our convolutional layer
    np.random.seed(42)
    K = np.random.rand(kernel_size, kernel_size)
    B = np.zeros(1)

    # create a random input X
    x_size = kernel_size+1
    X = np.random.rand(x_size, x_size)

    if kernel_size > x_size:
        with pytest.raises(ValueError):
            Y = convolve_1c(K,X, stride)
    
    # push X through convolutional layer      
    Y = convolve_1c(K,X, stride)

    # create pytorch convolutional layer
    torch_conv = quick_conv2d(K[None, None, :], B, stride=stride)

    # push X through pytorch layer
    torch_Y = torch_conv.forward(torch.tensor(X[None, :], dtype=torch.float32))

    # compare outputs
    assert np.allclose(torch_Y.detach().numpy(), Y)

@pytest.mark.parametrize("batch_size", [1,2])
@pytest.mark.parametrize("output_channels", [1,2])
@pytest.mark.parametrize("input_channels", [1,2])
@pytest.mark.parametrize("padding", [0,1])
@pytest.mark.parametrize("stride", [1])
@pytest.mark.parametrize("kernel_size", [2])
@pytest.mark.parametrize("x_size", [3])
def test_convolve(batch_size:int, output_channels: int, input_channels: int, padding: int, stride: int, kernel_size: int, x_size: int):

    # create our convolutional layer
    np.random.seed(42)
    W = np.random.rand(output_channels,input_channels,kernel_size,kernel_size)
    B = np.zeros(output_channels)

    # Create random input X
    X = np.random.rand(batch_size, input_channels, x_size, x_size)

    # push X through our convolutional layer
    Y = convolve(W, X, padding, stride)

    # create pytorch convolutional layer
    torch_conv = quick_conv2d(W, B, padding, stride)
    
    # push X through pytorch layer
    torch_Y = torch_conv.forward(torch.tensor(X, dtype=torch.float32))
    
    # compare outputs
    assert np.allclose(Y, torch_Y.detach())


@pytest.mark.parametrize("Y_height", [2,3])
@pytest.mark.parametrize("Y_width", [2,3])
def test_softmax(Y_height: int, Y_width: int):
    Y = np.random.rand(Y_height, Y_width)

    softmax_Y = softmax(Y)
    torch_softmax_Y = torch.softmax(torch.tensor(Y), dim=-1)

    assert np.allclose(softmax_Y, torch_softmax_Y)