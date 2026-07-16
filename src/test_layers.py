# per-layer gradient checks against PyTorch

import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import Layer, ConvLayer

class TestBaseLayer:
    def test_instantiate(self):
        with pytest.raises(TypeError):
            my_layer_class = Layer()


@pytest.mark.parametrize("output_channels", [1,2])
@pytest.mark.parametrize("input_channels", [1,2])
@pytest.mark.parametrize("padding", [0,1])
@pytest.mark.parametrize("stride", [1,2])
def test_forward(output_channels: int, input_channels: int, padding: int, stride: int):
    torch_conv = nn.Conv2d(input_channels, 
                        output_channels, 
                        kernel_size=(2,2), 
                        stride=stride, 
                        padding=padding)
    
    np.random.seed(42)
    X = np.random.rand(input_channels,3,3)
                
    X_torch = torch.tensor(X, dtype=torch.float32)

    W = np.random.rand(output_channels, input_channels, 2, 2)

    W_torch = torch.tensor(W, dtype=torch.float32)

    B = np.random.rand(output_channels)
    
    B_torch = torch.tensor(B, dtype=torch.float32)

    with torch.no_grad():
        torch_conv.weight.copy_(W_torch)
        torch_conv.bias.copy_(B_torch)
    
    Y_torch = torch_conv.forward(X_torch)

    my_conv = ConvLayer(W, B, stride, padding)
    Y = my_conv.forward(X)
    
    assert np.allclose(Y_torch.detach().numpy(), Y)