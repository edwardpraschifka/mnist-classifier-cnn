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


class TestConvLayer:
    def test_forward(self):
        torch_conv = nn.Conv2d(in_channels=1, 
                         out_channels=2, 
                         kernel_size=(2,2), 
                         stride=1, 
                         padding=0)

        X = np.array([
            [1,0,1],
            [0,1,1],
            [1,1,1]]
        ).reshape(1,3,3)
                    
        X_torch = torch.tensor(X, dtype=torch.float32).reshape(1,1,3,3)

        W = np.array([
            [
                [1,0],
                [1,0]
            ],
            [
                [0,1],
                [0,1]
            ],
        ]).reshape(2,1,2,2)

        W_torch = torch.tensor(W, dtype=torch.float32)
        
        B = np.array([0,0])
        B_torch = torch.tensor(B, dtype=torch.float32)

        with torch.no_grad():
            torch_conv.weight.copy_(W_torch)
            torch_conv.bias.copy_(B_torch)
        
        Y_torch = torch_conv.forward(X_torch)

        my_conv = ConvLayer(W,B)
        Y = my_conv.forward(X)

        assert np.array_equal(Y_torch[0].detach().numpy(), Y)