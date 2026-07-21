# per-layer gradient checks against PyTorch

import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import Layer, ConvLayer
from utils import quick_conv2d

class TestBaseLayer:
    def test_instantiate(self):
        with pytest.raises(TypeError):
            my_layer_class = Layer()


class TestConvLayer:
    @pytest.mark.parametrize("batch_size", [1,2])
    @pytest.mark.parametrize("input_channels", [1,2])
    @pytest.mark.parametrize("output_channels", [1,2])
    def test_forward(self, batch_size: int, input_channels:int, output_channels: int):
        
        # create our convolutional layer
        np.random.seed(42)
        W = np.random.rand(output_channels, input_channels, 2, 2)
        B = np.random.rand(output_channels)
        my_conv = ConvLayer(W, B)

        # push random input through convolutional layer
        X = np.random.rand(batch_size, input_channels, 3, 3)
        Y = my_conv.forward(X)

        # create pytorch convolutional layer
        torch_conv = quick_conv2d(W, B)

        # push same input through pytorch layer
        torch_Y = torch_conv.forward(torch.tensor(X, dtype=torch.float32))

        # compare outputs
        assert np.allclose(torch_Y.detach().numpy(), Y)