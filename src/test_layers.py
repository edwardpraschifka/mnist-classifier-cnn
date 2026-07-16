# per-layer gradient checks against PyTorch

import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import Layer, ConvLayer
from utils import convolve

class TestBaseLayer:
    def test_instantiate(self):
        with pytest.raises(TypeError):
            my_layer_class = Layer()


class TestConvLayer:
    @pytest.mark.parametrize("output_channels", [1,2,3,4])
    def test_forward(self, output_channels: int):
        np.random.seed(42)
        input_channels, padding, stride = 2, 0, 1
        X = np.random.rand(input_channels,3,3)
        W = np.random.rand(output_channels, input_channels, 2, 2)
        B = np.random.rand(output_channels)

        Y_expected = convolve(W, X, padding, stride) + B[:, None, None]

        my_conv = ConvLayer(W, B, stride, padding)
        Y = my_conv.forward(X)

        assert np.allclose(Y_expected, Y)