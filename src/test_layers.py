# per-layer gradient checks against PyTorch

import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import Layer, ConvLayer, ReluLayer, PoolLayer
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
        torch_X = torch.tensor(X, dtype=torch.float32)
        torch_Y = torch_conv.forward(torch_X)

        # compare outputs
        assert np.allclose(torch_Y.detach().numpy(), Y)


    @pytest.mark.parametrize("batch_size", [1,2])
    @pytest.mark.parametrize("input_channels", [1,2])
    @pytest.mark.parametrize("output_channels", [1,2])
    @pytest.mark.parametrize("kernel_size", [2])
    @pytest.mark.parametrize("X_size", [5])
    def test_backward(self, X_size, kernel_size, output_channels, input_channels, batch_size):

        # create our convolutional layer
        np.random.seed(42)
        W = np.random.rand(output_channels, input_channels, kernel_size, kernel_size).astype(np.float32)
        B = np.random.rand(output_channels).astype(np.float32)
        my_conv = ConvLayer(W, B)

        # push random input through convolutional layer
        X = np.random.rand(batch_size, input_channels, X_size, X_size).astype(np.float32)
        Y = my_conv.forward(X)

        # create pytorch convolutional layer
        torch_conv = quick_conv2d(W, B)

        # push same input through pytorch layer
        torch_X = torch.tensor(X, requires_grad=True)
        torch_Y = torch_conv.forward(torch_X)

        # generate random dL_dOut and calculate
        # gradients
        dL_dOut = np.random.rand(*torch_Y.shape)
        torch_dL_dOut = torch.tensor(dL_dOut, dtype=torch.float32)
        my_conv.backward(dL_dOut)
        torch_Y.backward(torch_dL_dOut)

        # compare outputs
        assert np.allclose(my_conv.dL_dX, torch_X.grad)
        assert np.allclose(my_conv.dL_dB, torch_conv.bias.grad)
        assert np.allclose(my_conv.dL_dW, torch_conv.weight.grad)


    @pytest.mark.parametrize("batch_size", [2])
    @pytest.mark.parametrize("input_channels", [2])
    @pytest.mark.parametrize("output_channels", [2])
    @pytest.mark.parametrize("kernel_size", [2])
    @pytest.mark.parametrize("X_size", [5])
    @pytest.mark.parametrize("learning_rate", [0.001,0.01])
    def test_update(self, learning_rate, X_size, kernel_size, output_channels, input_channels, batch_size):

        # create our convolutional layer
        np.random.seed(42)
        W = np.random.rand(output_channels, input_channels, kernel_size, kernel_size).astype(np.float32)
        B = np.random.rand(output_channels).astype(np.float32)
        my_conv = ConvLayer(W, B)

        # push random input through convolutional layer
        X = np.random.rand(batch_size, input_channels, X_size, X_size).astype(np.float32)
        Y = my_conv.forward(X)

        # create pytorch convolutional layer
        torch_conv = quick_conv2d(W, B)

        # push same input through pytorch layer
        torch_X = torch.tensor(X, requires_grad=True)
        torch_Y = torch_conv.forward(torch_X)

        # generate random dL_dOut and calculate
        # gradients
        dL_dOut = np.random.rand(*torch_Y.shape)
        torch_dL_dOut = torch.tensor(dL_dOut, dtype=torch.float32)
        my_conv.backward(dL_dOut)
        torch_Y.backward(torch_dL_dOut)

        # update
        with torch.no_grad():
            torch_conv.weight -= learning_rate * torch_conv.weight.grad
            torch_conv.bias -= learning_rate * torch_conv.bias.grad

        my_conv.update(learning_rate)

        # compare outputs
        assert np.allclose(my_conv.W, torch_conv.weight.detach().numpy())
        assert np.allclose(my_conv.B, torch_conv.bias.detach().numpy())


class TestReluLayer:

    @pytest.mark.parametrize("batch_size", [1,2])
    @pytest.mark.parametrize("input_channels", [1,2])
    @pytest.mark.parametrize("X_size", [5])
    def test_forward(self, X_size, input_channels, batch_size):

        # create our ReLU layer
        my_relu = ReluLayer()

        # push random input through ReLU layer
        X = np.random.rand(batch_size, input_channels, X_size, X_size).astype(np.float32)
        Y = my_relu.forward(X)

        # create pytorch convolutional layer
        torch_relu = nn.ReLU()

        # push same input through pytorch layer
        torch_X = torch.tensor(X)
        torch_Y = torch_relu.forward(torch_X)

        assert np.allclose(Y, torch_Y)


    @pytest.mark.parametrize("batch_size", [1,2])
    @pytest.mark.parametrize("input_channels", [1,2])
    @pytest.mark.parametrize("X_size", [5])
    def test_backward(self, X_size, input_channels, batch_size):

        # create our ReLU Layer
        my_relu = ReluLayer()

        # push random input through ReLU layer
        X = np.random.rand(batch_size, input_channels, X_size, X_size).astype(np.float32)
        Y = my_relu.forward(X)

        # create pytorch ReLU layer
        torch_relu = nn.ReLU()

        # push same input through pytorch layer
        torch_X = torch.tensor(X, requires_grad=True)
        torch_Y = torch_relu.forward(torch_X)

        # generate random dL_dOut and calculate
        # gradients
        dL_dOut = np.random.rand(*torch_Y.shape)
        torch_dL_dOut = torch.tensor(dL_dOut, dtype=torch.float32)
        my_relu.backward(dL_dOut)
        torch_Y.backward(torch_dL_dOut)

        # compare outputs
        assert np.allclose(my_relu.dL_dX, torch_X.grad)


class TestPoolLayer:

    @pytest.mark.parametrize("batch_size", [2])
    @pytest.mark.parametrize("input_channels", [2])
    @pytest.mark.parametrize("X_size", [4])
    @pytest.mark.parametrize("pool_size", [2])
    def test_forward(self, pool_size, X_size, input_channels, batch_size):
                
        # create our pooling layer
        my_pool = PoolLayer(pool_size)

        # push random input through pooling layer
        X = np.random.rand(batch_size, input_channels, X_size, X_size).astype(np.float32)
        Y = my_pool.forward(X)

        # create pytorch pooling layer
        torch_pool = nn.MaxPool2d(pool_size, return_indices=True)

        # push same input through pytorch layer
        torch_X = torch.tensor(X, requires_grad=True)
        (torch_Y, torch_argmax) = torch_pool.forward(torch_X)


        # compare returned y values
        assert np.allclose(Y, torch_Y.detach().numpy())
    
        # compared stored argmax masks
        assert np.allclose(my_pool.argmax_mask, torch_argmax)