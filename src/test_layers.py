# per-layer gradient checks against PyTorch

import pytest
import numpy as np
import torch
import torch.nn as nn

from layers import Layer, ConvLayer, ReluLayer, PoolLayer, FlattenLayer

class TestBaseLayer:
    def test_instantiate(self):
        with pytest.raises(TypeError):
            my_layer_class = Layer()


@pytest.fixture
def setup_layers(request):
    layer_type = request.param.get("layer_type", ConvLayer)
    batch_size = request.param.get("batch_size", 1)
    output_channels = request.param.get("output_channels", 1)
    input_channels = request.param.get("input_channels", 1)
    K_size = request.param.get("K_size", 3)
    X_size = request.param.get("X_size", 4)

    W = np.random.rand(output_channels, input_channels, K_size, K_size)
    B = np.random.rand(output_channels)
    X = np.random.rand(batch_size, input_channels, X_size, X_size)
    

    torch_W = torch.tensor(W, dtype=torch.float32, requires_grad=True)
    torch_B = torch.tensor(B, dtype=torch.float32, requires_grad=True)
    torch_X = torch.tensor(X, dtype=torch.float32, requires_grad=True)

    if layer_type == ConvLayer:
        my_layer = ConvLayer(W, B)
        torch_layer = nn.Conv2d(input_channels, output_channels, K_size)

        with torch.no_grad():
            torch_layer.weight.copy_(torch_W)
            torch_layer.bias.copy_(torch_B)

    if layer_type == ReluLayer:
        my_layer = ReluLayer()
        torch_layer = nn.ReLU()
    
    if layer_type == PoolLayer:
            my_layer = PoolLayer(K_size)
            torch_layer = nn.MaxPool2d(K_size, return_indices=True)

    if layer_type == FlattenLayer:
            my_layer = FlattenLayer()
            torch_layer = nn.Flatten()

    return my_layer, X, torch_layer, torch_X

class TestConvLayer:
    params = [{"layer_type": ConvLayer, "batch_size": 1, "output_channels": 1, "input_channels": 1, "K_size": 2, "X_size": 3},
                                            {"layer_type": ConvLayer, "batch_size": 2, "output_channels": 1, "input_channels": 2, "K_size": 2, "X_size": 3},
                                            {"layer_type": ConvLayer, "batch_size": 1, "output_channels": 2, "input_channels": 2, "K_size": 3, "X_size": 4},
                                            {"layer_type": ConvLayer, "batch_size": 2, "output_channels": 2, "input_channels": 2, "K_size": 3, "X_size": 4}]
    
    @pytest.mark.parametrize("setup_layers", params, indirect=True)
    def test_forward(self, setup_layers):

        my_layer, X, torch_layer, torch_X = setup_layers
        Y = my_layer.forward(X)
        torch_Y = torch_layer.forward(torch_X)
        assert np.allclose(torch_Y.detach().numpy(), Y)


    @pytest.mark.parametrize("setup_layers", params, indirect=True)
    def test_backward(self, setup_layers):

        my_layer, X, torch_layer, torch_X = setup_layers
        Y = my_layer.forward(X)
        torch_Y = torch_layer.forward(torch_X)

        dL_dOut = np.random.rand(*torch_Y.shape)
        torch_dL_dOut = torch.tensor(dL_dOut, dtype=torch.float32)

        my_layer.backward(dL_dOut)
        torch_Y.backward(torch_dL_dOut)

        assert np.allclose(my_layer.dL_dX, torch_X.grad)
        assert np.allclose(my_layer.dL_dB, torch_layer.bias.grad)
        assert np.allclose(my_layer.dL_dW, torch_layer.weight.grad)

    @pytest.mark.parametrize("learning_rate", [0.1, 0.01, 0.001])
    @pytest.mark.parametrize("setup_layers", [{"layer_type": ConvLayer, "batch_size": 2, "output_channels": 2, "input_channels": 2, "K_size": 2, "X_size": 3}], indirect=True)
    def test_update(self, setup_layers, learning_rate):

        my_layer, X, torch_layer, torch_X = setup_layers
        Y = my_layer.forward(X)
        torch_Y = torch_layer.forward(torch_X)

        dL_dOut = np.random.rand(*torch_Y.shape)
        torch_dL_dOut = torch.tensor(dL_dOut, dtype=torch.float32)
        
        my_layer.backward(dL_dOut)
        torch_Y.backward(torch_dL_dOut)

        with torch.no_grad():
            torch_layer.weight -= learning_rate * torch_layer.weight.grad
            torch_layer.bias -= learning_rate * torch_layer.bias.grad

        my_layer.update(learning_rate)

        # compare outputs
        assert np.allclose(my_layer.W, torch_layer.weight.detach().numpy())
        assert np.allclose(my_layer.B, torch_layer.bias.detach().numpy())


class TestReluLayer:
    params = [{"layer_type": ReluLayer, "batch_size": 1, "X_size": 3},
            {"layer_type": ReluLayer, "batch_size": 2, "X_size": 3},
            {"layer_type": ReluLayer, "batch_size": 1, "X_size": 4},
            {"layer_type": ReluLayer, "batch_size": 2, "X_size": 4}]
    
    @pytest.mark.parametrize("setup_layers", params, indirect=True)
    def test_forward(self, setup_layers):

        my_layer, X, torch_layer, torch_X = setup_layers
        Y = my_layer.forward(X)
        torch_Y = torch_layer.forward(torch_X)

        assert np.allclose(Y, torch_Y.detach().numpy())


    @pytest.mark.parametrize("setup_layers", params, indirect=True)
    def test_backward(self, setup_layers):

        my_layer, X, torch_layer, torch_X = setup_layers
        Y = my_layer.forward(X)
        torch_Y = torch_layer.forward(torch_X)

        dL_dOut = np.random.rand(*torch_Y.shape)
        torch_dL_dOut = torch.tensor(dL_dOut, dtype=torch.float32)
        my_layer.backward(dL_dOut)
        torch_Y.backward(torch_dL_dOut)

        assert np.allclose(my_layer.dL_dX, torch_X.grad)


class TestPoolLayer:

    params = [{"layer_type": PoolLayer, "batch_size": 1, "K_size": 2, "X_size": 3},
            {"layer_type": PoolLayer, "batch_size": 2, "K_size": 2, "X_size": 3},
            {"layer_type": PoolLayer, "batch_size": 1, "K_size": 3, "X_size": 4},
            {"layer_type": PoolLayer, "batch_size": 2, "K_size": 3, "X_size": 4}]

    @pytest.mark.parametrize("setup_layers", params, indirect=True)
    def test_forward(self, setup_layers):    
        my_layer, X, torch_layer, torch_X = setup_layers
        Y = my_layer.forward(X)
        (torch_Y, torch_argmax) = torch_layer.forward(torch_X)

        assert np.allclose(Y, torch_Y.detach().numpy())
        assert np.allclose(my_layer.argmax_mask, torch_argmax)

    @pytest.mark.parametrize("setup_layers", params, indirect=True)
    def test_backward(self, setup_layers):
        my_layer, X, torch_layer, torch_X = setup_layers
        Y = my_layer.forward(X)
        (torch_Y, torch_argmax) = torch_layer.forward(torch_X)

        dL_dOut = np.random.rand(*torch_Y.shape)
        torch_dL_dOut = torch.tensor(dL_dOut, dtype=torch.float32)
        my_layer.backward(dL_dOut)
        torch_Y.backward(torch_dL_dOut)
        assert np.allclose(my_layer.dL_dX, torch_X.grad)

class TestFlattenLayer:
    params = [{"layer_type": FlattenLayer, "batch_size": 1, "input_channels": 1, "X_size": 3},
            {"layer_type": FlattenLayer, "batch_size": 2, "input_channels": 1, "X_size": 3},
            {"layer_type": FlattenLayer, "batch_size": 1, "input_channels": 2, "X_size": 4},
            {"layer_type": FlattenLayer, "batch_size": 2, "input_channels": 2, "X_size": 4}]
    
    @pytest.mark.parametrize("setup_layers", params, indirect=True)
    def test_forward(self, setup_layers):
        my_layer, X, torch_layer, torch_X = setup_layers
        Y = my_layer.forward(X)
        torch_Y = torch_layer.forward(torch_X)

        assert np.allclose(Y, torch_Y.detach().numpy())

    @pytest.mark.parametrize("setup_layers", params, indirect=True)
    def test_backward(self, setup_layers):

        my_layer, X, torch_layer, torch_X = setup_layers
        Y = my_layer.forward(X)
        torch_Y = torch_layer.forward(torch_X)

        dL_dOut = np.random.rand(*torch_Y.shape)
        torch_dL_dOut = torch.tensor(dL_dOut, dtype=torch.float32)
        my_layer.backward(dL_dOut)
        torch_Y.backward(torch_dL_dOut)
        
        assert np.allclose(my_layer.dL_dX, torch_X.grad)