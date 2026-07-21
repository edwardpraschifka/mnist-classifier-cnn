# Layer base class + ConvLayer, PoolLayer, ReLULayer, FlattenLayer, FCLayer

"""Defines the Layer abstract base class and its concrete subclasses.

Each Layer subclass implements a forward pass, a backward pass, and
(optionally) a parameter update step, forming the building blocks of a
neural network defined as an ordered list of Layer instances."""

from abc import ABC, abstractmethod
import numpy as np

from utils import convolve

class Layer(ABC):

    @abstractmethod
    def forward(self, X: np.ndarray):
        """Return layer's output for input X"""

        pass

    @abstractmethod
    def backward(self, dL_dOut):
        """Return dL_dInput, given dL_dOutput"""

        pass
    
    def update(self, lr):
        """Apply gradient descent step to layer's parameters"""

        pass


class ConvLayer(Layer):
    """A 2D convolutional layer with learnable kernels and biases.

    Performs a batched multi-channel 2D cross-correlation over its input,
    optionally with padding and stride, followed by a per-output-channel
    bias addition."""

    def __init__(self, W: np.ndarray, B: np.ndarray, stride = 1, padding = 0):
        """Initializes the layer with fixed initial weights and biases.

        Args:
            W: The weight tensor, shape (out_channels, in_channels, kh, kw).
            B: The bias vector, shape (out_channels,). One scalar bias per
                output channel, broadcast across the batch and spatial axes.
            stride: Kernel step size along both spatial dimensions. Defaults to 1.
            padding: Number of zero-pad rows/columns added to each spatial
                side of the input before convolving. Defaults to 0."""

        self.W = W
        self.B = B
        self.stride = stride
        self.padding = padding


    def forward(self, X: np.ndarray):
        """Performs the forward pass: cross-correlate X with W, then add bias.

        Args:
            X: The input, shape (batch_size, in_channels, x_height, x_width).
                Must satisfy X.shape[1] == W.shape[1] (matching input channels).

        Returns:
            The output, shape (batch_size, out_channels, y_height, y_width),
            where y_height and y_width are determined by the input size,
            kernel size, stride, and padding."""

        Y = convolve(self.W, X, self.padding, self.stride)

        # broadcast along output channel axis
        return Y + self.B.reshape(1, -1, 1, 1)

    def backward(self, dL_dOut):
        """Computes dL/dW, dL/dB, and dL/dX.

        Given the upstream gradient, computes the gradients with respect to
        this layer's kernel and bias (stored on self for use by update()),
        and returns the gradient with respect to the input.

        Args:
            dL_dOut: The upstream gradient, shape matching forward()'s output.

        Returns:
            The gradient of the loss with respect to X, shape matching the
            X that was passed to forward()."""

        pass

    
    def update(self, lr):
        """Applies a gradient descent step to W and B.

        Uses the gradients dL/dW and dL/dB computed and stored during
        the most recent backward() call.

        Args:
            lr: The learning rate."""

        pass