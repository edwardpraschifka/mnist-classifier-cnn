# Layer base class + ConvLayer, PoolLayer, ReLULayer, FlattenLayer, FCLayer

"""Defines the Layer abstract base class and its concrete subclasses.

Each Layer subclass implements a forward pass, a backward pass, and
(optionally) a parameter update step, forming the building blocks of a
neural network defined as an ordered list of Layer instances."""

from abc import ABC, abstractmethod
import numpy as np

from utils import convolve_1c, convolve

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

        # filled after calling forward()
        self.X = None

        # filled after calling backward()
        self.dL_dW = None
        self.dL_dB = None
        self.dL_dX = None


    def forward(self, X: np.ndarray):
        """Performs the forward pass: cross-correlate X with W, then add bias.

        Args:
            X: The input, shape (batch_size, in_channels, x_height, x_width).
                Must satisfy X.shape[1] == W.shape[1] (matching input channels).

        Returns:
            The output, shape (batch_size, out_channels, y_height, y_width),
            where y_height and y_width are determined by the input size,
            kernel size, stride, and padding."""
        
        # store X for use in backward()
        self.X = X

        Y = convolve(self.W, X, self.padding, self.stride)

        # broadcast along output channel axis
        return Y + self.B.reshape(1, -1, 1, 1)

    def backward(self, dL_dOut):
        """Computes dL/dW, dL/dB, and dL/dX.

        Given the upstream gradient, computes the gradients with respect to
        this layer's kernel and bias (stored on self for use by update()),
        and returns the gradient with respect to the input.

        Args:
            dL_dOut: The upstream gradient, shape matching forward()'s output."""

        # compute dL/dW
        self.dL_dW = np.zeros(np.shape(self.W))

        output_channels = np.shape(self.W)[0]
        input_channels = np.shape(self.W)[1]
        batch_size = np.shape(self.X)[0]

        for i in range(output_channels): 
            for j in range(input_channels): 
                    for b in range(batch_size):
                        self.dL_dW[i][j] += convolve_1c(dL_dOut[b][i], self.X[b][j])
        
        # compute dL/dB
        self.dL_dB = np.sum(dL_dOut, axis=(0, 2, 3))

        # compute dL/dX
        W_rot = np.rot90(self.W, k=2, axes = (2,3))
        W_rot = W_rot.transpose(1, 0, 2, 3)
        _, _, kh, kw = self.W.shape
        pad = ((0,0), (0,0), (kh-1, kh-1), (kw-1,kw-1))
        dL_dOut_pad = np.pad(dL_dOut, pad)

        self.dL_dX = convolve(W_rot, dL_dOut_pad)
    
    def update(self, lr):
        """Applies a gradient descent step to W and B.

        Uses the gradients dL/dW and dL/dB computed and stored during
        the most recent backward() call.

        Args:
            lr: The learning rate."""

        self.W -= self.dL_dW * lr
        self.B -= self.dL_dB * lr

class ReluLayer(Layer):
    """Applies the elementwise ReLU activation, f(x) = max(0, x).

    This layer has no learnable parameters and applies a purely elementwise
    transformation, so its forward and backward passes preserve the input
    shape exactly. Any input shape is supported.

    Attributes:
        X: The input array from the most recent forward() call, cached for
            backward()'s use in constructing the ReLU mask. None before the
            first forward pass.
        dL_dX: The gradient of the loss with respect to X, computed and stored
            during backward(). None before the first backward pass.
    """

    def __init__(self):
        """Initializes the layer with no parameters and empty caches."""

        # filled after calling forward()
        self.X = None

        # filled after caling backward()
        self.dL_dX = None
        

    def forward(self, X: np.ndarray):
        """Applies elementwise ReLU to X.

        Args:
            X: The input array of any shape.

        Returns:
            An array of the same shape as X, with negative entries set to 0
            and non-negative entries left unchanged.
        """

        self.X = X
        return np.maximum(0, X)

    def backward(self, dL_dOut):
        """Computes dL/dX given dL/dOut.

        Uses the ReLU backward rule: gradient passes through unchanged
        wherever the forward input was positive, and is zeroed elsewhere.

        Args:
            dL_dOut: The gradient of the loss with respect to this layer's
                output, matching the shape of what forward() returned.
        """

        self.dL_dX = (self.X > 0) * dL_dOut