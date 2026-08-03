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

class PoolLayer(Layer):
    """Applies 2D max-pooling with a fixed window size and stride.

    Divides the input's spatial dimensions into non-overlapping (or overlapping,
    depending on stride) windows and outputs the maximum value from each window,
    reducing the spatial resolution. Applied independently per channel and per
    batch example — the batch and channel dimensions pass through unchanged.

    This layer has no learnable parameters. During the forward pass, the position
    of the max within each window is cached so that backward() can route the
    upstream gradient to that specific position (and zero elsewhere).

    Attributes:
        pool_size: Tuple (ph, pw) — the pooling window's height and width.
        stride: The step size between windows along both spatial dimensions.
        argmax_mask: The cached argmax positions from the most recent forward()
            call, needed by backward() to route gradients correctly. None before
            the first forward pass.
        dL_dX: The gradient of the loss with respect to X, computed and stored
            during backward(). None before the first backward pass.
    """

    def __init__(self, pool_size: int):
        """Initializes the pooling layer.

        Args:
            pool_size: Tuple (ph, pw) specifying the window height and width.
        """

        self.pool_size = pool_size

        # filled after caling forward()
        self.argmax_mask = None

        # filled after caling backward()
        self.dL_dX = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Applies max-pooling to X.

        For each pooling window, keeps the maximum value and records its
        position (used by backward()).

        Args:
            X: The input, shape (batch_size, channels, x_height, x_width).

        Returns:
            The pooled output, shape (batch_size, channels, y_height, y_width).
        """

        batch_size, channels, xh, xw = X.shape
        yh, yw = xh//self.pool_size, xw//self.pool_size 
        y = np.zeros((batch_size, channels, yh, yw))
        X_flat = X.reshape(batch_size, channels, -1)
        self.argmax_mask = []

        for i in range(0, xh - self.pool_size + 1, self.pool_size):
            for j in range(0, xw - self.pool_size + 1, self.pool_size):
                window = X[:,:,i:i+self.pool_size, j:j+self.pool_size]

                # corresponding y indices
                k, l = i//self.pool_size, j//self.pool_size

                # === find y for this window (easy - just max value of window)===
                y[:, :, k, l] = np.max(window, axis=(2,3))

                # === find argmax_mask for this window ===
                # flatten window, apply argmax on each training example/channel
                # combination
                window_flat = window.reshape(batch_size, channels, -1)
                window_flat_i = np.argmax(window_flat, axis=2).reshape(-1)

                # convert flattened index to pairwise coordinates
                # relative to window
                window_i, window_j = np.unravel_index(window_flat_i, (self.pool_size,self.pool_size))

                # convert pairwise window coordinates to
                # pairwise X coordinates
                X_i, X_j = i + window_i, j + window_j

                # convert pairwise X coordinates to flattened
                # X coordinates
                X_flat_i = (X_i * xw) + X_j

                # update argmax_mask with flattened X coordinates
                self.argmax_mask.append(X_flat_i)

        self.argmax_mask = np.stack(self.argmax_mask, axis=-1).reshape(y.shape)
        return y
        


    def backward(self, dL_dOut: np.ndarray) -> np.ndarray:
        """Computes dL/dX given dL/dOut by routing gradient to argmax positions.

        For each pooled output position, the upstream gradient is placed at
        the input position that was the argmax during forward() (recorded in
        argmax_mask). All other positions in each window receive zero.

        In the case of ties within a window (multiple positions holding the
        same max value), the gradient is routed to whichever position was
        selected as "the" argmax during forward — dependent on the
        tie-breaking convention used there.

        Args:
            dL_dOut: The upstream gradient, shape matching forward()'s output.

        Returns:
            The gradient of the loss with respect to X, same shape as the X
            passed to forward. Mostly zeros — nonzero only at the argmax
            positions from the forward pass.
        """

        pass
    