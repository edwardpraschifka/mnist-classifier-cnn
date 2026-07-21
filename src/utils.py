import numpy as np
import torch
import torch.nn as nn

def convolve_1c(K: np.ndarray,X: np.ndarray, stride=1) -> np.ndarray:
    """Performs a cross-correlation of a single 2D kernel over a single 2D input.

    Args:
        K (np.ndarray): The 2D kernel, shape (kh, kw).
        X (np.ndarray): The 2D input, shape (xh, xw). Must satisfy xh >= kh and xw >= kw.
        stride (int): kernel step size

    Returns:
        A 2D array of shape (xh - kh + 1, xw - kw + 1) containing the
        cross-correlation output at each valid sliding-window position.

    Raises:
        ValueError: If X is smaller than K in either spatial dimension.

    Example:
        ```K = np.array([[1, 0], [0, 1]])
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        convolve_1c(K, X)
        array([[6, 8], [12, 14]])"""

    kh, kw = np.shape(K)
    xh, xw = np.shape(X)

    if xh < kh or xw < kw:
        raise ValueError(f"dimensions of K ({kh,kw}) cannot exceed dimensions of X ({xh,xw})")

    yh, yw = int(np.ceil((xh-kh+1)/stride)), int(np.ceil((xw-kw+1)/stride))
    Y = np.zeros((yh,yw))

    for i in range(yh):
        for j in range(yw):
            Y[i][j] = np.sum(K * X[(i*stride):(i*stride)+kh, (j*stride):(j*stride)+kw])
    
    return Y

def convolve(W: np.ndarray, X: np.ndarray, padding=0, stride=1):
    """Performs a convolution over multiple channels"""

    pad_width = [(0,0), (padding, padding), (padding, padding)]
    X = np.pad(X, pad_width)

    wh, ww, kh, kw = np.shape(W)
    xc, xh, xw = np.shape(X)
    yc, yh, yw = wh, int(np.ceil((xh-kh+1)/stride)), int(np.ceil((xw-kw+1)/stride))
    Y = np.zeros((yc, yh, yw))

    for i in range(wh):
            Y[i] = np.sum([convolve_1c(W_ij, X_j, stride) for (W_ij, X_j) in zip(W[i],X)], axis=0)

    return Y

def softmax(X: np.ndarray):
    """Computes softmax for each row of X"""

    e_X = np.exp(X)
    return e_X/np.sum(e_X, axis=1, keepdims=True)


def quick_conv2d(W: np.ndarray, B: np.ndarray, padding=0, stride=1):
    """Quickly makes a conv2d layer in pytorch"""

    output_channels, input_channels, wh, _ = np.shape(W)
    conv = nn.Conv2d(input_channels, output_channels, wh, stride, padding)

    with torch.no_grad():
            conv.weight.copy_(torch.tensor(W))
            conv.bias.copy_(torch.tensor(B))

    return conv