import numpy as np
import torch
import torch.nn as nn

def convolve_1c(K: np.ndarray,X: np.ndarray, stride=1):
    """Performs a convolution over a single channel"""

    kh, kw = np.shape(K)
    xh, xw = np.shape(X)

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


def quick_conv2d(W: np.ndarray, B: np.ndarray, X: np.ndarray, padding: int, stride: int):
    """Quickly makes a conv2d layer in pytorch"""

    input_channels, xh, _ = np.shape(X)
    output_channels, _, wh, _ = np.shape(W)
    conv = nn.Conv2d(input_channels, output_channels, wh, stride, padding)

    with torch.no_grad():
            conv.weight.copy_(torch.tensor(W))
            conv.bias.copy_(torch.tensor(B))

    return conv