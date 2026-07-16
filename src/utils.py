import numpy as np

def convolve_1c(K: np.ndarray,X: np.ndarray):
    """Performs a convolution over a single channel"""

    kh, kw = np.shape(K)
    xh, xw = np.shape(X)

    yh, yw = xh - kh + 1, xw - kw + 1
    Y = np.zeros((yh,yw))

    for i in range(yh):
        for j in range(yw):
            Y[i][j] = np.sum(K * X[i:i+kh, j:j+kw])
    
    return Y

def convolve(W: np.ndarray, X: np.ndarray):
    """Performs a convolution over multiple channels"""

    wh, ww, kh, kw = np.shape(W)
    xc, xh, xw = np.shape(X)
    yc, yh, yw = wh, xh - kh + 1, xw - kw + 1
    Y = np.zeros((yc, yh, yw))

    for i in range(wh):
        for j in range(ww):
            Y[i] += convolve_1c(W[i][j], X[j])

    return Y

