import numpy as np
import torch
import torch.nn as nn

def convolve_1c(K: np.ndarray,X: np.ndarray, stride=1) -> np.ndarray:
    """Performs a cross-correlation of a single 2D kernel over a single 2D input.

    Args:
        K (np.ndarray): The 2D kernel, shape (kernel_height, kernel_width).
        X (np.ndarray): The 2D input, shape (x_height, x_width). Must satisfy x_height >= kernel_height and x_width >= kernel_width.
        stride (int): kernel step size

    Returns:
        A 2D array of shape `ceil((input_size - kernel_size + 1) / stride)` containing the
        cross-correlation output at each valid sliding-window position.

    Raises:
        ValueError: If X is smaller than K in either spatial dimension.

    Example:
        ```K = np.array([[1, 0], [0, 1]])
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        convolve_1c(K, X)
        array([[6, 8], [12, 14]])"""

    kernel_height, kernel_width = np.shape(K)
    x_height, x_width = np.shape(X)

    if x_height < kernel_height or x_width < kernel_width:
        raise ValueError(f"dimensions of K ({kernel_height,kernel_width}) cannot exceed dimensions of X ({x_height,x_width})")

    y_height, y_width = int(np.ceil((x_height-kernel_height+1)/stride)), int(np.ceil((x_width-kernel_width+1)/stride))
    Y = np.zeros((y_height,y_width))

    for i in range(y_height):
        for j in range(y_width):
            Y[i][j] = np.sum(K * X[(i*stride):(i*stride)+kernel_height, (j*stride):(j*stride)+kernel_width])
    
    return Y

def convolve(W: np.ndarray, X: np.ndarray, padding=0, stride=1):
    """Performs a cross-correlation of a weight tensor over several multi-channel inputs.

    Args:
        W (np.ndarray): The 4D weight tensor, shape (output_channels, input_channels, kernel_height, kernel_width).
        X (np.ndarray): The 4D input, shape (batch_size, input_channels, x_height, x_width). Must satisfy x_height >= kernel_height and x_width >= kernel_width.
        padding (int): adds `padding` layers of zeros to all sides of X.
        stride (int): kernel step size.

    Returns:
        A 4D array of shape (batch_size, output_channels, x_height - kernel_height + 1, x_width - kernel_width + 1) containing the
        cross-correlation output at each valid sliding-window position.

    Raises:
        ValueError: If W's `input_channels` doesn't match X's `input_channels`."""
    
    # apply padding to X
    pad_width = [(0,0), (0,0), (padding, padding), (padding, padding)]
    X = np.pad(X, pad_width)

    # label and check dimensions
    output_channels, input_channels, kernel_height, kernel_width = np.shape(W)
    batch_size, _, x_height, x_width = np.shape(X)
    y_height, y_width = int(np.ceil((x_height-kernel_height+1)/stride)), int(np.ceil((x_width-kernel_width+1)/stride))
    Y = np.zeros((batch_size, output_channels, y_height, y_width))

    if W.shape[1] != X.shape[1]:
        raise ValueError(f"W's input channels (f{W.shape[1]}) must match X's input channels (f{X.shape[1]})")

    for batch in range(batch_size):
            for oc in range(output_channels):
                Y[batch][oc] = np.sum([convolve_1c(W[oc][ic], X[batch][ic], stride) for ic in range(input_channels)], axis=0)

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

def quick_target(target_height: int, target_width: int) -> np.ndarray:
    """Generates a random one-hot target array.

    Creates a 2D array where each row is a one-hot vector — a single entry
    set to 1 at a uniformly random column position, with all other entries
    set to 0. Useful for constructing dummy classification targets in tests.

    Args:
        target_height: The number of rows (e.g. batch size, or number of examples).
        target_width: The number of columns (e.g. number of classes).

    Returns:
        A 2D array of shape (target_height, target_width) where each row contains
        exactly one 1 and (target_width - 1) zeros. The position of the 1 in each
        row is uniformly random and independent across rows.

    Example:
        ```>>> np.random.seed(0)
        >>> quick_target(3, 4)
        array([[0., 0., 0., 1.],
               [0., 1., 0., 0.],
               [0., 0., 1., 0.]])
    """

    target = np.zeros((target_height, target_width))

    for row in range(target_height):
        index = np.random.randint(0, target_width)
        target[row][index] = 1
    
    return target