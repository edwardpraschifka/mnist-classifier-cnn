# Layer base class + ConvLayer, PoolLayer, ReLULayer, FlattenLayer, FCLayer

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
        """Return derivative of loss with respect to layer's
            input, given derivative of loss with respect to
            layer's output"""

        pass
    
    def update(self, lr):
        """Update layer's parameters, if it has any"""

        pass


class ConvLayer(Layer):

    def __init__(self, W: np.ndarray, B: np.ndarray, stride = 1, padding = 0):
        self.W = W
        self.B = B
        self.stride = stride
        self.padding = padding


    def forward(self, X: np.ndarray):
        Y = convolve(self.W, X, self.padding, self.stride)
        return np.array([Y_i + B_i for (Y_i,B_i) in zip(Y,self.B)])

    def backward(self, dL_dOut):
        pass

    
    def update(self, lr):
        pass