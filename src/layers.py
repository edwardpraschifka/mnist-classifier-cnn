# Layer base class + ConvLayer, PoolLayer, ReLULayer, FlattenLayer, FCLayer

from abc import ABC, abstractmethod
import numpy as np

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