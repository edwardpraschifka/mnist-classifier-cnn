# per-layer gradient checks against PyTorch

import pytest
import numpy as np

from layers import Layer

class TestBaseLayer:
    def test_instantiate(self):
        with pytest.raises(TypeError):
            my_layer_class = Layer()