# softmax + cross-entropy

import pytest
import numpy as np
import torch
import torch.nn as nn

from losses import softmax, cross_ent

def test_softmax():
    X = torch.randn(3, 5)

    softmax_X = softmax(X.detach().numpy())
    torch_softmax_X = torch.softmax(X, dim=-1)

    assert np.allclose(softmax_X, torch_softmax_X)