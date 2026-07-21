# softmax + cross-entropy

import pytest
import numpy as np
import torch
import torch.nn as nn

from losses import loss, loss_grad
from utils import quick_target

@pytest.mark.parametrize("X_height", [2,3])
@pytest.mark.parametrize("X_width", [2,3] )
def test_loss(X_height: int, X_width: int):

    np.random.seed(42)
    X = np.random.rand(X_height, X_width)
    target = quick_target(X_height, X_width)
    
    my_loss = loss(X, target)

    cel = nn.CrossEntropyLoss()

    torch_loss = cel(torch.tensor(X),torch.tensor(target))

    assert np.allclose(my_loss, torch_loss)

def test_loss_grad():
    torch.manual_seed(42)
    X = torch.randn(3, 5, requires_grad=True)
    target = torch.zeros(3, 5)
    target[0][2] = 1
    target[1][0] = 1
    target[2][4] = 1
    
    dL_dOut = loss_grad(X.detach().numpy(), target.detach().numpy())

    cel = nn.CrossEntropyLoss()
    torch_loss = cel(X, target)
    torch_loss.backward()

    assert np.allclose(X.grad, dL_dOut)