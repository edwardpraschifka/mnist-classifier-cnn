# softmax + cross-entropy

import pytest
import numpy as np
import torch
import torch.nn as nn

from losses import loss, loss_grad

def test_loss():
    torch.manual_seed(42)
    X = torch.randn(3, 5)
    target = torch.zeros(3, 5)
    target[0][2] = 1
    target[1][0] = 1
    target[2][4] = 1
    
    my_loss = loss(X.detach().numpy(), target.detach().numpy())

    cel = nn.CrossEntropyLoss()

    torch_loss = cel(X,target)

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