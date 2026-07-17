# softmax + cross-entropy

import pytest
import numpy as np
import torch
import torch.nn as nn

from losses import softmax, cross_ent, softmax_cross_ent_grad

def test_softmax():
    X = torch.randn(3, 5)

    softmax_X = softmax(X.detach().numpy())
    torch_softmax_X = torch.softmax(X, dim=-1)

    assert np.allclose(softmax_X, torch_softmax_X)


def test_cross_ent():
    torch.manual_seed(42)
    X = torch.randn(3, 5)
    softmax_X = torch.softmax(X, dim=-1)
    target = torch.zeros(3, 5)
    target[0][2] = 1
    target[1][0] = 1
    target[2][4] = 1
    
    cross_ent_softmax = cross_ent(softmax_X.detach().numpy(), target.detach().numpy())

    loss = nn.CrossEntropyLoss(reduction='none')
    output = loss(X, target)

    assert np.allclose(cross_ent_softmax, output)


def test_softmax_cross_ent_grad():
    torch.manual_seed(42)
    X = torch.randn(1, 100, requires_grad=True)
    target = torch.zeros(1, 100)
    target[0][42] = 1
    
    dL_dOut = softmax_cross_ent_grad(X.detach().numpy(), target.detach().numpy(),)

    loss = nn.CrossEntropyLoss()
    output = loss(X, target)
    output.backward()

    assert np.allclose(X.grad, dL_dOut)