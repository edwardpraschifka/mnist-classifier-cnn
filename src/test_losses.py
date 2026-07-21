# softmax + cross-entropy

import pytest
import numpy as np
import torch
import torch.nn as nn

from losses import loss, loss_grad
from utils import quick_target

@pytest.mark.parametrize("Y_height", [2,3])
@pytest.mark.parametrize("Y_width", [2,3])
def test_loss(Y_height: int, Y_width: int):

    np.random.seed(42)
    Y = np.random.rand(Y_height, Y_width)
    target = quick_target(Y_height, Y_width)
    
    my_loss = loss(Y, target)

    cel = nn.CrossEntropyLoss()

    torch_loss = cel(torch.tensor(Y),torch.tensor(target))

    assert np.allclose(my_loss, torch_loss)

@pytest.mark.parametrize("Y_height", [2,3])
@pytest.mark.parametrize("Y_width", [2,3])
def test_loss_grad(Y_height: int, Y_width: int):

    np.random.seed(42)
    Y = np.random.rand(Y_height, Y_width)
    target = quick_target(Y_height, Y_width)
    
    dL_dOut = loss_grad(Y, target)

    cel = nn.CrossEntropyLoss()
    torch_Y = torch.tensor(Y, requires_grad=True)
    torch_loss = cel(torch_Y, torch.tensor(target))
    torch_loss.backward()

    assert np.allclose(torch_Y.grad, dL_dOut)