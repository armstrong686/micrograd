import math

import pytest
import torch

from mygrad.engine import Value
def test_sigmoid_against_pytorch():
    # MyGrad
    x = Value(0.7)
    result = x.sigmoid()
    result.backward()

    # PyTorch
    torch_x = torch.tensor(
        0.7,
        dtype=torch.float64,
        requires_grad=True
    )

    torch_result = torch.sigmoid(torch_x)
    torch_result.backward()

    assert math.isclose(
        result.data,
        torch_result.item(),
        rel_tol=1e-9,
        abs_tol=1e-9
    )

    assert math.isclose(
        x.grad,
        torch_x.grad.item(),
        rel_tol=1e-9,
        abs_tol=1e-9
    )
def test_tanh_against_pytorch():
    # MyGrad
    x = Value(0.7)
    result = x.tanh()
    result.backward()

    # PyTorch
    torch_x = torch.tensor(
        0.7,
        dtype=torch.float64,
        requires_grad=True
    )

    torch_result = torch.tanh(torch_x)
    torch_result.backward()

    assert math.isclose(
        result.data,
        torch_result.item(),
        rel_tol=1e-9,
        abs_tol=1e-9
    )

    assert math.isclose(
        x.grad,
        torch_x.grad.item(),
        rel_tol=1e-9,
        abs_tol=1e-9
    )