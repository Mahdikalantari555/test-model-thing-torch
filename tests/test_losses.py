import math
import pytest
import torch
from src.model.rtu import (
    loss_variance,
    loss_pred_mse,
    loss_crossentropy,
    loss_stop_mse,
    compute_losses,
)

def test_variance_loss_ddof0():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    # Population variance ddof=0
    pop_var = torch.var(x, unbiased=False)
    expected_l_var = torch.clamp(1.0 - torch.sqrt(pop_var + 1e-4), min=0.0)
    actual_l_var = loss_variance(x)
    assert torch.allclose(actual_l_var, expected_l_var)

def test_ce_loss():
    output = torch.randn(256)
    nextb = 100
    expected = -output[nextb] + torch.logsumexp(output, dim=-1)
    actual = loss_crossentropy(output, nextb)
    assert torch.allclose(actual, expected)

def test_stop_mse_loss():
    stop = torch.tensor([0.2])
    # end = False -> target 0.0
    l0 = loss_stop_mse(stop, end=False)
    assert torch.allclose(l0, torch.tensor([(0.2 - 0.0) ** 2]))

    # end = True -> target 1.0
    l1 = loss_stop_mse(stop, end=True)
    assert torch.allclose(l1, torch.tensor([(0.2 - 1.0) ** 2]))

def test_total_loss_composition():
    dim = 32
    x = torch.randn(dim)
    output = torch.randn(256)
    stop = torch.tensor([0.1])
    tgt = torch.randn(dim)
    nextb = 10

    total, losses = compute_losses(x, output, stop, tgt=tgt, nextb=nextb, end=True)

    expected_total = (
        loss_variance(x)
        + loss_pred_mse(x, tgt)
        + loss_crossentropy(output, nextb)
        + loss_stop_mse(stop, end=True)
    )
    assert torch.allclose(total, expected_total)
    assert 'var' in losses
    assert 'pred' in losses
    assert 'ce' in losses
    assert 'stop' in losses
