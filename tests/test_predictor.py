import pytest
import torch
from src.model.rtu import Model, loss_pred_mse

def test_latent_predictor_objective():
    dim = 64
    model = Model(dim=dim, layers=2)
    nextb = 65

    target = model.encoder(nextb).detach()
    pred = torch.randn(dim)

    mse = loss_pred_mse(pred, target)
    expected_mse = torch.mean((pred - target) ** 2)
    assert torch.allclose(mse, expected_mse)

def test_predictor_computed_only_when_nextb_not_none():
    dim = 64
    model = Model(dim=dim, layers=2)
    c = 42

    # when nextb is None (eval/chat sampling)
    with torch.no_grad():
        b, stop = model(c, nextb=None, frozen=True)
    assert isinstance(b, int)
    assert isinstance(stop, float)

    # when nextb is provided (training step)
    b, stop = model(c, nextb=43, end=False, frozen=False)
    assert isinstance(b, int)
    assert isinstance(stop, float)
