import os
import tempfile
import pytest
import torch
from src.model.rtu import Model

def test_checkpoint_roundtrip_and_traces():
    dim = 64
    layers = 2
    model1 = Model(dim=dim, layers=layers)

    # Perform a few steps to populate states, traces, and optimizer states
    model1(65, nextb=66, end=False, frozen=False)
    model1(66, nextb=67, end=True, frozen=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "model.safetensors")
        model1.save(ckpt_path)
        assert os.path.exists(ckpt_path)

        # Load into fresh model
        model2 = Model(dim=dim, layers=layers)
        model2.load(ckpt_path)

        # Check weights
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2, atol=1e-6)

        # Check buffers
        for b1, b2 in zip(model1.blocks, model2.blocks):
            assert torch.allclose(b1.states, b2.states, atol=1e-6)
            assert torch.allclose(b1.decaytrace, b2.decaytrace, atol=1e-6)
            assert torch.allclose(b1.embedtrace, b2.embedtrace, atol=1e-6)

        assert model1.step_count == model2.step_count

        # Check deterministic output
        torch.manual_seed(123)
        out1, stop1 = model1(70, frozen=True)
        torch.manual_seed(123)
        out2, stop2 = model2(70, frozen=True)
        assert out1 == out2
        assert abs(stop1 - stop2) < 1e-5

def test_checkpoint_reset_clears_state():
    dim = 32
    model = Model(dim=dim, layers=2)
    model(10, nextb=20, frozen=False)

    assert model.blocks[0].states.abs().sum() > 0
    model.reset()
    for b in model.blocks:
        assert torch.all(b.states == 0.0)
        assert torch.all(b.decaytrace == 0.0)
        assert torch.all(b.embedtrace == 0.0)
