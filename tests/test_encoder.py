import pytest
import torch
import torch.nn as nn
from src.model.rtu import Encoder, Layer

def test_encoder_byte_output_shape():
    encoder = Encoder(dim=256)
    out = encoder(42)
    assert out.shape == (256,)
    assert out.dtype == torch.float32

def test_encoder_block_stack():
    dim = 128
    layers = 4
    blocks = [Layer(dim=dim, spread=16) for _ in range(layers)]
    enc = torch.randn(dim)
    x = enc.clone()
    dummies = [torch.zeros(dim) for _ in range(layers)]

    states, decays = [], []
    for i, layer in enumerate(blocks):
        x, state, decay = layer(enc, x, dummies[i])
        states.append(state)
        decays.append(decay)
        assert x.shape == (dim,)
        assert state.shape == (dim,)
        assert decay.shape == (dim,)

def test_layer_residual_and_norm_order():
    dim = 64
    layer = Layer(dim=dim, spread=8)
    enc = torch.randn(dim)
    x = torch.randn(dim)
    dummy = torch.zeros(dim)

    # Inspect module layers
    assert isinstance(layer.norm, nn.LayerNorm)
    assert isinstance(layer.weights, nn.Linear)
    assert layer.weights.bias is None  # bias=False
    assert isinstance(layer.silu, nn.SiLU)

    x_out, state, decay = layer(enc, x, dummy)
    # verify residual formula: x_out = x + silu(weights(norm(state)))
    expected_delta = layer.silu(layer.weights(layer.norm(state)))
    assert torch.allclose(x_out, x + expected_delta, atol=1e-6)
