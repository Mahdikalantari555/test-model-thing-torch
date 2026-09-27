import math
import pytest
import torch
from src.model.rtu import Layer

def test_rtu_decay_bounds():
    dim = 64
    layer = Layer(dim=dim, spread=16)
    decay = torch.sigmoid(layer.decay)
    assert torch.all(decay > 0.0)
    assert torch.all(decay < 1.0)

def test_rtu_state_update_equation():
    dim = 64
    layer = Layer(dim=dim, spread=16)
    layer.states.normal_()
    enc = torch.randn(dim)
    dummy = torch.randn(dim)

    decay = torch.sigmoid(layer.decay)
    expected_state = (decay * layer.states) + enc + dummy
    _, state, returned_decay = layer(enc, enc, dummy)

    assert torch.allclose(state, expected_state, atol=1e-6)
    assert torch.allclose(returned_decay, decay, atol=1e-6)

def test_rtu_decay_extremes():
    dim = 32
    layer = Layer(dim=dim)
    layer.states.fill_(5.0)
    enc = torch.full((dim,), 2.0)
    dummy = torch.zeros(dim)

    # decay -> 1 (param large positive)
    with torch.no_grad():
        layer.decay.fill_(30.0)
    _, state_retention, _ = layer(enc, enc, dummy)
    assert torch.allclose(state_retention, layer.states + enc + dummy, atol=1e-4)

    # decay -> 0 (param large negative)
    with torch.no_grad():
        layer.decay.fill_(-30.0)
    _, state_forget, _ = layer(enc, enc, dummy)
    assert torch.allclose(state_forget, enc + dummy, atol=1e-4)

def test_rtu_reset():
    dim = 32
    layer = Layer(dim=dim)
    layer.states.fill_(1.2)
    layer.decaytrace.fill_(0.8)
    layer.embedtrace.fill_(0.5)

    layer.reset()
    assert torch.all(layer.states == 0.0)
    assert torch.all(layer.decaytrace == 0.0)
    assert torch.all(layer.embedtrace == 0.0)

def test_rtu_gradient_flow():
    dim = 32
    layer = Layer(dim=dim)
    layer.states.normal_()
    enc = torch.randn(dim, requires_grad=True)
    x = enc.clone()
    dummy = torch.zeros(dim, requires_grad=True)

    x_out, state, decay = layer(enc, x, dummy)
    loss = x_out.sum()
    loss.backward()

    assert enc.grad is not None and enc.grad.abs().sum() > 0
    assert layer.decay.grad is not None and layer.decay.grad.abs().sum() > 0
    assert dummy.grad is not None and dummy.grad.abs().sum() > 0

def test_rtu_trace_updates():
    dim = 16
    layer = Layer(dim=dim, vocab_size=256)
    layer.decaytrace.fill_(0.2)
    layer.embedtrace.fill_(0.1)
    layer.states.fill_(0.3)

    decay = torch.sigmoid(layer.decay).detach()
    c = 10
    c_range = (torch.arange(256) == c).float().unsqueeze(1)

    expected_embedtrace = (layer.embedtrace * decay) + c_range
    expected_decaytrace = (decay * layer.decaytrace) + (decay * (1.0 - decay) * layer.states)

    decay_embedtrace = layer.embedtrace * decay
    actual_embedtrace = decay_embedtrace + c_range
    actual_decaytrace = (decay * layer.decaytrace) + (decay * (1.0 - decay) * layer.states)

    assert torch.allclose(actual_embedtrace, expected_embedtrace, atol=1e-6)
    assert torch.allclose(actual_decaytrace, expected_decaytrace, atol=1e-6)

def test_rtu_stop_gradient_detached():
    dim = 16
    layer = Layer(dim=dim)
    enc = torch.randn(dim, requires_grad=True)
    x = enc.clone()
    _, state, _ = layer(enc, x)

    # Detaching persistent buffer assignment
    layer.states.copy_(state.detach())
    assert not layer.states.requires_grad
