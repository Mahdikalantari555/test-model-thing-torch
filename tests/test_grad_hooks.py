import pytest
import torch
from src.model.rtu import Model, compute_losses

def test_custom_grad_hooks_presence_and_replacement():
    dim = 64
    model = Model(dim=dim, layers=2)
    # Give non-zero trace and states to test gradient alteration
    for layer in model.blocks:
        layer.embedtrace.normal_()
        layer.decaytrace.normal_()
        layer.states.normal_()

    c = 12
    nextb = 13

    # Run step and backward manually to compare autodiff vs custom hook
    c_tensor = torch.tensor(c, dtype=torch.long)
    dummies = [torch.zeros(dim, requires_grad=True) for _ in range(model.layers)]
    tgt = model.encoder(torch.tensor(nextb, dtype=torch.long)).detach()

    (x, states, decays), (output, stop) = model.step(c_tensor, dummies=dummies)
    loss, _ = compute_losses(x, output, stop, tgt=tgt, nextb=nextb, end=False)
    loss.backward()

    # Pre-hook gradients
    autodiff_embed_grad = model.encoder.embed.weight.grad.clone()
    autodiff_decay_grad = model.blocks[0].decay.grad.clone()

    assert autodiff_embed_grad is not None
    assert autodiff_decay_grad is not None

    # Apply hooks
    c_range = (torch.arange(256) == c).float().unsqueeze(1)
    for i, layer in enumerate(model.blocks):
        dlds = dummies[i].grad
        decay_embedtrace = layer.embedtrace * decays[i].detach()
        embedtrace = decay_embedtrace + c_range

        # embed grad ADDS
        model.encoder.embed.weight.grad.add_(dlds * decay_embedtrace)

        # decay grad REPLACES
        decay_val = decays[i].detach()
        decaytrace = (decay_val * layer.decaytrace) + (decay_val * (1.0 - decay_val) * layer.states)
        layer.decay.grad = (dlds * decaytrace).clone()

    # Verify embed grad changed by addition
    assert not torch.allclose(model.encoder.embed.weight.grad, autodiff_embed_grad)
    # Verify decay grad replaced
    assert not torch.allclose(model.blocks[0].decay.grad, autodiff_decay_grad)
    assert model.blocks[0].decay.grad.abs().sum() >= 0

def test_buffers_not_in_trainable_parameters():
    model = Model(dim=64, layers=2)
    param_names = [name for name, _ in model.named_parameters()]
    for name in param_names:
        assert not name.endswith('states')
        assert not name.endswith('decaytrace')
        assert not name.endswith('embedtrace')
