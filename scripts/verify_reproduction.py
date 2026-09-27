import os
import time
import math
import torch
import psutil

from src.model.rtu import Model, compute_losses
from src.data import load_corpus

def run_audits():
    print("=== 1. TENSOR SHAPE & PARAMETER AUDIT ===")
    dim = 512
    layers = 16
    model = Model(dim=dim, layers=layers)
    total_params = model.count()
    actual_params = sum(p.numel() for p in model.parameters())

    print(f"Model count(): {total_params:,}")
    print(f"Actual trainable params: {actual_params:,}")
    assert total_params == actual_params, f"Mismatch: {total_params} != {actual_params}"

    c = 65
    dummies = [torch.zeros(dim) for _ in range(layers)]
    (x, states, decays), (logits, stop) = model.step(c, dummies=dummies)

    print(f"Encoder output shape: {model.encoder(c).shape}")
    print(f"Residual x shape: {x.shape}")
    print(f"States shape (per layer): {states[0].shape}")
    print(f"Decays shape (per layer): {decays[0].shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Stop shape: {stop.shape}")

    assert model.encoder(c).shape == (dim,)
    assert x.shape == (dim,)
    assert states[0].shape == (dim,)
    assert decays[0].shape == (dim,)
    assert logits.shape == (256,)
    assert stop.shape == (1,)

    print("\n=== 2. ACTIVATION STATISTICS AUDIT ===")
    torch.manual_seed(42)
    sample_text = b"Numerical verification test sequence."
    model.reset()
    for b in sample_text:
        model(b, frozen=True)

    state_means = [s.mean().item() for s in [b.states for b in model.blocks]]
    state_stds = [s.std().item() for s in [b.states for b in model.blocks]]
    print(f"Layer 0 state mean: {state_means[0]:.6f}, std: {state_stds[0]:.6f}")
    print(f"Layer 15 state mean: {state_means[-1]:.6f}, std: {state_stds[-1]:.6f}")
    print(f"Decoder logits mean: {logits.mean().item():.6f}, std: {logits.std().item():.6f}")
    print(f"Stop value: {stop.item():.6f}")

    print("\n=== 3. GRADIENT AUDIT ===")
    model.reset()
    torch.manual_seed(42)
    c_in = 65
    c_next = 66
    model(c_in, nextb=c_next, end=False, frozen=False)
    emb_grad_norm = model.encoder.embed.weight.grad.norm().item()
    decay_grad_norm = model.blocks[0].decay.grad.norm().item()
    weights_grad_norm = model.blocks[0].weights.weight.grad.norm().item()

    print(f"encoder.embed.weight grad norm: {emb_grad_norm:.6f}")
    print(f"blocks[0].decay grad norm: {decay_grad_norm:.6f}")
    print(f"blocks[0].weights grad norm: {weights_grad_norm:.6f}")

    print("\n=== 4. TINYSTORIES EXPERIMENT (PHASE 8 GATE) ===")
    # Train on TinyStories corpus (CPU execution)
    train_ds = load_corpus("tinystories", seed=42)
    small_model = Model(dim=128, layers=4, rate=1e-3, bound=(50, 200))
    small_model.reset()

    process = psutil.Process(os.getpid())
    start_mem = process.memory_info().rss / (1024 * 1024)

    steps = 150
    losses = []
    start_time = time.time()

    for step_i in range(steps):
        c, n, is_end = train_ds[step_i % len(train_ds)]
        c_t = torch.tensor(c, dtype=torch.long)
        n_t = torch.tensor(n, dtype=torch.long)
        dummies = [torch.zeros(small_model.dim, requires_grad=True) for _ in range(small_model.layers)]

        small_model.zero_grad(set_to_none=True)
        with torch.no_grad():
            tgt = small_model.encoder(n_t).detach()
        (x_step, states_step, decays_step), (out_step, stop_step) = small_model.step(c_t, dummies=dummies)
        tot_loss, _ = compute_losses(x_step, out_step, stop_step, tgt=tgt, nextb=n_t, end=is_end)
        tot_loss.backward()

        c_range = (torch.arange(256) == c).float().unsqueeze(1)
        for i, layer in enumerate(small_model.blocks):
            dlds = dummies[i].grad if dummies[i].grad is not None else torch.zeros(small_model.dim)
            decay_embed = layer.embedtrace * decays_step[i].detach()
            layer.embedtrace.copy_((decay_embed + c_range).detach())
            small_model.encoder.embed.weight.grad.add_(dlds * decay_embed)

            dec_val = decays_step[i].detach()
            decaytrace = (dec_val * layer.decaytrace) + (dec_val * (1.0 - dec_val) * layer.states)
            layer.decay.grad = (dlds * decaytrace).clone()

            layer.states.copy_(states_step[i].detach())
            layer.decaytrace.copy_(decaytrace.detach())

        small_model.optimizer.step()
        losses.append(tot_loss.item())

        if (step_i + 1) % 30 == 0:
            avg_loss = sum(losses[-30:]) / 30.0
            print(f"Step {step_i + 1}/{steps} - Loss: {avg_loss:.4f}")

    elapsed = time.time() - start_time
    end_mem = process.memory_info().rss / (1024 * 1024)
    tokens_per_sec = steps / elapsed

    print(f"\nThroughput: {tokens_per_sec:.2f} tokens/sec")
    print(f"Memory RSS: {end_mem:.2f} MB (Delta: {end_mem - start_mem:.2f} MB)")
    print(f"Initial loss: {losses[0]:.4f} -> Final loss: {sum(losses[-10:])/10:.4f}")

    return {
        "total_params": total_params,
        "tokens_per_sec": tokens_per_sec,
        "start_loss": losses[0],
        "final_loss": sum(losses[-10:]) / 10,
        "mem_mb": end_mem,
    }

if __name__ == "__main__":
    run_audits()
