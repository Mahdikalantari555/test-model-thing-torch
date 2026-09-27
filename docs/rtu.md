# Recurrent Trace Unit (RTU) Architecture & Memory Spec

## Overview

The Recurrent Trace Unit (RTU) is a recurrent memory cell designed for byte-level language modeling and streaming latent prediction without a conventional attention context window.

## Core State Equations

For layer $i$ at step $t$ given input representation $\text{enc}_t$ and residual vector $x_t$:

$$
\text{decay}_t = \sigma(\text{decay\_param}_i)
$$

$$
\text{state}_t = \text{decay}_t \odot \text{state}_{t-1} + \text{enc}_t + \text{dummy}_t
$$

$$
x_{t+1} = x_t + \text{SiLU}\Big(W_i \, \text{LayerNorm}(\text{state}_t)\Big)
$$

- $\text{decay\_param}$ is initialized across dimensions by spanning log half-lives across the layer spread.
- $\text{dummy}_t$ is an identity carrier allowing automatic differentiation to recover $d\mathcal{L} / d(\text{state}_t)$.

## Persistent Trace Buffers

Each layer maintains three persistent states:
1. `states` $(\text{dim})$: Hidden memory vector, updated via exponential moving average (EMA).
2. `decaytrace` $(\text{dim})$: Decayed state accumulator tracking state changes over time.
3. `embedtrace` $(\text{vocab\_size}, \text{dim})$: Decayed one-hot accumulator recording byte history.

Updates per step:
$$
\text{embedtrace}_t = \text{decay}_t \odot \text{embedtrace}_{t-1} + \mathbf{1}_{c_t}
$$
$$
\text{decaytrace}_t = \text{decay}_t \odot \text{decaytrace}_{t-1} + \text{decay}_t \odot (1 - \text{decay}_t) \odot \text{states}_{t-1}
$$

## Custom Gradient Hooks

To enable long-range memory updates beyond standard truncated backpropagation, custom gradient hooks adjust the encoder embedding weights and decay parameters directly:

1. **Embedding weights**:
$$
\nabla_{W_{\text{embed}}} \leftarrow \nabla_{W_{\text{embed}}} + \frac{\partial \mathcal{L}}{\partial \text{state}_i} \otimes (\text{embedtrace}_{i} \odot \text{decay}_i)
$$
*(Gradient adds to autodiff)*

2. **Decay parameters**:
$$
\nabla_{\text{decay}_i} \leftarrow \frac{\partial \mathcal{L}}{\partial \text{state}_i} \odot \text{decaytrace}_i
$$
*(Gradient replaces autodiff)*

## Detach & Persistence Semantics

In PyTorch, `states`, `decaytrace`, and `embedtrace` are registered buffers detached from the computational graph (`requires_grad=False`). They are updated in-place via `.copy_()` using detached values, preventing the optimizer from erroneously treating them as trainable weights while allowing weights and traces to round-trip through checkpoints.
