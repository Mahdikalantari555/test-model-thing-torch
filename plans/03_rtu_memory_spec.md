# 03 RTU Memory Spec (Component C — Critical)

## State update equations

Per layer, per step:

  decay     = sigmoid(decay_param)              # learned scalar per dim
  state     = decay * states + enc + dummy      # EMA-style update
  x_out     = x + silu(W(layer_norm(state)))

So the hidden state is a leaky-integrator / EMA over the encoder output.
`decay` is the retention gate: high decay -> long memory, low -> forget.

## Memory persistence logic

Three persistent (non-learned, non-optimizer) buffers per layer:
- states      (dim,)      — the recurrent hidden state
- decaytrace  (dim,)      — decayed accumulator of the state
- embedtrace  (256, dim)  — decayed one-hot accumulator of the input byte

All three are updated in the custom gradient hook, then stop_gradient'd
and materialized with eval(). They are saved/loaded to disk.

Update rules:
  embedtrace = embedtrace * decay + onehot(c)          # onehot(c) shape (256,)
  decaytrace = decay * decaytrace + decay * (1 - decay) * states

embedtrace tracks "which bytes have been seen recently, weighted by
how long ago via decay". decaytrace tracks "what the state was, weighted
by how much it should still matter".

## Hidden state lifecycle

- Created at init: zeros.
- Updated every step in __call__ (training and chat).
- Preserved across save/load (so memory carries from training into chat).
- reset() zeros everything.

## Reset behavior

Model.reset() -> for each layer: states, decaytrace, embedtrace := zeros.

## Gradient flow

The custom hooks inject gradients:
  d(embed.weight) += dlds * (embedtrace * decay)        # from current input history
  d(decay_param)  = dlds * decaytrace                   # from state history

dlds = dL/d(state_i) comes from autodiff through the residual stream.

[VERIFY] Do the custom grads ADD to or REPLACE autodiff grads?
[VERIFY] Are states/decaytrace/embedtrace in trainable_parameters()?
        If yes, they'd be optimized by AdamW, which contradicts the
        manual mutation. Almost certainly excluded.

## Why this is the crux

The whole "infinite memory, no context window" claim rests here.
The recurrent state is the memory. The trace buffers shape what the
gradients teach the embedding and the decay gate. Getting the
stop_gradient semantics wrong (e.g. not detaching, or detaching the
wrong thing) breaks the memory mechanism entirely.

## Translation risk

MLX stop_gradient and PyTorch .detach() should map directly.
MLX eval() forces lazy materialization -> PyTorch is eager, so this is
a no-op but the ORDER of mutations matters. In PyTorch, mutating a
tensor that is part of the autograd graph is fine as long as you
detach the persistent copy. Must keep states as a detached leaf that
is updated in place.