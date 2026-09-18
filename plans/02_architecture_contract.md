# 02 Architecture Contract

This is the interface contract the PyTorch implementation must satisfy.
Tests encode it. Implementation just satisfies tests.

## Global conventions

- Byte vocabulary V = 256.
- All hidden work in float32 (MLX default). Pin dtypes explicitly in PyTorch.
- dim = 512, layers = 16 for the 4.5M config.
- Single sequence element at a time (no batching in the recurrent core).
  Batched only at the data/pipeline level if needed.

## Encoder

Inputs:  c: int in [0, 255]
Output:  tensor shape (dim,), dtype float32
Params:  weight (256, dim)
Init:    [VERIFY] MLX Embedding init scheme
Test:    tests/test_embedding.py
         - shape (dim,)
         - identical dims
         - [VERIFY] whether init values can be reproduced

## Layer (RTU)

Inputs:  enc (dim,), x (dim,), dummy (dim,)
Output:  x_out (dim,), state (dim,), decay (dim,)  all float32
Params:  decay (dim,), weights (dim, dim, bias=False),
         norm.weight (dim,), norm.bias (dim,)
State:   states (dim,), decaytrace (dim,), embedtrace (256, dim)

Equations:
  decay     = sigmoid(decay_param)
  state     = decay * states + enc + dummy
  x_out     = x + silu(weights(layer_norm(state)))
  return x_out, state, decay

Test: tests/test_encoder.py (blocks), tests/test_rtu.py (memory)

## Model

Inputs:  currb: int, nextb: int | None, end: bool, notrace: bool
Output:  (sampled_byte: int, stop: float)
Side:    updates persistent state, optimizer, writes checkpoints via Runtime

Forward path (step):
  enc = encoder(currb)
  x = enc
  for layer: x, state, decay = layer(enc, x, dummies[i])
  return (x, states, decays), decoder(x)

## Decoder

Inputs:  x (dim,)
Output:  logits (256,), stop (float in [0,1])
Params:  decode Linear(dim,256), stop Linear(dim,1)

## Latent predictor objective

When nextb is not None:
  tgt = stop_gradient(encoder(nextb))   # target is the NEXT byte's embedding
  pred_mse = mean(square(x - tgt))
So the model predicts the next byte's latent representation from the
current latent. This is the JEPA-style latent prediction.

## Losses (exact)

  L_var = max(0, 1 - sqrt(var(x) + 1e-4))
  L_pred = mean(square(x - tgt))              # only if nextb is not None
  L_ce   = -log_softmax(output)[nextb]
           = -output[nextb] + logsumexp(output)   # only if nextb is not None
  L_stop = mean(square(stop - (1.0 if end else 0.0)))  # only if nextb is not None
  L_total = L_var + L_pred + L_ce + L_stop

[VERIFY] var(x) ddof: population or sample?

## Custom gradient hooks (CRITICAL)

These run AFTER value_and_grad returns grads. They mutate grads and
persistent state. They are the memory mechanism.

For each layer i, with dlds = dL/d(state_i):

  embedtrace_i = embedtrace_i * decay_i + (arange(256) == c)[:, None]
  grads[encoder.embed.weight] += dlds * (embedtrace_i * decay_i)

  decaytrace_i = decay_i * decaytrace_i + decay_i * (1 - decay_i) * states_i
  grads[layers][i].decay = dlds * decaytrace_i

  states_i     = stop_gradient(states_i)
  decaytrace_i = stop_gradient(decaytrace_i)
  embedtrace_i = stop_gradient(embedtrace_i)
  eval(states_i, decaytrace_i, embedtrace_i)

Interpretation: embedtrace is a decayed one-hot accumulator of the
input byte. decaytrace is a decayed accumulator of the state. Both feed
custom gradients into the encoder embedding and the decay gate.

[VERIFY] Do these ADD to autodiff grads or REPLACE them?
[VERIFY] Are states/decaytrace/embedtrace excluded from trainable_parameters()?

## Sampling

  probs = softmax(output)
  entropy = -sum(probs * log(probs + 1e-8)) / log(256)   # in bits, 0..1
  temp = max(0.1, temp_param * (1 - temp_param * entropy))
  return categorical(output / temp).item()

## Reset

zeros states, decaytrace, embedtrace for every layer.

## Save / Load

Every 500 steps. Atomic (temp + os.replace).
Keys: m.<param>, o.<opt>, state.<i>, decaytrace.<i>, embedtrace.<i>