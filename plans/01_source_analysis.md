# 01 Source Analysis

## Source of truth

Fork: https://github.com/Mahdikalantari555/test-model-thing-torch
Files read in full: main.py (253 lines), benchmark.py (94 lines), README.md

IMPORTANT: The fork is still MLX code. Both files `import mlx`.
It is the source of truth to translate FROM, not a PyTorch port.

## What the model is

A byte-level recurrent language model with internal memory state.
Not an LLM. Proof of concept, 4.5M params, trained ~12h on simplewiki.

## Components

### Encoder (main.py:6-11)
- nn.Embedding(256, dim). Byte vocabulary 0..255.
- __call__(x) -> embed(x). Output shape (dim,).

### Decoder (main.py:13-19)
- decode: Linear(dim, 256). Output logits over bytes.
- stop: Linear(dim, 1). Sigmoid -> stop probability.
- __call__(x) -> (decode(x), sigmoid(stop(x)))

### Layer / RTU (main.py:21-39)
Learnable params: decay (dim,), weights Linear(dim,dim,bias=False), norm LayerNorm(dim).
Persistent state (saved/loaded): states (dim,), decaytrace (dim,), embedtrace (256, dim).
Non-learnable buffers: decaytrace, embedtrace.

Forward(enc, x, dummy):
  decay = sigmoid(self.decay)              # elementwise gate in (0,1)
  state = decay * self.states + enc + dummy
  return x + silu(weights(norm(state))), state, decay

So: new_state = decay * old_state + enc + dummy.
Residual: x_out = x + silu(W(norm(state))).

### Model (main.py:41-165)
- layers: list of Layer, length = layers
- optimizer: AdamW(lr)
- sample(output): softmax, entropy in bits, temp = max(0.1, temp*(1 - temp*entropy)),
  return categorical(output/temp).item()
- reset(): zeros all states, decaytrace, embedtrace for every layer.
- step(c, dummies): runs encoder then all layers, returns (x, states, decays), decoder(x)
- __call__(currb, nextb, end, notrace): the training/eval entry point.

### Losses (main.py:96-106), inside value_and_grad
  loss = max(0, 1 - sqrt(var(x) + 1e-4))          # variance loss
  if nextb is not None:
    tgt = stop_gradient(encoder(nextb))
    loss += mean(square(x - tgt))                  # pred mse
    loss += -output[nextb] + logsumexp(output)     # crossentropy (exact form)
    loss += mean(square(stop - (1.0 if end else 0.0)))  # stop mse
  return loss, (states, decays, output, stop)

### Custom gradient hooks (main.py:114-128)
For each layer i, given dlds = dL/d(state_i):
  embedtrace = embedtrace * decay_i + (arange(256)==c)[:,None]
  grads[encoder][embed][weight] += dlds * (embedtrace * decay_i)
  decaytrace = decay_i * decaytrace + decay_i*(1-decay_i)*states_i
  grads[layers][i][decay] = dlds * decaytrace
  states_i = stop_gradient(states_i)             # persistent state update
  decaytrace_i = stop_gradient(decaytrace_i)
  embedtrace_i = stop_gradient(embedtrace_i)
  eval(...)                                       # force materialization

These are hand-written gradient corrections for the persistent state.
They are NOT produced by autodiff. This is the most delicate part.

### Runtime (main.py:167-252)
- save(): every 500 steps. Saves params, optimizer state, and per-layer
  states/decaytrace/embedtrace. Atomic via temp file + os.replace.
- load(): reads m.*, o.*, state.*, decaytrace.*, embedtrace.* prefixes.
- dataset(): reads wikipedia_clean/**/wiki_*, line by line, pairwise bytes.
- chat(): user input loop, then generate until stop > threshold (0.35).

## Hyperparameters (main.py:252)
dim=512, layers=16, temp=0.75, lr=5e-4, threshold=0.35

## Parameter count (main.py:253 comment)
(256*dim) + (dim*dim + dim*2 + dim) + (256*dim + dim + 1)
Per layer: dim*dim (weights) + dim (norm weight) + dim (norm bias) + dim (decay)
Decoder: 256*dim (decode) + dim (stop weight) + 1 (stop bias)
Encoder: 256*dim
For dim=512, layers=16 -> 4.5M. Matches README.

## RESOLVED — MLX 0.32.2 defaults (verified against installed source)

MLX cannot execute in this environment (Linux wheel is CUDA-only;
`libmlx.so` absent, no GPU). The defaults below were confirmed by reading
the installed `mlx` 0.32.2 package source directly, not by running it.
This supersedes every `[VERIFY]` annotation in this repo's plans.

1. **nn.Embedding init** — `nn/layers/embedding.py:17-18`
   `scale = sqrt(1/dims); weight = mx.random.normal(shape=(V,dims), scale=scale)`
   → **normal, std = 1/√dim**. PyTorch `nn.Embedding` default is `normal(0,1)`
   (unscaled). The port MUST reinit: `nn.init.normal_(emb.weight, std=1/sqrt(dim))`.

2. **nn.Linear init** — `nn/layers/linear.py:62-68`
   `scale = sqrt(1/input_dims); weight,bias = uniform(-scale, scale)`
   → **uniform(±1/√fan_in)**. This is exactly PyTorch's own `nn.Linear` default
   (`kaiming_uniform_(a=√5)` reduces to bound `1/√fan_in`). **No reinit needed.**

3. **LayerNorm** — `nn/layers/normalization.py`, `LayerNorm.__init__`
   `eps=1e-5, affine=True, bias=True, weight=ones, bias=zeros`
   → **identical to PyTorch `nn.LayerNorm` defaults**. No action.

4. **mx.var ddof** — `core/__init__.pyi:2513`
   `def var(a, ..., ddof=0, ...)` → **population variance (ddof=0)**.
   PyTorch `torch.var` defaults to `unbiased=True` (ddof=1). The port MUST pass
   `unbiased=False`, else the variance loss term is scaled wrong.

5. **MLX AdamW** — `optimizers.py:579-580, 538-545`
   `weight_decay=0.01` (default — the repo's `opt.AdamW(learning_rate=lr)` call
   never overrides it, so **weight decay 0.01 is silently active**),
   `bias_correction=False` (default). Uncorrected path is literally
   `parameter - lr * m / (sqrt(v) + eps)`. Bias correction is gated behind an `if`.
   → The port needs a **custom Adam step**; stock `torch.optim.AdamW` always
   applies bias correction with no flag to disable it. Decoupled weight decay
   (`param *= (1 - lr*wd)` before the Adam update, `optimizers.py:590`) matches
   PyTorch's decoupled form, so only the bias-correction piece differs.

6. **Custom grad hooks: add vs replace** — `main.py:117-121`
   Asymmetric: `grads["encoder"]["embed"]["weight"] += ...` (**adds** to autodiff),
   `grads["layers"][i]["decay"] = ...` (**replaces** autodiff — plain `=`, the
   sigmoid-through-decay gradient is discarded).

7. **states/decaytrace/embedtrace in trainable_parameters()?** — YES.
   `nn/layers/base.py:235-243`:
   ```
   def valid_parameter_filter(module, key, value):
       return isinstance(value, (dict, list, mx.array)) and not key.startswith("_")
   def trainable_parameter_filter(module, key, value):
       return Module.valid_parameter_filter(module, key, value) and key not in module._no_grad
   ```
   There is **no type distinction** between a weight and a state buffer — only
   the `_` prefix and the `_no_grad` set. `Layer.states`/`decaytrace`/`embedtrace`
   are plain `mx.array` attributes with no underscore and are never frozen in the
   training path, so they **do** land in `trainable_parameters()`.

   Consequence (`optimizers.py:109`): `apply_gradients` maps over `gradients`
   (`tree_map(self.apply_single, gradients, parameters, self.state)`), so the
   optimizer updates **every key present in grads**, including the trace buffers.
   This runs *after* the loop's `layer.states = stop_gradient(states[i])`, so the
   manually-set clean state is immediately clobbered by a small Adam perturbation
   (~1.6e-3/step at lr=5e-4, b1=0.9, b2=0.999, bias correction off — systematic,
   not noise). The plans previously guessed "almost certainly excluded"; that
   guess was wrong.

   **Decision needed before Phase A:** bug-for-bug (let the optimizer touch the
   buffers, matching MLX numerically) vs. fix (register them as non-learnable
   buffers, matching what the code appears to *intend*).