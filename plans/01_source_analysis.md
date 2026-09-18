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

## [VERIFY] Items needing MLX runtime confirmation

1. nn.Embedding init scheme and nn.Linear init scheme in MLX.
   Affects identical initialization (Component A requirement).
2. MLX var(x) default: unbiased (ddof=1) or population (ddof=0)?
   Directly affects the variance loss magnitude.
3. Whether MLX LayerNorm uses bias by default and its init.
4. MLX AdamW bias correction and weight decay placement.
5. Whether the hand-written grad hooks override or augment autodiff grads.
6. Whether states/decaytrace/embedtrace are trainable_parameters() or not
   (they are mutated in place, not via optimizer).