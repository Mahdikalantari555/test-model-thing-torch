# 08 Test Contracts

Each test encodes the architecture contract from 02_architecture_contract.md.
Tests are written BEFORE implementation (TDD). Implementation must satisfy.

## tests/test_embedding.py
- Constructor: nn.Embedding(256, dim)
- forward(byte) -> shape (dim,)
- forward(0) != forward(255) (different rows)
- dtype float32
- [VERIFY] init values match MLX scheme

## tests/test_encoder.py
- Encoder(byte) -> (dim,)
- Stack N encoder blocks: same tensor flow, same normalization,
  same residual structure
- residual: x_out = x + block(x)
- LayerNorm applied before the linear

## tests/test_rtu.py  (CRITICAL)
- state update: state == decay * states + enc + dummy
- decay == sigmoid(decay_param), in (0,1)
- decay=1 -> state = states + enc + dummy (full retention)
- decay=0 -> state = enc + dummy (full forget)
- reset() zeros states, decaytrace, embedtrace
- gradients flow: dL/d(enc), dL/d(decay_param) are non-zero
- embedtrace update: embedtrace == embedtrace*decay + onehot(c)
- decaytrace update: decaytrace == decay*decaytrace + decay*(1-decay)*states
- stop_gradient semantics: persistent state is detached

## tests/test_predictor.py
- latent predictor objective: target = encoder(nextb)
- pred = current latent x
- mse = mean(square(x - target))
- only computed when nextb is not None

## tests/test_losses.py
- L_var = max(0, 1 - sqrt(var(x) + 1e-4))
- L_ce  = -output[nextb] + logsumexp(output)
- L_pred = mean(square(x - tgt))
- L_stop = mean(square(stop - (1.0 if end else 0.0)))
- L_total = L_var + L_pred + L_ce + L_stop
- [VERIFY] var ddof

## tests/test_sampling.py
- softmax(output) sums to 1
- entropy in [0, 1] bits
- temp = max(0.1, temp_param * (1 - temp_param * entropy))
- categorical(output/temp) returns int in [0, 255]

## tests/test_grad_hooks.py  (CRITICAL)
- custom grads for encoder.embed.weight exist and are non-zero
- custom grads for layer.decay exist and are non-zero
- grads ADD to (or REPLACE) autodiff grads — [VERIFY] which
- states/decaytrace/embedtrace NOT in trainable params — [VERIFY]

## tests/test_checkpoint.py
- save then load round-trips params, optimizer state, and all traces
- after load, forward pass reproduces pre-save output
- atomic save (temp file + replace)
- reset() clears persistent state

## tests/test_data.py
- len(), getitem(), collate_fn() interface
- streaming support
- deterministic sampling with fixed seed
- TinyStories, SimpleWiki, custom text corpus