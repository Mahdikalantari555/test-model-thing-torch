# 07 Numerical Verification

## Guiding decision

Numerical similarity matters more than architectural novelty.
Bit-exact reproduction is NOT expected. Three sources of drift are
unavoidable and must be documented, not hidden:

1. RNG. MLX's PRNG algorithm differs from PyTorch's. The model uses
   random.categorical only (sampling). Weights are NOT initialized from
   RNG in the source — see 01_source_analysis.md. So this affects only
   sampling, not training dynamics. Mitigation: fix torch seeds.

2. Default dtypes / init. MLX Embedding/Linear init schemes differ from
   PyTorch defaults. RESOLVED against source: Embedding is normal(std=1/√dim)
   (must reinit); Linear is uniform(±1/√fan_in) (matches PyTorch's own default).
   This is the biggest controllable drift source.

3. var() convention. mx.var default ddof is 0 (population), RESOLVED. If wrong,
   the variance loss term scales differently and the whole loss curve shifts.

## Verification plan

reports/reproduction_report.md will contain:

### 1. Tensor shape audit
For each component, compare MLX output shape vs PyTorch output shape.
Must match exactly. Automated: run both, assert shapes.

### 2. Parameter count audit
Compare total trainable params. Must equal 4.5M for dim=512, layers=16.
Formula: (256*dim) + layers*(dim*dim + dim + dim + dim) + (256*dim + dim + 1)

### 3. Activation statistics
For a fixed input byte sequence, compare:
- mean/std of each layer's state
- mean/std of decoder logits
- stop value
Must be close (within tolerance), not identical.

### 4. Loss curve comparison
Train both on the same tiny input, plot train loss over steps.
PyTorch should track MLX within a band. Divergence = bug.

### 5. Gradient audit
For a fixed (c, nextb, end), compare grads of:
- encoder.embed.weight
- each layer.decay
- each layer.weights
Must be close. The custom hook grads are the hardest to match.

## Tolerance policy

- Shapes: exact
- Param count: exact
- Activations/losses/grads: relative tolerance 1e-3, documented per metric
- Bit-exact: not claimed

## [RESOLVED] blocking items

All seven confirmed against the installed MLX 0.32.2 source (MLX cannot execute
in this environment, so defaults were read from the package source, not run).
Full citations in `01_source_analysis.md`.

1. Embedding: **normal, std = 1/√dim**. Port must reinit.
2. Linear: **uniform(±1/√fan_in)** — identical to PyTorch's own default. No reinit.
3. LayerNorm: **eps=1e-5, affine=True, bias=True, weight=ones, bias=zeros** —
   identical to PyTorch defaults.
4. var: **ddof=0 (population)**. Port must use `torch.var(unbiased=False)`.
5. Custom grads: **asymmetric** — embed.weight ADDS, decay REPLACES autodiff.
6. states/decaytrace/embedtrace **ARE in trainable_parameters()** — MLX's filter
   has no weight-vs-buffer distinction. They get Adam-updated, clobbering the
   just-set stop_gradient state.
7. AdamW: **weight_decay=0.01 (silent, repo never overrides it), bias_correction=False
   (default)**. Port needs a custom step; stock `torch.optim.AdamW` always applies
   bias correction.

Caveat: items 1, 4, 6, 7 each introduce a real divergence from stock PyTorch
defaults. Numerical similarity (relative 1e-3) is achievable only if the port
matches these MLX defaults explicitly — see §Tolerance policy.