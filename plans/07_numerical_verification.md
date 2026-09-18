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
   PyTorch defaults. [VERIFY] the MLX schemes and match them, else the
   first forward differs. This is the biggest controllable drift source.

3. var() convention. mx.var default ddof is [VERIFY]. If wrong, the
   variance loss term scales differently and the whole loss curve shifts.

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

## [VERIFY] blocking items

These must be confirmed against the MLX runtime before verification can
be trusted. If any is wrong, the comparison is invalid.

1. MLX Embedding init scheme
2. MLX Linear init scheme
3. MLX LayerNorm init (weight/bias, eps)
4. mx.var ddof (population vs sample)
5. Whether custom grads ADD to or REPLACE autodiff grads
6. Whether states/decaytrace/embedtrace are in trainable_parameters()
7. MLX AdamW bias correction and weight decay