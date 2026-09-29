# Proposal

## Why

`DroidEngine` already has a `PlasticAdapter` contract and an associative RTU, but the current RTU's surprise-scaled outer-product momentum accumulates writes without computing a key-residual correction against the current memory. Its learned retention and erasure-like transform are not a key-targeted `erase()` API, so correlated writes can still saturate or interfere. A gated Delta Rule explicitly corrects a stored association and supports controlled overwriting. Sparse high-dimensional codes should further reduce overlap between learned patterns.

The proposal is motivated by the reported test-time memory results in Titans, the gated delta-rule direction in RWKV-7/Gated DeltaNet, and sparse high-dimensional associative coding in FlyModel. These references motivate experiments; the quantitative targets below are acceptance goals to validate on a fixed benchmark, not claims about current repository performance.

## What Changes

- Add `DeltaPlasticAdapter` implementing the existing `PlasticAdapter` API, with a multi-head gated Delta Rule update:
  `S_t = S_{t-1} * (1 - beta_t) + beta_t * outer(K_t, V_t - K_t^T S_{t-1})`.
- Learn bounded per-head `beta_t` gates with `sigmoid(W_beta x)` and add explicit `erase()` using a Householder-reflected key direction and bounded selective attenuation.
- Add `SparseCoder` for 384-dimensional inputs to 2048-dimensional codes with TopK 128 activations (about 93.75% zeros), plus a sparse/delta composition.
- Register `delta_rtu`, `sparse_delta`, and `fly_model` in the plastic-adapter registry; preserve `associative_rtu` loading for existing profiles.
- Add a reproducible 20-task continual-learning benchmark targeting mean memory loss below 0.10, compared with an EWC reference target of approximately 0.24 under the same protocol.

## Impact

- **New capabilities:** `adaptation/delta-memory` and `adaptation/sparse-coding`.
- **Code areas:** `src/model/rtu.py`, `src/model/plastic_adapter.py`, `src/model/droid.py`, adapter/profile serialization, and new adapter/benchmark tests.
- **Compatibility:** Existing adapter names and saved states remain readable. New profiles may select `delta_rtu`; persisted profiles keep their saved adapter unless explicitly migrated.
- **Dependencies:** No mandatory external dependency; PyTorch remains the numerical backend.
- **Target outcome:** lower interference and less forgetting, with measured memory loss <0.10 on the defined 20-task benchmark versus the EWC comparison target (~0.24).

## Risks

- An incorrectly scaled residual or gate can make updates unstable or overwrite unrelated keys. Normalize key vectors, clamp gates, test no-op/full-write limits, and verify orthogonal-key preservation.
- Householder transforms are orthogonal, but the selective attenuation step is not lossless. Bound erase strength and report both target removal and collateral retrieval change.
- TopK training can produce dead units or unstable gradients. Add reconstruction/sparsity objectives, deterministic tie handling, and code-usage diagnostics.
- Memory-loss numbers vary with task order and benchmark definition. Version the dataset, seeds, optimizer, and EWC baseline; do not publish a comparison without the full protocol.
- Adapter state shapes differ from existing RTU checkpoints. Version state dictionaries and require explicit migration rather than silently reshaping incompatible state.

## Research References

- Titans, *Learning to Memorize at Test Time*: https://arxiv.org/html/2501.00663v1
- RWKV-7 / hybrid-attention research: https://presenc.ai/research/hybrid-attention-models-mamba-jamba-rwkv-2026
- FlyModel sparse associative memory: https://direct.mit.edu/neco/article/35/11/1797/117579/Reducing-Catastrophic-Forgetting-With-Associative
