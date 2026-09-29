# Design

## Architecture

Keep the current `PlasticAdapter` contract and `associative_rtu` implementation intact for checkpoint compatibility. Add a `DeltaPlasticAdapter` beside it and make adapter selection explicit through `get_adapter()`. New profiles can opt into `delta_rtu`; legacy profiles continue loading their recorded adapter. The Delta adapter operates on normalized per-head keys and values, predicts the value already associated with a key (`K_t^T S_{t-1}`), and writes only the residual. A learned sigmoid gate controls the amount of decay and update for each head.

For sparse variants, compose the 384-dimensional anchor output with a learned `SparseCoder`: encoder to 2048 features, deterministic TopK-128 activation, and decoder back to the adapter's output dimension. `sparse_delta` exposes this composition directly. `fly_model` is the FlyModel-style configured sparse/delta profile using the same public adapter contract; it is not a second incompatible memory API.

A 20-task benchmark compares the dense delta adapter, sparse/delta adapter, and EWC baseline with identical data order, optimizer budget, random seeds, evaluation points, and task-retention metric. Checkpoints include adapter type/version, gate and coder parameters, sparse-code configuration, and memory state.

## Components

- **`DeltaPlasticAdapter`** — implements `update`, `retrieve`, `predict_next`, `compute_surprise`, `state_dict`, and `load_state_dict`; projects inputs to per-head `K`/`V`, computes the Delta residual, and updates `S` under `no_grad` or a documented differentiable training path.
- **Gated Delta update** — `beta_t = sigmoid(W_beta x_t)` is computed per head and bounded to `[0, 1]`. The residual is `V_t - K_t^T S_{t-1}`. Tests cover a zero gate, a saturated gate, repeated identical writes, and a changed target value.
- **Selective erase** — build a Householder reflection that maps the normalized selected key direction to a designated basis direction; attenuate only that reflected row by the requested bounded erase strength; apply the inverse reflection. The operation returns a report of target-key recall and collateral change.
- **`SparseCoder`** — encode 384-D vectors to 2048-D activations, keep the 128 largest-magnitude entries, and decode/reconstruct. It exposes active-count, reconstruction error, and per-feature utilization for diagnostics.
- **Registry and profile loading** — map `delta_rtu` to the dense adapter, `sparse_delta` to the sparse/delta composition, and `fly_model` to the FlyModel-configured adapter. Loading an old `associative_rtu` checkpoint never selects a new class implicitly.
- **Benchmark/tests** — define memory loss as the mean drop in held-out task score from the best score observed after each task to the final score, and publish per-task as well as aggregate results.

## Alternatives

- **Continue additive outer-product updates:** rejected as the only long-term path because it does not use a residual prediction to correct an existing association and does not expose targeted erasure.
- **Replace the existing adapter in place:** rejected for the first release because saved state keys and semantics differ; use an additive adapter name and an explicit migration path.
- **Use only dense Delta memory:** useful as a baseline, but it does not address correlated-pattern interference as directly as sparse codes.
- **Use hard deletion without reflection:** simpler, but can disturb correlated values sharing a projection direction; the Householder-based operation makes the selected direction explicit and measurable.
- **Use an unstructured encoder with no TopK constraint:** rejected because a fixed, inspectable sparsity budget is needed for interference and memory accounting.

## Risks / Trade-offs

- A 2048-D code increases projection compute even though it lowers code overlap. Benchmark end-to-end latency and memory, not only nonzero count.
- Householder erasure depends on a stable key projection; degenerate or near-zero keys must be rejected or handled as a no-op.
- Dense and sparse adapters need different state shapes. Versioned checkpoints and conversion tests are required.
- The `<0.10` memory-loss goal is a research target. If the benchmark cannot reproduce the EWC reference, report the delta and revise the target rather than changing the task protocol.
