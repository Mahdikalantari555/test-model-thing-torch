# Spec Delta

## Purpose

Specify sparse high-dimensional associative codes for reducing pattern overlap and interference in continual learning.

## ADDED Requirements

### Requirement: 384-to-2048 sparse encoding
The system SHALL provide a `SparseCoder` that accepts a 384-dimensional input, produces a 2048-dimensional latent code, and retains the 128 largest-magnitude activations for each non-empty example.

#### Scenario: Encode with fixed TopK budget
- **WHEN** a finite 384-dimensional embedding is encoded
- **THEN** the returned code SHALL have dimension 2048
- **THEN** exactly 128 positions SHALL be active after TopK selection, with deterministic tie handling
- **THEN** measured zero sparsity SHALL be 93.75% for a code with 128 nonzero values

#### Scenario: Invalid input shape
- **WHEN** an input does not have the configured 384-dimensional shape
- **THEN** the coder SHALL raise a clear shape error or use an explicitly configured projection
- **THEN** it SHALL NOT silently truncate or pad the input

### Requirement: Sparse code reconstruction
The coder SHALL provide a decoder or reconstruction path and SHALL report reconstruction error and active-feature utilization so sparse coding quality can be evaluated separately from memory retrieval.

#### Scenario: Encode and decode
- **WHEN** an input is encoded and decoded
- **THEN** the output SHALL return to the configured 384-dimensional embedding space
- **THEN** reconstruction metrics SHALL be finite and available for benchmark reporting

### Requirement: Sparse adapter registry integration
The plastic-adapter registry SHALL expose `sparse_delta` and `fly_model` entries that compose sparse coding with the Delta adapter while satisfying the common adapter API. `delta_rtu` SHALL remain available as the dense Delta option.

#### Scenario: Resolve sparse adapter names
- **WHEN** `get_adapter("sparse_delta")` or `get_adapter("fly_model")` is called
- **THEN** each name SHALL resolve to a constructible adapter with the configured 384-to-2048 TopK-128 pipeline
- **THEN** save/load SHALL preserve coder, adapter, and code-shape configuration

### Requirement: Reduce cross-task interference
On the same fixed pattern set and update budget, sparse/delta memory SHALL report active-set overlap and retrieval interference against a dense 384-dimensional Delta baseline. The benchmark target SHALL be at least 20% lower mean cross-task interference without more than a 5 percentage-point loss in retrieval quality.

#### Scenario: Compare sparse and dense interference
- **WHEN** dense and sparse/delta adapters are evaluated on the same correlated multi-task patterns
- **THEN** the report SHALL include mean pairwise active-set overlap, retrieval quality, and task-retention scores
- **THEN** the result SHALL meet the target only when measured interference is at least 20% lower and retrieval-quality loss is no more than 5 percentage points
