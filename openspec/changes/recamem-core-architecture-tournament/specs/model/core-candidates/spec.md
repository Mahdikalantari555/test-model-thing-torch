# Spec Delta

## Purpose

Provides a uniform sequence-mixing core contract and parameter-matched configurations so all seven candidate backbones (recurrent diagonal, selective SSM, matrix decay, gated-delta, attention-only SAN, and standard Transformer) can be trained and evaluated interchangeably without confounding changes to surrounding model components.

## ADDED Requirements

### Requirement: Uniform sequence-mixing core contract
The system SHALL expose a common sequence-core interface that accepts batches of token embeddings, optional recurrent or bounded-window state from a prior segment, and returns sequence hidden representations alongside updated state and state-footprint metadata. Replacing one registered core with another SHALL NOT require modifying the tokenizer, loss objective, decision readout, router, or evaluation harness.

#### Scenario: Core swap preserves surrounding pipeline
- **WHEN** a model configuration changes only the registered sequence-core identifier among the seven candidate backbones
- **THEN** the forward pass SHALL return hidden states of identical batch, sequence-length, and model-dimension shape
- **THEN** the training loop and evaluation harness SHALL execute without code or data-pipeline changes

#### Scenario: Stateful chunk continuation
- **WHEN** a sequence of length $2C$ is processed either as one pass of length $2C$ or as two sequential chunks of length $C$ carrying state forward
- **THEN** every recurrent candidate core (`C0`–`C4`) SHALL produce numerically equivalent outputs within documented floating-point tolerance (`1e-4` in fp32)
- **THEN** calling core reset SHALL zero or clear all carried sequence state deterministically

### Requirement: Seven isolated candidate sequence cores
The system SHALL provide seven distinct, selectable sequence-mixing cores for controlled comparison:
1. `C0` (`unstacked_rtu`): parallel input-independent diagonal exponential moving average layers reading the raw token embedding only;
2. `C1` (`stacked_gated_rtu`): stacked residual layers with input-dependent forget gates initialized across log-spaced timescales and convex diagonal vector state updates;
3. `C2` (`selective_ssm`): stacked selective diagonal state-space layers with input-dependent step/projection parameters and short causal convolution;
4. `C3` (`retnet_decay_matrix`): stacked multi-head matrix-state recurrence with scalar/head decay and additive outer-product writes without delta-rule overwrite;
5. `C4` (`gated_delta_matrix`): stacked multi-head gated-delta recurrence updating matrix state via $S_t = \alpha_t S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top$ with L2-normalized keys/queries, chunkwise parallel scan, output gating, SwiGLU, and configurable short causal convolution (`conv_on` / `conv_off`);
6. `C5` (`needle_san`): attention-only sliding-window grouped-query attention with QK-normalization, pinned sink prefix positions, and lightweight/Hadamard channel mixing;
7. `C6` (`softmax_transformer`): causal multi-head attention with rotary positional embeddings, QK-normalization, pre-RMSNorm, and SwiGLU feed-forward blocks.

#### Scenario: Every candidate core instantiates and reports state complexity
- **WHEN** each of the seven candidate core identifiers (`C0` through `C6`) is requested from the registry
- **THEN** the system SHALL construct a runnable model and report its state scaling class (`constant_vector`, `constant_matrix`, `sliding_window_kv`, or `linear_kv`) and per-sequence state size in bytes
- **THEN** toggling short causal convolution off on `C4` SHALL disable the depthwise convolution while keeping the gated-delta recurrence unchanged

### Requirement: Parameter-budget matching across tiers
The system SHALL compute deterministic width, head, and feed-forward dimensions for each candidate core so that all candidates within a comparison tier (`micro` ~2M, `small` ~6M, `target` ~17M) match the target trainable parameter budget within `±2.0%` under the same vocabulary size.

#### Scenario: Matched tier configurations stay within tolerance
- **WHEN** tier configurations are generated for `C0` through `C6` at a shared vocabulary size and tier target (`2_000_000`, `6_000_000`, or `17_000_000`)
- **THEN** every candidate's actual trainable parameter count SHALL fall within `±2.0%` of the tier target
- **THEN** requesting an unsatisfiable parameter budget SHALL fail with a descriptive configuration error rather than silently running an unmatched model size
