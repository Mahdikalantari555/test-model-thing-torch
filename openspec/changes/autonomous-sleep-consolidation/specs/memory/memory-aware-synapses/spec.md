# Spec Delta

## Purpose

Specify MAS-style parameter importance estimation, reversible pruning of low-importance plastic weights, and quality reporting.

## ADDED Requirements

### Requirement: Squared-L2 output-sensitivity importance
The system SHALL estimate importance for eligible plastic-adapter parameters from the squared-L2 output objective. For each sampled input batch it SHALL perform one backward pass and accumulate a documented sensitivity statistic, such as the running mean of `abs(gradient)` for `||f(x)||_2^2`.

#### Scenario: Estimate importance without labels
- **WHEN** representative unlabeled inputs are provided to `MemoryAwareSynapses`
- **THEN** importance SHALL be computed from the squared-L2 output objective without requiring task labels
- **THEN** one backward pass per sampled batch SHALL contribute to the running estimate
- **THEN** each eligible parameter SHALL receive a finite, non-negative importance score

### Requirement: Prune ten percent of low-importance eligible weights
The system SHALL select the lowest-importance 10% of eligible plastic-adapter parameters using deterministic tie handling and SHALL preserve a reversible pruning mask. Frozen anchor weights and non-parameter memory buffers SHALL be excluded.

#### Scenario: Prune low-importance parameters
- **WHEN** pruning is requested with the default ratio on an adapter with eligible parameters
- **THEN** the operation SHALL mask exactly 10% of eligible parameter elements, subject to integer rounding documented in the report
- **THEN** higher-importance elements SHALL not be selected ahead of lower-importance elements solely due to iteration order
- **THEN** the returned report SHALL include selected count, total eligible count, threshold, and before/after quality metrics

#### Scenario: Restore pruned state
- **WHEN** a checkpoint containing MAS scores and pruning masks is loaded
- **THEN** the same mask and importance scores SHALL be restored
- **THEN** an explicit restore operation SHALL recover original parameter values from the reversible checkpoint data

### Requirement: Pruning scope and quality audit
MAS pruning SHALL only mutate parameters in an explicit allowlist and SHALL report retrieval/retention quality before and after pruning.

#### Scenario: Protect frozen and stateful components
- **WHEN** MAS pruning runs
- **THEN** it SHALL NOT change frozen semantic-anchor parameters or non-parameter associative state buffers
- **THEN** a quality regression SHALL be recorded in the report rather than omitted
