# Spec Delta

## Purpose

Specify a gated Delta Rule plastic memory adapter with learned forgetting, explicit selective erasure, versioned persistence, and a reproducible continual-learning target.

## ADDED Requirements

### Requirement: Gated Delta Rule updates
The system SHALL provide a `DeltaPlasticAdapter` implementing the `PlasticAdapter` contract. For each head and update, it SHALL compute the residual `R_t = V_t - K_t^T S_{t-1}` and update memory according to `S_t = S_{t-1} * (1 - beta_t) + beta_t * outer(K_t, R_t)`, with a bounded gate `beta_t`.

#### Scenario: Correct an existing association
- **WHEN** an adapter first stores key `K` with value `V1` and then receives the same key with value `V2`
- **THEN** the second write SHALL use the value residual relative to `K^T S` rather than add an uncorrected outer product
- **THEN** retrieval for `K` SHALL move toward `V2` without unbounded state growth

#### Scenario: Gate bounds update and forgetting
- **WHEN** the learned `beta_t` gate is evaluated for any finite input
- **THEN** every head's gate SHALL be in `[0, 1]`
- **WHEN** `beta_t` is zero
- **THEN** the associative state SHALL remain unchanged
- **WHEN** `beta_t` is one
- **THEN** the state SHALL apply the full residual correction and the specified retention behavior

### Requirement: Learned gated forgetting
The system SHALL compute `beta_t` from a learned `W_beta` projection followed by sigmoid, and SHALL expose the effective per-head gate in diagnostics so update/forgetting behavior can be tested and audited.

#### Scenario: Per-head gates
- **WHEN** an input is processed by a multi-head adapter
- **THEN** `W_beta` SHALL produce a gate for each configured head
- **THEN** changing one head's gate SHALL NOT silently alter the configured gates of other heads

### Requirement: Householder selective erasure
The adapter SHALL expose `erase(key, strength)` that uses a Householder reflection to isolate a selected key direction and attenuates that direction by a bounded strength in `[0, 1]` without applying global decay to all memory.

#### Scenario: Erase one association
- **WHEN** `erase(key, strength=1.0)` is called for a stored key
- **THEN** recall for that selected association SHALL be reduced to the configured erase tolerance
- **THEN** recall for an orthogonal control key SHALL remain within 0.05 cosine similarity of its pre-erasure score
- **THEN** the result SHALL report the selected key's before/after score and collateral change

#### Scenario: Invalid erase request
- **WHEN** erase strength is outside `[0, 1]` or the key is non-finite
- **THEN** the adapter SHALL reject the request with a clear validation error and SHALL NOT mutate memory

### Requirement: Continual-learning memory-loss target
The project SHALL evaluate the adapter on a versioned 20-task benchmark using the same task order, budget, seeds, and metric for Delta and EWC. Mean memory loss SHALL target less than `0.10`; the comparison SHALL include the EWC reference target of approximately `0.24` and publish actual results and protocol.

#### Scenario: Run the 20-task benchmark
- **WHEN** the benchmark completes all 20 tasks
- **THEN** it SHALL report per-task retention and aggregate mean memory loss for both Delta and EWC
- **THEN** a result SHALL be described as meeting the target only if measured Delta memory loss is below `0.10` under the published protocol
- **THEN** missing or non-comparable EWC results SHALL be reported as such rather than inferred

### Requirement: Versioned adapter persistence
The adapter SHALL save and restore its parameters, associative state, gate configuration, and format version, and SHALL reject incompatible state shapes rather than silently reinterpret them.

#### Scenario: State round trip
- **WHEN** a Delta adapter is saved and loaded into a compatible adapter
- **THEN** subsequent retrieval and update results SHALL match within the configured numerical tolerance

#### Scenario: Legacy adapter selection
- **WHEN** a profile declares `associative_rtu`
- **THEN** loading that profile SHALL continue to construct the legacy adapter and SHALL NOT silently substitute `delta_rtu`
