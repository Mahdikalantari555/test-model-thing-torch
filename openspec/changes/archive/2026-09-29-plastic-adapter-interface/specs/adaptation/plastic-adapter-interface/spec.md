# Spec Delta

## Purpose

Abstract `PlasticAdapter` interface defining the contract between Droid orchestration and plastic memory implementations (associative RTU, LoRA, Mamba, etc.), enabling hot-swappable memory mechanisms.

## ADDED Requirements

### Requirement: PlasticAdapter abstract base class
The system SHALL define an abstract base class `PlasticAdapter` with required methods: `update(x: Tensor) -> Tensor`, `retrieve(q: Tensor) -> Tensor`, `predict_next() -> Tensor`, `compute_surprise(x: Tensor) -> tuple[Tensor, float]`, `state_dict() -> dict`, `load_state_dict(dict)`.

#### Scenario: Concrete adapter implements all methods
- **WHEN** a class inherits from `PlasticAdapter`
- **THEN** it MUST implement all abstract methods
- **THEN** `isinstance(adapter, PlasticAdapter)` SHALL be True

### Requirement: Update with input embedding
The system SHALL accept an input embedding $x \in \mathbb{R}^D$ and update internal plastic state, returning the adapted output.

#### Scenario: Update returns adapted output
- **WHEN** `adapter.update(x)` is called
- **THEN** internal state SHALL be updated
- **THEN** output tensor of shape $(D,)$ SHALL be returned

### Requirement: Retrieve with query embedding
The system SHALL accept a query embedding $q \in \mathbb{R}^D$ and return a retrieved/conditioned representation.

#### Scenario: Retrieve conditions on query
- **WHEN** `adapter.retrieve(q)` is called
- **THEN** output SHALL reflect plastic memory conditioned on $q$
- **THEN** output shape SHALL be $(D,)$

### Requirement: Predict next expectation
The system SHALL generate a top-down prediction $\hat{e}_t$ from current state for surprise computation.

#### Scenario: Prediction available before input
- **WHEN** `adapter.predict_next()` is called
- **THEN** prediction tensor of shape $(D,)$ SHALL be returned

### Requirement: Surprise computation
The system SHALL compute prediction error and surprise scalar against an input.

#### Scenario: Surprise for novel input
- **WHEN** `adapter.compute_surprise(x)` called with orthogonal input
- **THEN** returns (error_vector, surprise≈1.0)

### Requirement: State serialization
The system SHALL serialize and deserialize complete plastic state via `state_dict()`/`load_state_dict()`.

#### Scenario: Round-trip state preservation
- **WHEN** `state = adapter.state_dict(); new_adapter.load_state_dict(state)`
- **THEN** `new_adapter` SHALL behave identically to `adapter`