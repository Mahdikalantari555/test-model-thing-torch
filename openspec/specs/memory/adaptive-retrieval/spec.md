# adaptive-retrieval Specification

## Purpose

Adaptive retrieval policy: the system selects retrieval strategy (episodic-only, semantic-only, hybrid RRF, contextual-boosted) based on query type and user feedback history, optimizing for precision or recall as needed.

## Requirements

### Requirement: Query classification for strategy selection
The system SHALL classify queries into types: `definitional`, `specific_fact`, `temporal`, `relational`, `ambiguous` and select default strategy per type.

#### Scenario: Definitional query uses semantic store
- **WHEN** query classified as `definitional`
- **THEN** default strategy SHALL be `semantic_primary`

#### Scenario: Temporal query uses episodic with recency
- **WHEN** query classified as `temporal`
- **THEN** default strategy SHALL be `episodic_recency_weighted`

### Requirement: Feedback-driven strategy adaptation
The system SHALL track user feedback (approve/reject) per strategy and query type, shifting probability toward strategies with higher approval rates.

#### Scenario: Strategy probability updates on feedback
- **WHEN** user approves result from `hybrid_rrf` strategy
- **THEN** `hybrid_rrf` probability SHALL increase for that query type
- **WHEN** user rejects result from `episodic_only`
- **THEN** `episodic_only` probability SHALL decrease

### Requirement: Strategy exploration
The system SHALL occasionally (epsilon-greedy, $\epsilon=0.1$) try alternative strategies to discover better policies.

#### Scenario: Exploration tries alternatives
- **WHEN** $\epsilon$-greedy triggers
- **THEN** a non-default strategy SHALL be selected
- **THEN** outcome SHALL update strategy probabilities

### Requirement: Policy persistence
The system SHALL persist strategy probabilities per Droid profile and restore on load.

#### Scenario: Policy survives restart
- **WHEN** Droid profile is saved and reloaded
- **THEN** strategy probabilities SHALL match pre-save values