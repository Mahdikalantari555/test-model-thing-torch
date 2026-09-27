# Spec Delta

## Purpose

Non-monotonic truth maintenance system (TMS) using AGM belief revision to detect and supersede contradictory propositions instead of allowing conflicting facts to coexist in the knowledge store.

## ADDED Requirements

### Requirement: Contradiction detection via SVO slot clash
The system SHALL detect when a newly taught proposition directly contradicts an existing stored fact by checking for functional-predicate slot clashes (same subject and predicate, different object) and explicit polarity inversion.

#### Scenario: Functional-predicate slot clash
- **WHEN** user teaches "The capital of West Germany is Bonn"
- **WHEN** user later teaches "The capital of West Germany is Berlin"
- **THEN** the system SHALL detect the slot clash on the `capital_of` functional predicate
- **THEN** the older fact SHALL be marked as `superseded_by` the newer fact

#### Scenario: Polarity inversion
- **WHEN** user teaches "Remote sensing requires physical contact"
- **WHEN** user later teaches "Remote sensing does not require physical contact"
- **THEN** the system SHALL detect the explicit negation polarity inversion
- **THEN** the older fact SHALL be marked as superseded

### Requirement: Non-monotonic superseding
The system SHALL mark outdated facts as `superseded` with `valid_until` and `superseded_by` metadata, excluding them from active retrieval masks while preserving full audit trails.

#### Scenario: Superseded facts excluded from retrieval
- **WHEN** a fact has been marked as superseded
- **WHEN** a query is executed
- **THEN** the superseded fact SHALL NOT appear in the active retrieval results
- **THEN** the superseded fact SHALL remain queryable in the audit log

### Requirement: Two-tier contradiction gating
The system SHALL gate contradiction checks using a two-tier filter: high semantic similarity ($\ge 0.65$) combined with deterministic slot clash or polarity inversion analysis, executing in under 1 ms on CPU.

#### Scenario: Low-latency contradiction check
- **WHEN** a new proposition is embedded and compared against existing facts
- **THEN** the contradiction check SHALL complete in under 1 ms on CPU
- **THEN** false positives from unrelated semantic similarities below 0.65 SHALL be rejected

### Requirement: Preserved audit trail
The system SHALL preserve all facts including superseded ones in the persistent knowledge store with full metadata (timestamps, step counts, novelty scores, supersession chains).

#### Scenario: Audit trail completeness
- **WHEN** a fact is superseded
- **THEN** the original fact text, timestamp, and step count SHALL remain unchanged in the database
- **THEN** the `superseded_by` field SHALL reference the ID of the superseding fact