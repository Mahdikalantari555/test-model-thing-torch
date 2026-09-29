# Spec Delta

## Purpose

Contradiction detection and resolution during merge, distillation, and feedback: identify conflicting propositions (same subject-predicate, different object; explicit negation) and apply resolution policies.

## ADDED Requirements

### Requirement: SVO slot clash detection
The system SHALL detect functional-predicate slot clashes: same subject and predicate (e.g., `capital_of`) but different objects.

#### Scenario: Slot clash detected
- **WHEN** propositions "capital of France is Paris" and "capital of France is Lyon" coexist
- **THEN** `detect_contradiction()` SHALL return a `SlotClash` with subject "France", predicate "capital_of", objects ["Paris", "Lyon"]

### Requirement: Polarity inversion detection
The system SHALL detect explicit negation conflicts: same proposition with and without negation tokens.

#### Scenario: Polarity inversion detected
- **WHEN** propositions "remote sensing requires contact" and "remote sensing does not require contact" coexist
- **THEN** `detect_contradiction()` SHALL return a `PolarityInversion` with shared content and opposing polarity

### Requirement: Resolution policies
The system SHALL support resolution policies: `supersede_newest`, `supersede_highest_confidence`, `mark_both_review`, `keep_as_alternatives`.

#### Scenario: Supersede newest
- **WHEN** resolving with `policy="supersede_newest"`
- **THEN** the newer proposition SHALL be kept, older marked `superseded_by`

#### Scenario: Mark for review
- **WHEN** resolving with `policy="mark_both_review"`
- **THEN** both propositions SHALL get `needs_review=true` flag
- **THEN** neither SHALL be excluded from retrieval until reviewed

### Requirement: Contradiction audit log
The system SHALL log all detected contradictions with detection timestamp, involved fact IDs, resolution policy, and outcome.

#### Scenario: Audit trail preserved
- **WHEN** a contradiction is resolved
- **THEN** the audit log SHALL contain the full resolution record
- **THEN** the log SHALL be queryable for analysis