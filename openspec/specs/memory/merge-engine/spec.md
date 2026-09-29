# merge-engine Specification

## Purpose

Multi-profile memory fusion: combine two or more Droid memories into a unified memory with conflict detection and resolution policies (keep newest, keep highest novelty, manual review).

## Requirements

### Requirement: Memory merge with conflict detection
The system SHALL merge two Droid memories by unioning their knowledge stores and fusing plastic states, detecting conflicts where both memories contain propositions about the same subject-predicate with different objects.

#### Scenario: Merge detects slot clashes
- **WHEN** Droid A has "capital of France is Paris" and Droid B has "capital of France is Lyon"
- **THEN** `merge_memories(A, B)` SHALL detect the `capital_of` slot clash
- **THEN** the conflict SHALL be reported with both propositions

### Requirement: Configurable conflict resolution policies
The system SHALL resolve merge conflicts using a configurable policy: `keep_newest` (by timestamp), `keep_highest_novelty`, `keep_both_as_alternatives`, or `require_manual`.

#### Scenario: keep_newest selects later timestamp
- **WHEN** merging with `policy="keep_newest"`
- **THEN** the proposition with later timestamp SHALL be retained
- **THEN** the other SHALL be marked superseded

#### Scenario: keep_highest_novelty selects more surprising fact
- **WHEN** merging with `policy="keep_highest_novelty"`
- **THEN** the proposition with higher novelty score SHALL be retained

#### Scenario: keep_both stores alternatives
- **WHEN** merging with `policy="keep_both_as_alternatives"`
- **THEN** both propositions SHALL be stored with `alternative_of` linking

### Requirement: Plastic state fusion
The system SHALL fuse plastic associative states by weighted averaging of $S$ matrices and trace vectors, weighted by each Droid's fact count and average novelty.

#### Scenario: Fusion preserves both memories' associations
- **WHEN** `merge_memories(A, B)` completes
- **THEN** the fused $S$ matrix SHALL encode associations from both A and B
- **THEN** the fused trace $h$ SHALL be weighted average

### Requirement: Merge audit trail
The system SHALL record merge operations with source versions, conflict count, resolution decisions, and resulting version ID.

#### Scenario: Merge creates auditable version
- **WHEN** merge completes
- **THEN** a new version SHALL be created with `merge_of: [version_A, version_B]`
- **THEN** conflict resolutions SHALL be logged