# memory-snapshots Specification

## Purpose

State snapshot system for memory checkpointing, versioning, and rollback: serialize the complete plastic state (associative matrices, momentum, trace, decay) alongside knowledge store version for point-in-time recovery.

## Requirements

### Requirement: Full state snapshot serialization
The system SHALL serialize the complete `PlasticAssociativeRTU` state (associative matrices $S$, momentum buffers, trace $h$, decay parameters) plus knowledge store version identifier into a single portable artifact.

#### Scenario: Snapshot captures all plastic state
- **WHEN** `create_snapshot()` is called
- **THEN** the artifact SHALL contain $S$, momentum, $h$, decay, and knowledge store checkpoint ID
- **THEN** the artifact SHALL be serializable to disk via safetensors

### Requirement: Snapshot versioning with metadata
The system SHALL assign monotonically increasing version IDs to snapshots with timestamp, step count, fact count, and memory health metrics at capture time.

#### Scenario: Version metadata recorded
- **WHEN** a snapshot is created
- **THEN** it SHALL carry `version`, `timestamp`, `step_count`, `fact_count`, `health_metrics`

### Requirement: Rollback to snapshot
The system SHALL restore `PlasticAssociativeRTU` state and knowledge store to a prior snapshot version, discarding subsequent changes.

#### Scenario: Rollback restores exact state
- **WHEN** `restore_snapshot(version)` is called
- **THEN** $S$, momentum, $h$, decay SHALL match the snapshot exactly
- **THEN** knowledge store SHALL be restored to the checkpointed version