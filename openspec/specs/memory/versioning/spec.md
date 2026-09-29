# versioning Specification

## Purpose

Memory versioning with full state snapshots, branching, and point-in-time rollback for the Droid's plastic state and knowledge store.

## Requirements

### Requirement: Snapshot creation with version metadata
The system SHALL create a complete memory snapshot containing the plastic RTU state (associative matrices, trace, decay), knowledge store checkpoint, and metadata (version ID, timestamp, step count, fact count, parent version).

#### Scenario: Snapshot captures full state
- **WHEN** `create_snapshot("v1.0")` is called
- **THEN** the snapshot SHALL include plastic state, knowledge DB, and metadata
- **THEN** the snapshot SHALL be serializable to disk

#### Scenario: Version metadata recorded
- **WHEN** a snapshot is created
- **THEN** metadata SHALL include `version_id`, `timestamp`, `step_count`, `fact_count`, `parent_version`

### Requirement: Snapshot listing and inspection
The system SHALL list all available snapshots with their metadata and allow inspecting snapshot contents without restoring.

#### Scenario: List snapshots
- **WHEN** `list_snapshots()` is called
- **THEN** it SHALL return a list of version IDs with metadata sorted by timestamp

### Requirement: Point-in-time rollback
The system SHALL restore the Droid to a prior snapshot version, replacing plastic state and knowledge store atomically.

#### Scenario: Rollback restores exact prior state
- **WHEN** `rollback_to("v1.0")` is called
- **THEN** plastic state SHALL match snapshot exactly
- **THEN** knowledge store SHALL match snapshot exactly
- **THEN** current version SHALL become child of rolled-back version (branch)

### Requirement: Branching from snapshot
The system SHALL support creating a new branch from any snapshot, allowing divergent memory evolution.

#### Scenario: Branch creates independent lineage
- **WHEN** `branch_from("v1.0", "experiment-a")` is called
- **THEN** new version `"experiment-a"` SHALL start from `"v1.0"` state
- **THEN** subsequent changes SHALL not affect `"v1.0"` lineage