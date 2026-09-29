# Spec Delta

## Purpose

Specify autonomous sleep-like memory consolidation, bounded autonomous retrieval, an explicit sleep API/command, and an opt-in background worker.

## ADDED Requirements

### Requirement: Autonomous retrieval without an external pattern list
The system SHALL provide `SleepConsolidation.autonomous_retrieval()` that retrieves replay candidates from internal Droid memory using self-inhibition/refractory state. The method SHALL NOT require or accept the expected pattern list used by evaluation.

#### Scenario: Recover correlated patterns
- **WHEN** the evaluation harness stores 100 correlated patterns and runs autonomous retrieval without passing a pattern list
- **THEN** the retrieval sequence SHALL recover at least 95 distinct expected patterns within the configured bounded cycle
- **THEN** the report SHALL include unique recovered count, duplicate count, steps, and coverage

#### Scenario: Self-inhibition reveals another pattern
- **WHEN** one attractor is retrieved repeatedly during a sleep cycle
- **THEN** its self-inhibition SHALL increase within configured bounds
- **THEN** the next retrieval attempts SHALL be able to surface other eligible patterns rather than repeatedly returning the same attractor

### Requirement: Sleep API and report
`DroidEngine.sleep()` SHALL execute a bounded consolidation cycle and return a structured report distinguishing replay, fact pruning, parameter pruning, duration, status, and warnings.

#### Scenario: Manual sleep
- **WHEN** a caller invokes `DroidEngine.sleep()` on a non-empty Droid
- **THEN** exactly one configured cycle SHALL run
- **THEN** the method SHALL return a report even when a phase has no eligible work
- **THEN** the report SHALL not claim recovered patterns or pruning that did not occur

#### Scenario: Empty memory
- **WHEN** sleep is requested on an empty Droid
- **THEN** the method SHALL return a successful no-op report with zero replay and pruning counts
- **THEN** it SHALL not raise an unhandled exception

### Requirement: Opt-in autonomous sleep worker
The system SHALL support a profile-configured, cancellable auto-sleep worker with at most one active worker per Droid. Automatic sleep SHALL be disabled by default and SHALL respect configured idle/time limits.

#### Scenario: Start and stop auto-sleep
- **WHEN** auto-sleep is enabled for a Droid
- **THEN** one worker SHALL run according to the configured trigger and cycle budget
- **WHEN** the Droid is stopped or its profile is closed
- **THEN** the worker SHALL receive cancellation and join without leaving a duplicate thread

#### Scenario: Repeated UI initialization
- **WHEN** an application rerun requests auto-sleep startup multiple times for the same Droid
- **THEN** the manager SHALL reuse the existing worker rather than start another one

### Requirement: `/sleep` command
The chat/CLI surface SHALL route `/sleep` to the same `DroidEngine.sleep()` API before ordinary retrieval or teaching command handling.

#### Scenario: Invoke sleep from chat
- **WHEN** the user sends `/sleep`
- **THEN** one sleep cycle SHALL execute
- **THEN** the response SHALL summarize the report and the command text SHALL NOT be inserted as a knowledge fact
