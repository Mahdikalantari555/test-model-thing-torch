# Spec Delta: Droid Profile Management

## Purpose

Enables creating, selecting, saving, and switching between named Droid instances, each maintaining independent persistent memory states, configs, and audit logs.

## ADDED Requirements

### Requirement: Named Droid Isolation
The system SHALL support creating and switching between multiple Droid instances stored under `droids/<droid-name>/`.

#### Scenario: Switching active Droid
- **WHEN** user selects a different Droid (e.g. `droid-remote-sensing`) from the profile selector
- **THEN** the active RTU memory state, logs, and metadata SHALL switch cleanly to the target profile without cross-contamination

#### Scenario: Memory round-trip persistence
- **WHEN** a Droid absorbs knowledge and is saved
- **THEN** re-loading that Droid SHALL restore its exact internal memory states and audit log history
