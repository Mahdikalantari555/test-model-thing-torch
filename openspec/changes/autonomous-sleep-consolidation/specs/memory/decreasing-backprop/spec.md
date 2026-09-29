# Spec Delta

## Purpose

Specify a persisted Decreasing Backprop learning-rate schedule for stability/plasticity control.

## ADDED Requirements

### Requirement: Exponential decreasing learning rate
The system SHALL provide `DecreasingBackprop` with `lr_i = base_lr * decay_factor^i`, where `i` is a zero-based persisted update/task index and `0 < decay_factor <= 1`.

#### Scenario: Compute schedule values
- **WHEN** `base_lr` and `decay_factor` are valid and index `i` is requested
- **THEN** the returned learning rate SHALL equal `base_lr * decay_factor**i` within floating-point tolerance
- **THEN** index zero SHALL return `base_lr`
- **THEN** rates SHALL be non-increasing as `i` increases

#### Scenario: Reject invalid schedule settings
- **WHEN** `base_lr` is non-positive, `decay_factor` is outside `(0, 1]`, or the index is negative
- **THEN** the schedule SHALL reject the configuration with a clear validation error
- **THEN** it SHALL NOT silently clamp an invalid parameter

### Requirement: Schedule persistence and optimizer integration
The schedule SHALL update the optimizer rate at the documented task/update boundary and persist its index and configuration with the Droid profile/checkpoint.

#### Scenario: Resume after restart
- **WHEN** a Droid with schedule index `i` is saved and reloaded
- **THEN** its next learning rate SHALL be computed from the restored index and configuration
- **THEN** restart SHALL NOT reset the schedule to index zero unless explicitly requested

#### Scenario: Advance after committed task
- **WHEN** one task/update boundary is committed
- **THEN** the schedule index SHALL advance exactly once
- **THEN** retrying a failed or uncommitted update SHALL NOT advance the index

### Requirement: Stability reporting
The system SHALL expose the current schedule index and effective learning rate in training/sleep reports so stability experiments can be reproduced.

#### Scenario: Inspect effective rate
- **WHEN** a caller requests schedule diagnostics
- **THEN** diagnostics SHALL include `base_lr`, `decay_factor`, `i`, and `lr_i`
