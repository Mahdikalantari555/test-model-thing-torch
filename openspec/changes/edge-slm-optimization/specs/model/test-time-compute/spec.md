# Spec Delta

## Purpose

Specify bounded diverse verifier-guided tree search for allocating test-time inference compute to compact edge models.

## ADDED Requirements

### Requirement: Diverse verifier tree search
The system SHALL provide `TestTimeCompute` that generates diverse candidate continuations, expands a bounded search tree, deduplicates candidates, and selects a result using an explicit verifier interface. Search SHALL expose configurable candidate, depth, token, and wall-clock limits.

#### Scenario: Search under a fixed budget
- **WHEN** test-time compute is enabled with a finite budget
- **THEN** candidate generation and tree expansion SHALL stop when any configured budget is exhausted
- **THEN** the selected answer SHALL include a verifier score and a trace of candidates/budget use
- **THEN** the system SHALL return the best available candidate or a clear no-candidate result without exceeding the budget

#### Scenario: Preserve diversity
- **WHEN** multiple branches are sampled
- **THEN** the search SHALL apply a configured diversity/deduplication policy
- **THEN** duplicate candidates SHALL NOT consume unbounded verifier work

### Requirement: Small-model plus verifier evaluation target
The project SHALL evaluate a 1B-class model with verifier-guided tree search against an explicitly identified 8B-class baseline on a versioned task benchmark and a declared equal wall-clock or token budget. The target is statistically higher task score for the 1B-plus-search configuration on the declared benchmark; this SHALL NOT be generalized to unmeasured tasks.

#### Scenario: Compare 1B plus TTC with 8B
- **WHEN** the paired benchmark is run using fixed prompts, decoding settings, verifier version, and resource budget
- **THEN** it SHALL report the 1B baseline, 1B-plus-TTC, and selected 8B baseline scores
- **THEN** the target SHALL be considered met only when 1B-plus-TTC exceeds the 8B score with the predeclared statistical test
- **THEN** the report SHALL identify the benchmark, models, quantization, verifier, latency, and token budget

### Requirement: No implicit memory writes
Test-time candidate generation SHALL NOT persist unselected or unverified candidate text into Droid knowledge without an explicit user approval or a separate configured acceptance policy.

#### Scenario: Candidate is rejected by verifier
- **WHEN** a generated branch fails the configured verifier threshold
- **THEN** it SHALL be excluded from final selection or marked as rejected
- **THEN** it SHALL NOT be inserted into the persistent knowledge store by default
