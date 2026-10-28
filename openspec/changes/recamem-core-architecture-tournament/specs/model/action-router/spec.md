# Spec Delta

## Purpose

Defines the five-way agent action router (`generate`, `classify`, `retrieve`, `tool`, `abstain`), deterministic risk-coverage and schema guards, and the equal-budget Split-Specialist control needed to test whether one shared core can perform both generation and calibrated System-1 decisions without interference.

## ADDED Requirements

### Requirement: Five-way action routing candidates
The system SHALL evaluate and route incoming requests over the closed five-action space `{generate, classify, retrieve, tool, abstain}` through four interchangeable router candidates:
1. `R1` (`heuristic_cosine_gate`): deterministic keyword/pattern rules combined with anchor cosine similarity thresholds;
2. `R2` (`thresholded_confidence`): max-probability / energy thresholding fused with harmonic retrieval-generation confidence;
3. `R3` (`prompted_route_tokens`): autoregressive emission of dedicated `ROUTE_*` control tokens from the language-modeling head;
4. `R4` (`pooled_mlp_guarded`): learned MLP head reading the concatenation of attention-pooled query state and mean of the trailing 16 hidden states, wrapped in deterministic post-hoc guards.

#### Scenario: Every router candidate emits a uniform routing record
- **WHEN** any router candidate (`R1`–`R4`) evaluates a user request and optional context metadata
- **THEN** it SHALL return the selected action in `{generate, classify, retrieve, tool, abstain}`, per-action probabilities summing to `1.0`, a scalar confidence in `[0.0, 1.0]`, whether a deterministic guard overrode the raw argmax, and p50-measurable CPU latency in milliseconds

### Requirement: Deterministic risk-coverage and tool-permission guards
Regardless of the learned router's raw argmax, a deterministic software guard layer SHALL enforce risk-coverage abstention thresholds (fit strictly on a held-out calibration split) and pinned tool-schema / permission validation before any `tool` or `classify` decision is accepted for execution.

#### Scenario: Low-confidence action is overridden to abstain
- **WHEN** the router's calibrated action confidence or energy score falls below the threshold fit on the calibration split for the target risk ceiling
- **THEN** the guard layer SHALL override the final action to `abstain` and record `guard_triggered=true` with the triggering threshold
- **THEN** no downstream tool invocation or ungrounded answer generation SHALL occur

#### Scenario: Unauthorized or schema-invalid tool proposal is blocked
- **WHEN** the model proposes `tool` with a tool name absent from the pinned schema or arguments failing schema validation
- **THEN** the guard layer SHALL reject execution, record the validation failure reason, and fall back deterministically to `abstain`

### Requirement: Equal-budget Split-Specialist control architecture
To measure multi-task interference directly, the system SHALL provide a Split-Specialist control (`D4`) that partitions the exact same total parameter or resident-memory budget into an independent language-modeling core and an independent decision/routing encoder trained without weight sharing.

#### Scenario: Shared vs Split comparison respects total budget
- **WHEN** a Unified Shared-Core model and a Split-Specialist control (`D4`) are instantiated for Stage-3 interference evaluation
- **THEN** the sum of parameters (and total loaded resident memory) across the Split-Specialist's LM and decision/router modules SHALL NOT exceed the Unified Shared-Core model's budget by more than `2.0%`
- **THEN** both systems SHALL be evaluated on identical held-out LM, decision, calibration, and routing splits
