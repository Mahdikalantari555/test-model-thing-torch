# dual-mode-generation Specification

## Purpose

Dual-mode generation: fast path (RTU-only, <50ms) for factual recall and routing; slow path (RTU+LLM, 200–2000ms) for synthesis, reasoning, and open-ended generation. Automatic routing based on query type and confidence.

## Requirements

### Requirement: Fast path — RTU factual recall
The system SHALL answer factual queries using only `recall()` + template synthesis without LLM generation.

#### Scenario: Fast path for fact lookup
- **WHEN** query classified as `specific_fact` and retrieval confidence > 0.7
- **THEN** `chat()` SHALL return synthesized facts in < 50ms
- **THEN** no LLM generation SHALL occur

### Requirement: Slow path — RTU+LLM synthesis
The system SHALL use memory injection + LLM generation for synthesis, reasoning, creative tasks.

#### Scenario: Slow path for synthesis
- **WHEN** query classified as `synthesis`, `reasoning`, or `creative`
- **THEN** `generate_with_memory()` SHALL be called with appropriate injection strategy
- **THEN** response SHALL include LLM-generated text grounded in retrieved facts

### Requirement: Automatic path routing
The system SHALL route queries using System-1 decision head (from `system1-mcp-surface`): `decide_action(query)` returns `fast_recall` or `slow_synthesis` with confidence.

#### Scenario: Routing decision
- **WHEN** `decide_action("what is NDVI?")` returns `fast_recall` with confidence 0.92
- **THEN** fast path SHALL be used
- **WHEN** `decide_action("explain how remote sensing helps agriculture")` returns `slow_synthesis` with confidence 0.85
- **THEN** slow path SHALL be used

### Requirement: Explicit mode override
The user SHALL be able to force fast or slow mode via CLI flag or chat command (`/fast`, `/slow`).

#### Scenario: User forces slow mode
- **WHEN** user sends `/slow explain remote sensing`
- **THEN** slow path SHALL be used regardless of classification