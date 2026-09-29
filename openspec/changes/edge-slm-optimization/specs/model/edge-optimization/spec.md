# Spec Delta

## Purpose

Specify a provenance-aware catalog and resource-constrained recommendation API for on-device small language models.

## ADDED Requirements

### Requirement: On-device SLM catalog
The system SHALL provide a versioned `EdgeSLMRegistry` catalog containing at least these 13 model entries: Llama 3.2 1B and 3B; Qwen2.5 0.5B, 1.5B, and 3B; SmolLM2 135M, 360M, 1.7B, and 3B; Gemma 3 270M, 1B, and 4B; and Phi-4-mini. Entries SHALL record available model/quantization identifiers, parameter size, estimated RAM, task tags, throughput when known, source/provenance, and verification date.

#### Scenario: Inspect requested model families
- **WHEN** the registry is loaded
- **THEN** it SHALL expose all 13 requested model variants
- **THEN** missing device-specific throughput SHALL be represented as unknown rather than fabricated
- **THEN** source-reported estimates SHALL be distinguishable from local measurements

#### Scenario: Display sourced reference metadata
- **WHEN** a caller inspects catalog metadata
- **THEN** Llama 3.2 1B Q4_K_M SHALL include the source-reported estimate of about 1.5 GB RAM and 30–50 tokens/s
- **THEN** Qwen2.5 1.5B SHALL be tagged as a reasoning-oriented option from the cited survey
- **THEN** SmolLM2 1.7B SHALL include the reported 11T-token training-corpus note
- **THEN** Gemma 3 1B SHALL include the reported ~0.8 GB and 62.8% GSM8K reference with provenance
- **THEN** these fields SHALL be labeled source-reported, not guaranteed local performance

### Requirement: Resource- and task-aware recommendations
The registry SHALL implement `recommend(ram_gb, min_tok_s, task)` that filters models against RAM and requested throughput constraints, ranks remaining models using task tags/available benchmark metadata, and returns the rationale and provenance for each recommendation.

#### Scenario: Respect hard device budgets
- **WHEN** a caller specifies RAM and minimum tokens-per-second constraints
- **THEN** recommendations SHALL exclude candidates whose known RAM exceeds the budget
- **THEN** candidates with unknown throughput SHALL NOT be represented as meeting a hard minimum unless a local measurement is supplied
- **THEN** a no-match result SHALL be returned when no entry satisfies the constraints

#### Scenario: Recommend by task
- **WHEN** the requested task is reasoning, summarization, or general chat
- **THEN** eligible models SHALL be ranked using the task tags and available benchmark evidence
- **THEN** the response SHALL expose why each model was recommended and distinguish source claims from local test results

### Requirement: GGUF backend compatibility
Registry recommendations SHALL map to compatible model/quantization identifiers without loading or downloading a model as a side effect of listing or recommending it.

#### Scenario: Pass recommendation to inference
- **WHEN** a caller selects a recommended GGUF-compatible entry
- **THEN** its model identifier and quantization metadata SHALL be usable to configure `GgufBackend`
- **THEN** actual loading SHALL occur only after an explicit load/inference request
