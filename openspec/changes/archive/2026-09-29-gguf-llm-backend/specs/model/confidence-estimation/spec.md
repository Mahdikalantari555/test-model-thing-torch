# Spec Delta

## Purpose

Combined confidence estimation: fuse RTU retrieval confidence (semantic similarity, surprise, health metrics) with LLM generation confidence (token probability, self-consistency, entropy) for calibrated output reliability.

## ADDED Requirements

### Requirement: Retrieval confidence component
The system SHALL compute retrieval confidence from: max semantic similarity, surprise alignment, memory health (low interference, high quality).

#### Scenario: High retrieval confidence
- **WHEN** top fact has similarity 0.85, low interference, high health
- **THEN** `retrieval_confidence` SHALL be > 0.8

### Requirement: Generation confidence component
The system SHALL compute generation confidence from: mean token log-prob, self-consistency (multiple samples agreement), normalized entropy.

#### Scenario: High generation confidence
- **WHEN** tokens have high log-prob, 3/3 samples agree, low entropy
- **THEN** `generation_confidence` SHALL be > 0.8

### Requirement: Fused confidence score
The system SHALL combine retrieval and generation confidence as weighted harmonic mean: $C = 2 / (1/C_r + 1/C_g)$ with weights configurable per query type.

#### Scenario: Fused confidence reflects both
- **WHEN** $C_r=0.9, C_g=0.7$
- **THEN** fused $C \approx 0.78$
- **WHEN** $C_r=0.3, C_g=0.9$
- **THEN** fused $C \approx 0.45$ (retrieval limits)

### Requirement: Confidence thresholds for action
The system SHALL define thresholds: `high` (>0.8) = auto-execute, `medium` (0.5–0.8) = show with citation, `low` (<0.5) = defer to user.

#### Scenario: Low confidence triggers clarification
- **WHEN** fused confidence < 0.5
- **THEN** response SHALL include "I'm uncertain; here's what I found..." with citations