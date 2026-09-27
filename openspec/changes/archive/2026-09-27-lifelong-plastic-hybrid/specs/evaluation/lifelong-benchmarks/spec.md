# Spec Delta: Lifelong Learning Benchmarks

## Purpose

Provides reproducible continual learning evaluation protocols measuring retention, backward transfer, and catastrophic forgetting resistance.

## ADDED Requirements

### Requirement: Continual Retention Metric
The system SHALL evaluate model perplexity/loss on baseline domain $A$ after fine-tuning sequentially on novel domain $B$.

#### Scenario: Domain A retention measurement
- **WHEN** the model trains on novel domain text $B$
- **THEN** the retention ratio $\text{Loss}_A(\text{post}) / \text{Loss}_A(\text{pre})$ SHALL remain bounded below 1.25 to prevent catastrophic collapse
