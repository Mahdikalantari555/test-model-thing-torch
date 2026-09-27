# Spec Delta: Parallel Scan Training

## Purpose

Enables high-throughput batched training by executing linear recurrence over full sequence chunks using associative scan instead of single-step loops.

## ADDED Requirements

### Requirement: Chunked Recurrent Forward
The system SHALL compute multi-step linear hidden state trajectories in parallel chunks across sequence length $L$.

#### Scenario: Equivalence with sequential step
- **WHEN** a sequence of length $L$ is processed via chunked parallel scan
- **THEN** the resulting final hidden states and output logits SHALL match the sequential single-step execution within relative tolerance 1e-4

#### Scenario: Multi-sequence batching
- **WHEN** a batch of shape $(B, L)$ is provided to the training forward pass
- **THEN** the model SHALL execute recurrence across all batch elements concurrently without Python per-token loops
