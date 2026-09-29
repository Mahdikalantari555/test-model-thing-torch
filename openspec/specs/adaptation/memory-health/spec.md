# memory-health Specification

## Purpose

Memory health metrics quantifying saturation, forgetting rate, interference, and retrieval quality for observability and adaptive policy control.

## Requirements

### Requirement: Saturation metric
The system SHALL report memory saturation as the ratio of active episodic facts to configured capacity $K_{\max}$, plus the associative state Frobenius norm relative to a reference scale.

#### Scenario: Saturation near capacity
- **WHEN** episodic facts reach $90\%$ of $K_{\max}$
- **THEN** `get_memory_health()["saturation"]` SHALL return $\ge 0.9$

#### Scenario: Associative state norm indicates load
- **WHEN** `PlasticAssociativeRTU.S` accumulates many associations
- **THEN** `get_memory_health()["associative_norm"]` SHALL reflect Frobenius norm

### Requirement: Forgetting rate metric
The system SHALL report the effective forgetting rate as the fraction of facts whose accessibility $A_i(t)$ has fallen below a utility threshold over a rolling window.

#### Scenario: Forgetting rate reflects decay
- **WHEN** many facts have not been accessed for long periods
- **THEN** `get_memory_health()["forgetting_rate"]` SHALL be elevated

### Requirement: Interference metric
The system SHALL estimate interference as the mean cosine similarity between distinct fact embeddings in the knowledge store, indicating retrieval ambiguity.

#### Scenario: High interference warns of confusion
- **WHEN** fact embeddings cluster tightly
- **THEN** `get_memory_health()["interference"]` SHALL be $\ge 0.7$

### Requirement: Retrieval quality metric
The system SHALL track retrieval precision@k and mean reciprocal rank (MRR) over recent queries, with ground truth from user feedback (approve/reject).

#### Scenario: Quality reflects user satisfaction
- **WHEN** user approves retrieved facts
- **THEN** `get_memory_health()["retrieval_quality"]` SHALL increase
- **WHEN** user rejects retrieved facts
- **THEN** `get_memory_health()["retrieval_quality"]` SHALL decrease