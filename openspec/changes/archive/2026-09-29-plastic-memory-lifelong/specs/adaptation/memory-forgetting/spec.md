# Spec Delta

## Purpose

Ebbinghaus power-law forgetting with sleep consolidation: retrieval accessibility decays as a power function of elapsed time since last access; during idle cycles, salient facts are replayed into associative weights and low-utility episodic facts are pruned.

## ADDED Requirements

### Requirement: Power-law retrieval availability decay
The system SHALL compute fact accessibility as $A_i(t) = S_{\text{base}}(i) \cdot (1 + \alpha (t - t_i^{\text{last}}))^{-\beta_i}$ where $\beta_i$ increases with successful retrievals (spacing effect).

#### Scenario: Frequent retrieval increases stability
- **WHEN** a fact is successfully retrieved
- **THEN** its decay exponent $\beta_i$ SHALL decrease by factor $(1 - \delta_{\text{reinforce}})$ with $\delta_{\text{reinforce}} \in [0.1, 0.3]$
- **THEN** the fact SHALL decay more slowly thereafter

#### Scenario: One-off facts decay rapidly
- **WHEN** a fact is never retrieved after initial encoding
- **THEN** its accessibility SHALL follow the base power-law decay

### Requirement: Sleep consolidation during idle
The system SHALL run a background consolidation cycle when no queries have been received for $> 30$ seconds: sample high-utility facts, compute associative reconstruction loss, execute 3–5 gradient steps into plastic weights, and prune episodic facts with near-zero reconstruction error.

#### Scenario: Consolidation runs automatically on idle
- **WHEN** `consolidate_memory()` is called after idle period
- **THEN** it SHALL sample high-utility facts from the knowledge store
- **THEN** it SHALL execute gradient steps on `PlasticAssociativeRTU` parameters

#### Scenario: Absorbed facts pruned from episodic store
- **WHEN** a fact's reconstruction error falls below threshold
- **THEN** it SHALL be removed from the episodic knowledge store

### Requirement: Utility-based episodic pruning
The system SHALL compute fact utility as $\text{Utility}(i) = \text{AccessCount}_i \cdot \exp(-(t - t_i^{\text{last}}) / \tau_{\text{half}}) + \lambda \cdot s_i^{\text{novelty}}$ and evict bottom $P\%$ when capacity $K_{\max}$ is exceeded.

#### Scenario: Pruning triggers at capacity limit
- **WHEN** total episodic facts exceed $K_{\max} = 5000$
- **THEN** `consolidate_memory()` SHALL evict lowest-utility facts
- **THEN** remaining facts SHALL preserve audit trail metadata

#### Scenario: High-novelty facts resist pruning
- **WHEN** a fact has high novelty score $s_i$
- **THEN** its utility SHALL be boosted by $\lambda \cdot s_i$
- **THEN** it SHALL be retained preferentially