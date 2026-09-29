# semantic-store Specification

## Purpose

Semantic memory store: a separate, distilled proposition layer extracted from episodic experiences. Semantic facts are de-duplicated, stable, and lack temporal metadata — they represent consolidated knowledge rather than episodic traces.

## Requirements

### Requirement: Distillation from episodic to semantic
The system SHALL distill episodic facts into semantic propositions by clustering similar facts, resolving contradictions, and extracting the consensus proposition.

#### Scenario: Distillation clusters similar facts
- **WHEN** `distill_to_semantic()` is called
- **THEN** episodic facts with cosine similarity $\ge 0.85$ SHALL be clustered
- **THEN** each cluster SHALL produce one semantic proposition

#### Scenario: Contradictions resolved during distillation
- **WHEN** a cluster contains contradictory propositions
- **THEN** the contradiction resolution policy (from `memory/contradiction-resolution`) SHALL select the consensus
- **THEN** the semantic proposition SHALL reflect the resolved fact

### Requirement: Semantic store separate from episodic
The system SHALL maintain semantic propositions in a dedicated store with schema: `id, text, source_cluster_ids, confidence, created_at, access_count`.

#### Scenario: Semantic propositions lack temporal decay
- **WHEN** a semantic proposition is created
- **THEN** it SHALL NOT have `valid_until` or `superseded` fields
- **THEN** it SHALL have a `confidence` score from distillation consensus

### Requirement: Semantic retrieval with higher precision
The system SHALL retrieve from semantic store for definitional/generic queries and from episodic store for specific/temporal queries, fusing results.

#### Scenario: Generic query hits semantic store
- **WHEN** query is "what is remote sensing"
- **THEN** semantic store SHALL be primary source
- **THEN** results SHALL have higher precision for definitional queries

### Requirement: Continuous distillation
The system SHALL support incremental distillation: new episodic facts trigger re-distillation of affected clusters only.

#### Scenario: Incremental distillation on new facts
- **WHEN** new episodic facts are added
- **THEN** only clusters overlapping with new facts SHALL be re-distilled
- **THEN** semantic store SHALL update without full recomputation