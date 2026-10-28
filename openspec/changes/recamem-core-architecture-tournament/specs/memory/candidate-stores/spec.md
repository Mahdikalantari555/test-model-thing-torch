# Spec Delta

## Purpose

Unifies six candidate memory and online-adaptation mechanisms behind one pluggable benchmark contract so hashed n-gram tables, surprise-gated residual episodic stores, hybrid SQLite RRF retrieval, sparse delta fast-weights, persistent core state, and online gradient plasticity can be isolated on identical continual-learning and belief-revision workloads.

## ADDED Requirements

### Requirement: Six isolated memory and adaptation arms
The system SHALL expose a uniform memory-augmentation and online-adaptation interface supporting six selectable candidate arms attached to any sequence core:
1. `M0` (`persistent_core_state`): carry recurrent fast-weight state ($S_t$ or $s_t$) across session turns with weights frozen and no external store;
2. `M1` (`engram_ngram_hash`): hashed n-gram embedding table (`H=8192 × d` by default) gated into intermediate residual layers (`2` and `L/2`) via learned scalar/vector sigmoid gates;
3. `M2` (`surprise_residual_episodic`): external key-value episodic store that admits `(key, value)` writes only when token negative log-likelihood exceeds a surprise threshold and injects top-$k=4$ softmax reads into the residual stream via a learned gate while keeping backbone weights untouched;
4. `M3` (`hybrid_knowledge_rrf`): the repository's SQLite FTS5 BM25 + dense vector reciprocal-rank-fusion store with non-monotonic contradiction superseding, evaluated in both `prompt_stuffing` and gated-residual injection modes;
5. `M4` (`sparse_delta_adapter`): multi-head gated delta-rule associative fast weights with optional `384 -> 2048` TopK-128 sparse coding and Householder targeted erase (`DeltaPlasticAdapter` / `SparseDeltaAdapter`);
6. `M5` (`online_weight_plasticity`): test-time parameter gradient updates (online SGD, EWC-regularized SGD, or diagonal RTRL traces) executed on a snapshot-guarded copy of the model.

#### Scenario: Memory arm attaches and detaches cleanly
- **WHEN** any candidate arm (`M0`–`M5`) is attached to the selected sequence core for a multi-domain teaching episode
- **THEN** it SHALL expose uniform `ingest(fact_or_span)`, `query(context)`, `erase_or_supersede(target)`, `reset()`, and `footprint_bytes()` operations
- **THEN** running with `M0`–`M4` SHALL leave all base sequence-core `nn.Parameter` weights bit-identical before and after ingestion

### Requirement: Bounded parameter and memory overhead accounting
Every memory arm SHALL report both static parameter overhead (such as `M1` hash table weights) and dynamic per-fact / per-token state growth in bytes so accuracy gains are never evaluated without their RAM cost.

#### Scenario: Engram and episodic store report exact byte footprints
- **WHEN** `M1` (`engram_ngram_hash`) or `M2`/`M3`/`M4` ingests 1,000, 10,000, and 20,000 facts
- **THEN** the harness SHALL record static table bytes, dynamic store bytes, peak RSS delta in megabytes, and p50/p95 read latency in milliseconds
- **THEN** any `M1` configuration whose hash table exceeds the configured edge parameter cap SHALL be rejected at construction
