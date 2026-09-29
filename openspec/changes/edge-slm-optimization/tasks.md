# Tasks

## 1. HNSW and hybrid retrieval

- [ ] 1.1 Define a dense-index interface and implement `HNSWIndex` with add/update/delete/rebuild, persistence/version metadata, and exact-search fallback.
- [ ] 1.2 Integrate HNSW candidate IDs with FTS5 lexical candidates and existing RRF scoring without changing public recall result fields.
- [ ] 1.3 Ensure superseded/deleted facts are filtered and index rebuilds are atomic and repeatable.
- [ ] 1.4 Add 100k-fact benchmarks for p50/p95 latency, recall@5 against exact MVM, and a documented ~15 ms brute-force reference; target <5 ms and recall@5 >=0.95.
- [ ] 1.5 Preserve current exact mode and test missing optional dependency, corrupted index, and restart/rebuild behavior.

## 2. Reranking and Matryoshka retrieval

- [ ] 2.1 Implement optional `CrossEncoderReranker` over a bounded candidate set and expose pre/post scores.
- [ ] 2.2 Add a fixed labeled query set and measure Precision@5 before/after reranking; target +20% relative precision.
- [ ] 2.3 Implement `MatryoshkaAnchor` for validated 64-D and 384-D representations with versioned dual-index storage.
- [ ] 2.4 Add migration/re-embedding and rollback behavior; prohibit silent truncation of incompatible existing embeddings.
- [ ] 2.5 Measure 64-D first-stage latency and retrieval-quality loss against the 384-D baseline; target >=6x speed and <=5% relative quality loss.

## 3. Edge SLM catalog and recommendation

- [ ] 3.1 Implement `EdgeSLMRegistry` with the requested 13 models, quantization/format metadata, RAM estimates, task tags, provenance, and optional unknown speed fields.
- [ ] 3.2 Include the source-reported reference values for Llama 3.2 1B Q4_K_M, Qwen2.5 1.5B, SmolLM2 1.7B, and Gemma 3 1B with explicit provenance labels.
- [ ] 3.3 Implement deterministic `recommend(ram_gb, min_tok_s, task)` that filters hard RAM/speed constraints and returns task-ranked candidates with confidence/provenance.
- [ ] 3.4 Test low-RAM, throughput-constrained, reasoning, general-chat, unknown-device-speed, and no-match cases.
- [ ] 3.5 Integrate recommended IDs/config with `GgufBackend` without downloading or loading a model merely to display recommendations.

## 4. Test-time compute

- [ ] 4.1 Implement `TestTimeCompute` with diverse branching, candidate deduplication, verifier scoring, and strict depth/candidate/token/time budgets.
- [ ] 4.2 Add a trace/result contract with selected candidate, verifier evidence/score, and consumed budget.
- [ ] 4.3 Add fixed reasoning-task evaluation against an explicitly identified 8B reference model at the same wall-clock or token budget.
- [ ] 4.4 Verify a 1B+search/verifier configuration has statistically higher task score than the chosen 8B baseline on the declared benchmark before claiming the target.
- [ ] 4.5 Ensure unselected/unverified candidates are not persisted to knowledge without user approval.

## 5. Validation and compatibility

- [ ] 5.1 Run existing knowledge-store, DroidEngine, GGUF, profile, and checkpoint tests.
- [ ] 5.2 Verify lexical exact-match behavior and the existing <5 ms at 100k knowledge-store SLA on documented hardware.
- [ ] 5.3 Run model catalog/recommendation and TTC benchmarks with provenance and configuration recorded.
- [ ] 5.4 Document optional dependencies, exact fallback, profile flags, and index/embedding version migration.
