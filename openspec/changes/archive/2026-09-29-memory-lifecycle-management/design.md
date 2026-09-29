# Design

## Context

See `proposal.md`. Current `DroidEngine` has `teach()`, `recall()`, `chat()` with `KnowledgeStore` (SQLite + RRF) and `PlasticAssociativeRTU` (from `plastic-memory-lifelong` change). `DroidManager` handles profiles. This change adds lifecycle management on top.

## Goals / Non-Goals

**Goals:**
- Versioned snapshots with branching/rollback
- Multi-profile merge with conflict resolution
- Semantic store distilled from episodic
- Contradiction detection/resolution
- Feedback-driven adaptive retrieval
- Continuous background learning stream
- Test-time training during inference

**Non-Goals:**
- No change to ONNX anchor or RTU core math
- No change to Streamlit UI (CLI/MCP only)
- No new external dependencies

## Decisions

### Decision 1: Snapshot format extends existing safetensors + knowledge.db
- **Why**: `save_profile()` already writes `memory.safetensors` and exports `knowledge.db`. Snapshots add version metadata and bundle both.
- **Implementation**: `create_snapshot()` returns dict with `plastic_state` (from `PlasticAssociativeRTU.state_dict()`), `knowledge_db_path`, `metadata`. `restore_snapshot()` loads both.

### Decision 2: Semantic store as separate SQLite table in same DB
- **Why**: Avoids new file management; reuse `KnowledgeStore` connection. Add `semantic_facts` table with `id, text, source_cluster_ids, confidence, created_at, access_count`.
- **Implementation**: Extend `KnowledgeStore` with `semantic_` methods; distillation writes here.

### Decision 3: Merge operates at knowledge store level, plastic fusion at RTU level
- **Why**: Knowledge store has explicit facts with metadata; RTU has implicit associations. Merge both.
- **Implementation**: `merge_memories(A, B)` calls `KnowledgeStore.merge(other_store, policy)` and `PlasticAssociativeRTU.fuse(other_rtu, weight_A, weight_B)`.

### Decision 4: Contradiction resolution reuses existing SVO logic
- **Why**: `droid.py` already has `_is_contradiction()` with SVO slot clash + polarity inversion. Extend to return structured `Contradiction` objects.
- **Implementation**: Move `_is_contradiction` to `memory_lifecycle.py`, add `detect_all_contradictions()`, `resolve_contradictions(policy)`.

### Decision 5: Adaptive retrieval uses epsilon-greedy bandit per query type
- **Why**: Simple, proven, low overhead. 5 query types × 4 strategies = 20 arms.
- **Implementation**: `RetrievalPolicy` class with `select_strategy(query_type)`, `update(strategy, reward)`, `persist()`, `load()`.

### Decision 6: Continuous stream reuses `teach()` pipeline with higher threshold
- **Why**: Avoids duplicate code. Stream calls `_split_into_propositions()`, `_resolve_anaphora()`, then `teach()` with `surprise_threshold=0.6`.
- **Implementation**: `ContinuousStream` class with background thread, rate limiter, pause/resume.

### Decision 7: Test-time training reuses associative update with budget
- **Why**: `PlasticAssociativeRTU.update_associative_memory()` already exists. Add session budget counter.
- **Implementation**: `DroidEngine` tracks `ttt_steps_this_session`; `recall()` and feedback methods call `update_associative_memory()` if budget allows.

## Risks / Trade-offs

- **[Risk]** Merge conflict resolution may lose user intent if policy is wrong. → **Mitigation**: Default `require_manual`; log all conflicts for review.
- **[Risk]** Semantic distillation may oversimplify nuanced facts. → **Mitigation**: Keep episodic originals; semantic is additive layer.
- **[Risk]** Continuous stream may absorb noise. → **Mitigation**: Higher surprise threshold (0.6), rate limits, pause/resume.
- **[Risk]** Test-time training may drift weights. → **Mitigation**: Budget cap, LR decay, opt-out config.
- **[Risk]** Schema migrations for versioning/semantic/feedback columns. → **Mitigation**: `KnowledgeStore` migration on init with `ALTER TABLE IF NOT EXISTS`.

## Migration Plan

1. Create `src/model/memory_lifecycle.py` with `SnapshotManager`, `MergeEngine`, `SemanticStore`, `ContradictionResolver`, `RetrievalPolicy`, `ContinuousStream`
2. Extend `KnowledgeStore` schema: add `version_id`, `feedback`, `needs_review`, `semantic_facts` table
3. Add `create_snapshot()`, `rollback_to()`, `branch_from()`, `list_snapshots()` to `DroidEngine`
4. Add `merge_memory(other_droid, policy)`, `distill_to_semantic()` to `DroidEngine`
4. Add `approve()`, `reject()`, `correct()` to `DroidEngine`
5. Add `start_continuous_stream()`, `pause_stream()`, `get_stream_status()` to `DroidEngine`
6. Add test-time training budget logic to `recall()`, `approve()`, `reject()`, `correct()`
7. Update `DroidManager`: `merge_droids()`, `export_version()`, `import_version()`
8. Update `save_profile()`/`load_profile()` for version metadata
9. Tests: `tests/test_memory_lifecycle.py`, extend `test_droid_engine.py`
10. Run full test suite

## Open Questions

- Should semantic store have its own vector index or reuse episodic? → Start reusing; separate if >10k semantic facts.
- Merge plastic fusion: weighted average or concat + compress? → Start weighted average; monitor interference.
- Feedback reward signal: binary approve/reject or graded? → Start binary; extend to graded if needed.