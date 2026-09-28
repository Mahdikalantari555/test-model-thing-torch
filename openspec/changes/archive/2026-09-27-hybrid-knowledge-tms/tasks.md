# Tasks

## Phase 1: Knowledge Store Core
- [x] T1.1 Create `src/model/knowledge_store.py` with `KnowledgeStore` class wrapping SQLite WAL + FTS5 + dense vectors
- [x] T1.2 Implement `insert_fact(text, source, timestamp, step, novelty, vec)` with pre-normalization
- [x] T1.3 Implement `recall(query_vec, top_k, threshold)` using fused RRF ranking
- [x] T1.4 Implement `supersede(old_id, new_id)` marking outdated facts
- [x] T1.5 Implement `get_audit_trail()` returning all facts including superseded

## Phase 2: Integration into DroidEngine
- [x] T2.1 Update `DroidEngine.__init__` to use `KnowledgeStore` instead of `knowledge_bank` list
- [x] T2.2 Update `teach()` to run rolling anaphora resolution before proposition extraction
- [x] T2.3 Update `teach()` to run two-tier contradiction gating on each new proposition
- [x] T2.4 Update `teach()` to insert facts and vectors into `KnowledgeStore`
- [x] T2.5 Update `recall()` to delegate to `KnowledgeStore.recall()`
- [x] T2.6 Update `chat()` to use new recall interface and handle superseded facts

## Phase 3: Persistence
- [x] T3.1 Update `save_profile()` to persist SQLite DB alongside `memory.safetensors`
- [x] T3.2 Update `load_profile()` to restore SQLite DB from disk
- [x] T3.3 Implement `PRAGMA wal_checkpoint(TRUNCATE)` on save

## Phase 4: Tests
- [x] T4.1 Create `tests/test_knowledge_store.py` with unit tests for insert, recall, supersede, and RRF fusion
- [x] T4.2 Add integration tests in `tests/test_droid_engine.py` for anaphora resolution and contradiction detection
- [x] T4.3 Run full test suite: `PYTHONPATH=. /home/asus/miniforge3/envs/ai/bin/pytest tests`

## Phase 5: Validation
- [x] T5.1 Benchmark retrieval latency at 10k, 50k, and 100k facts
- [x] T5.2 Verify 0 KB query-time allocation at 50k facts
- [x] T5.3 Verify contradiction detection and superseding behavior
- [x] T5.4 Run `scripts/verify_remote_sensing_e2e.py` end-to-end