# Tasks

## 1. PlasticAssociativeRTU Core

- [ ] 1.1 Create `PlasticAssociativeRTU` class in `src/model/rtu.py` with multi-head associative state (4 heads × 96 × 96), projections (W_pred, W_K, W_V, W_Q, W_gate, W_alpha), and buffers (S, momentum, h_trace)
- [ ] 1.2 Implement `predict_next()` returning top-down expectation from h_trace
- [ ] 1.3 Implement `compute_surprise(x)` returning (error_vector, surprise_scalar) with cosine surprise clamped to [0, 2]
- [ ] 1.4 Implement `update_associative_memory(x, surprise_factor)` with momentum-augmented update and Householder-like erasure operator
- [ ] 1.5 Implement `retrieve(q_vec)` for associative recall from fast weights
- [ ] 1.6 Keep `RTUMemoryBlock` for backward compat; add deprecation comment
- [ ] 1.7 Verify: unit tests in `tests/test_rtu.py` pass for all new methods

## 2. KnowledgeStore Schema Extension

- [ ] 2.1 Add `access_count INTEGER DEFAULT 1` and `last_retrieved REAL` columns to `facts` table in `KnowledgeStore._create_schema()`
- [ ] 2.2 Implement migration in `KnowledgeStore.__init__` to add columns if missing (ALTER TABLE)
- [ ] 2.3 Update `recall()` in `KnowledgeStore` to increment `access_count` and update `last_retrieved` on hit
- [ ] 2.4 Verify: `tests/test_knowledge_store.py` covers new columns and migration

## 3. DroidEngine Integration — Teach with Surprise Gate

- [ ] 3.1 Update `DroidEngine.__init__` to instantiate `PlasticAssociativeRTU` as `self.memory`
- [ ] 3.2 In `teach()`: for each proposition embedding, call `memory.compute_surprise(e_i)` before insertion
- [ ] 3.3 In `teach()`: only insert into `KnowledgeStore` if `surprise >= 0.4` (configurable threshold)
- [ ] 3.4 In `teach()`: always call `memory.update_associative_memory(e_i, surprise_factor=surprise)`
- [ ] 3.5 Update `teach()` return dict to include `episodic_absorbed`, `avg_surprise`, `surprise_threshold`
- [ ] 3.6 Verify: `tests/test_droid_engine.py` test shows surprise gate reduces episodic inserts

## 4. DroidEngine Integration — Anti-Pollution Recall

- [ ] 4.1 In `recall()`: compute semantic sims via `KnowledgeStore.recall()` (existing RRF)
- [ ] 4.2 In `recall()`: compute contextual alignment `cos(memory.h_trace, fact_emb)` for candidates
- [ ] 4.3 In `recall()`: fuse as `final = semantic * (1 + 0.2 * max(0, contextual))` only if `semantic >= 0.48`
- [ ] 4.4 In `recall()`: update `access_count` and `last_retrieved` in knowledge store for returned facts
- [ ] 4.5 Verify: `tests/test_droid_engine.py` shows unrelated queries not boosted by h_trace

## 5. Consolidation & Forgetting

- [ ] 5.1 Implement `DroidEngine.consolidate_memory(max_prune_ratio=0.1)` with utility scoring and pruning
- [ ] 5.2 Add `access_count`/`last_retrieved` tracking in knowledge store recall (from 2.3)
- [ ] 5.3 Implement associative reconstruction loss gradient steps on `PlasticAssociativeRTU` params (3–5 steps, lr=1e-4)
- [ ] 5.4 Add orthogonality regularization on associative matrices during consolidation
- [ ] 5.5 Verify: `tests/test_droid_engine.py` test shows pruning reduces fact count, consolidation improves reconstruction

## 6. Snapshots & Versioning

- [ ] 6.1 Implement `PlasticAssociativeRTU.state_dict()` / `load_state_dict()` covering S, momentum, h_trace, decay
- [ ] 6.2 Implement `DroidEngine.create_snapshot()` returning dict with plastic state, knowledge store checkpoint, version metadata
- [ ] 6.3 Implement `DroidEngine.restore_snapshot(snapshot_dict)` restoring plastic state and knowledge store
- [ ] 6.4 Update `save_profile()` to include snapshot version in config.json
- [ ] 6.5 Update `load_profile()` to migrate from legacy memory.safetensors to new format
- [ ] 6.6 Verify: `tests/test_droid_engine.py` round-trip snapshot save/load

## 7. Memory Health Metrics

- [ ] 7.1 Implement `DroidEngine.get_memory_health()` returning dict: saturation, forgetting_rate, interference, retrieval_quality
- [ ] 7.2 Saturation: active_facts / K_max + associative_norm / reference_scale
- [ ] 7.3 Forgetting rate: fraction of facts with accessibility below threshold over rolling window
- [ ] 7.4 Interference: mean pairwise cosine similarity of fact embeddings (sample if >1000)
- [ ] 7.5 Retrieval quality: precision@k and MRR from recent queries with feedback
- [ ] 7.6 Verify: `tests/test_droid_engine.py` health metrics return expected ranges

## 8. Tests & Validation

- [ ] 8.1 Create `tests/test_plastic_associative_rtu.py` with unit tests for all `PlasticAssociativeRTU` methods
- [ ] 8.2 Extend `tests/test_droid_engine.py` with integration tests for surprise gate, consolidation, snapshots, health
- [ ] 8.3 Run full test suite: `PYTHONPATH=. /home/asus/miniforge3/envs/ai/bin/pytest tests -v`
- [ ] 8.4 Benchmark: verify RAM < 30 MB, recall latency < 5 ms at 50k facts, surprise gate reduces inserts by 40%+