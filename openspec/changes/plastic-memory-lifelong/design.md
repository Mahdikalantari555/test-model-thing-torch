# Design

## Context

See `proposal.md` for motivation. Current `RTUMemoryBlock` in `src/model/rtu.py` implements a 1D vector EMA. The knowledge store (`KnowledgeStore` in `src/model/knowledge_store.py`) handles episodic facts with RRF retrieval. `DroidEngine.teach()` and `recall()` in `src/model/droid.py` are the integration points.

## Goals / Non-Goals

**Goals:**
- Replace 1D EMA with multi-head associative memory (Titans + RWKV-7) without increasing RAM > 1.5 MB
- Add surprise-gated episodic admission reducing redundant fact storage by 40–60%
- Enable retrieval-time contextual modulation with zero centroid pollution
- Add bounded forgetting via Ebbinghaus power-law + sleep consolidation
- Provide snapshot/versioning for memory checkpoints and rollback
- Expose health metrics for adaptive policies

**Non-Goals:**
- No change to ONNX MiniLM anchor or knowledge store schema
- No change to Streamlit UI or DroidManager profile management
- No new external C++ dependencies; stdlib + torch only

## Decisions

### Decision 1: Titans momentum surprise + RWKV-7 erasure hybrid
- **Why**: Titans provides surprise-gated inner-loop optimization; RWKV-7 provides selective dimensional erasure via Householder-like operator. Together they solve eigen-saturation and enable overwrite.
- **Alternatives**: Pure Titans (no selective erasure); pure RWKV-7 (no surprise gating); fast weight programmers (ICML 2021) — less stable at edge scale.
- **Implementation**: `PlasticAssociativeRTU` in `src/model/rtu.py` with:
  - `heads=4`, `head_dim=96` for `dim=384`
  - $S \in \mathbb{R}^{4 \times 96 \times 96}$ associative matrices
  - $W_{\text{pred}}, W_K, W_V, W_Q$ projections
  - Dynamic $w_{\text{decay}}, \alpha$ per head via learned gates

### Decision 2: Surprise as admission gate, not just weight
- **Why**: Current `teach()` stores every proposition. Surprise gates episodic admission: only surprising facts ($s_t \ge 0.4$) enter the SQLite store; all facts update associative weights. Reduces store growth 40–60%.
- **Implementation**: In `DroidEngine.teach()`, call `memory.compute_surprise(e_i)`; if $s_t \ge \theta_{\text{surprise}}$, insert into `KnowledgeStore`; always call `memory.update_associative_memory(e_i, surprise_factor=s_t)`.

### Decision 3: Anti-pollution retrieval via multiplicative gating
- **Why**: Additive blending $q' = \alpha q + (1-\alpha) h$ pollutes query centroid. Multiplicative: $S_{\text{final}} = S_{\text{semantic}} \cdot [1 + \gamma \cdot \text{ReLU}(C_{\text{context}})]$ only for $S_{\text{semantic}} \ge \tau_{\text{base}}$.
- **Implementation**: In `DroidEngine.recall()`, compute semantic sims via `KnowledgeStore`, compute contextual alignment with `memory.h_trace`, fuse with hard gate $\tau_{\text{base}}=0.48$, boost $\gamma=0.2$.

### Decision 4: Ebbinghaus power-law + sleep consolidation
- **Why**: Continuous learning without bounds saturates edge storage. Power-law models spacing effect; sleep consolidation absorbs salient facts into parametric weights and prunes episodic store.
- **Implementation**: `DroidEngine.consolidate_memory(max_prune_ratio=0.1)` called from idle callback or manual trigger. Tracks `access_count`, `last_retrieved` per fact in `KnowledgeStore` (extend schema).

### Decision 5: Snapshot via safetensors + knowledge store checkpoint
- **Why**: Safetensors is already used for `memory.safetensors`; extend to include associative matrices. Knowledge store already has `checkpoint()`/`export_to()`.
- **Implementation**: `PlasticAssociativeRTU.state_dict()` includes all buffers; `DroidEngine.create_snapshot()` bundles with `knowledge.checkpoint()` and version metadata.

## Risks / Trade-offs

- **[Risk]** Associative matrix $S$ (4 × 96 × 96 = 36,864 params) adds ~0.15 MB — acceptable. → **Mitigation**: Keep heads=4 default; expose as config.
- **[Risk]** Surprise threshold $\theta_{\text{surprise}}$ tuning affects fact retention. → **Mitigation**: Default 0.4; expose in profile config; log surprise distribution.
- **[Risk]** Sleep consolidation gradient steps may drift associative weights. → **Mitigation**: Low LR ($10^{-4}$), orthogonality regularization $\lambda_{\text{ortho}} \|W^\top W - I\|_F^2$, only 3–5 steps.
- **[Risk]** Knowledge store schema needs `access_count`, `last_retrieved` columns for Ebbinghaus. → **Mitigation**: Add columns with default values; migration on load.

## Migration Plan

1. Create `PlasticAssociativeRTU` in `src/model/rtu.py` alongside `RTUMemoryBlock`
2. Update `DroidEngine.__init__` to instantiate `PlasticAssociativeRTU` as `self.memory`
3. Update `teach()`: add surprise computation, gated episodic insert, associative update
4. Update `recall()`: add contextual multiplicative scoring with hard gate
5. Add `consolidate_memory()`, `create_snapshot()`, `restore_snapshot()`, `get_memory_health()` to `DroidEngine`
6. Extend `KnowledgeStore` schema with `access_count`, `last_retrieved` columns (migration on load)
7. Update `save_profile()`/`load_profile()` to handle snapshot versioning
8. Add tests: `tests/test_rtu.py` for associative memory, `tests/test_droid_engine.py` for integration
9. Run full test suite

## Open Questions

- Should surprise threshold be per-profile adaptive (e.g., percentile of recent surprises)? → Start fixed, adapt in v1.3.
- Should consolidation run automatically on idle or only manual? → Start with manual trigger + CLI command; auto in v1.3.
- Snapshot format: include knowledge store or reference by version? → Include full knowledge DB export for portability (matches `.droid` package).