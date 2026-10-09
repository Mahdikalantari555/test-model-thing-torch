# Plan — RecaMem: turn this repo into a local-first plastic memory engine

Status: **plan (not implemented)**. Supersedes nothing in `plans/01–08` (those govern the
MLX→PyTorch reproduction, which this plan preserves as an optional built-in micro-model).

## Goal

Rebrand and restructure this repo in place into **RecaMem**: a local-first plastic memory engine
for foundation models. Foundation models stay frozen and supply language; RecaMem supplies adaptive
long-term memory, online learning, consolidation, forgetting, and trace-based recall — and every
claim is measured, not asserted.

## Decisions (recorded)

| # | Decision |
|---|---|
| 1 | Rebrand in place. `recamem` package; the byte-level RTU LM becomes an **optional built-in offline micro-model** (`EmbeddedRTUBackend`), not the headline. Any frozen FM plugs in via `recamem/backends.py` (Ollama / OpenAI-compatible / local GGUF). |
| 2 | **Two-tier plasticity.** FAST: no-grad Hebbian/delta-rule writes to associative state `S` + `h_trace` (immediate). SLOW: replay-trained meta-weights (`W_K/W_V/W_Q/W_pred/W_gate/W_alpha/beta`) during offline "sleep" cycles, protected by MAS importance + a validation gate with rollback. |
| 3 | Identity: package `recamem`, CLI `recamem`, profiles in `~/.recamem/agents/<name>/` (`RECAMEM_HOME` override). `droids/` remains readable; `recamem migrate droids/` converts. |
| 4 | Embedder: fail-fast real INT8 MiniLM. Mock is **explicit opt-in** (`--anchor mock`, `RECAMEM_ANCHOR=mock`). Anchor mode surfaced in `health()`, `recamem doctor`, WebUI, and every benchmark report. |
| 5 | Forgetting: strength decay (power law over time-since-retrieval × reinforcement × novelty × pin/feedback), `K_max` actually enforced, **consolidate-then-forget** (replay into plastic state before eviction), soft `retired` cold archive instead of hard DELETE. |
| 6 | Recall: **two-channel fusion** — plastic prior (trace-conditioned re-ranking + novelty/confidence) + episodic verbatim text. Hardcoded remote-sensing-tuned boosts are deleted. Retrieval writes back to the trace (reconsolidation on recall). |
| 7 | Validation: LoCoMo + LongMemEval-S + DMR, integrated via download-once cache into `~/.recamem/benchmarks/`, with a committed deterministic seed-selected subset and an offline bundled-fixture fallback. |

## Verified audit (evidence behind the plan)

Fact-checked against source, not inferred:

- **Cosmetic learning rate.** `effective_lr` computed at `src/model/droid.py:427`, reported at `:497`, **never applied to any update**. Biggest honesty gap vs. "online learning".
- **No gradient path through memory.** Every plastic update is `torch.no_grad()` (`src/model/rtu.py:571`, `:604`), so `W_K/W_V/W_Q/W_pred/W_gate/W_alpha` stay at `N(0, 1/sqrt(dim))` random init forever.
- **`retrieve()` is dead code.** Implemented at `src/model/rtu.py:611-641`, never called by `DroidEngine`. Trace-based recall does not exist in production; `h_trace` only feeds a hardcoded `1 + 0.2*contextual` boost (`droid.py:569-580`).
- **Recall tuned to the demo fixture.** `droid.py:542-580`: definitional `+0.25/+0.2` boost keyed on "is the acquisition"/"is a"/"is an", rare-word boost up to `+0.5`, gate `sim < 0.48`.
- **`K_max` never enforced.** Declared `droid.py:338`; no eviction code references it.
- **Forgetting manual and irreversible.** `consolidate_memory()` `droid.py:710-758` hard-DELETEs pruned facts; no scheduler ever calls it.
- **Two broken entry points.** `clear_knowledge()` called at `app.py:198` and `scripts/verify_remote_sensing_e2e.py:18` but **not defined on `DroidEngine`** — WebUI "Reset Memory" and the documented E2E script both raise `AttributeError`.
- **Silent mock anchor.** `OnnxMiniLM` falls back to a hashed bag-of-words embedder with hardcoded `remote_sensing`/`food`/`capital`/`definition` clusters on any exception (`onnx_anchor.py:100-110`, `:241-244`). `tests/test_onnx_anchor.py` passes against that mock.
- **Broken profile serialization.** `save_profile` adapter detection (`droid.py:1233-1244`) reads `self.memory._adapter_type`, never an attribute — only a state-dict key — so it always falls through to a class-name heuristic. Adapter metadata is silently dropped by the tensor-only filter at `:1262-1264`.
- **Dead/duplicated code.** Second `RTUMemoryBlock` at `droid.py:163-203` shadowed by the import at `:209`. `error_vec` assigned three times at `rtu.py:524-529`.
- **God class.** `DroidEngine` = 1,417 lines / ~40 public methods, with `hasattr()` duck typing throughout.
- **Untested surfaces.** No tests for `memory_lifecycle.py` (614), `droid_package.py`, `gguf_backend.py`, `mcp_server.py`, `cli.py`, `confidence.py`, `propositions.py`, `decision_head.py`, `plastic_adapter.py`, or the associative RTU's `retrieve`/`merge` paths.
- **README overstates.** Claims "43 tests" (actual: 57 test functions; 8,973 lines total across src/app/tests/scripts) and "zero catastrophic collapse" — `tests/test_conversational_teaching.py` only asserts the engine never goes silent; forgetting is never measured.
- **No packaging or CI.** No `pyproject.toml`/`setup.py`, no console entry point (CLI declares `prog="tmt-droid"` but nothing installs it), no `.github/workflows`, no lint/type config. `scripts/verify_reproduction.py` imports `psutil`, which is not in `requirements.txt`.
- **Latent thread-safety bug.** `KnowledgeStore` opens SQLite with `check_same_thread=False` (`knowledge_store.py:62`) while `ContinuousStream` runs a daemon thread — no single-writer lock.

## Target architecture

```
recamem/
  __init__.py          version + public API
  __main__.py          python -m recamem
  cli.py               chat teach recall decide serve mcp doctor bench migrate package rtu
  mcp_server.py        stdio JSON-RPC MCP (logs to stderr)
  anchor.py            MiniLMAnchor (fail-fast, INT8 ONNX) + MockAnchor (explicit) + health()
  adapters.py          PlasticAdapter ABC + registry (tmt.plastic_adapters -> recamem.plastic_adapters)
  rtu.py               RTUBlock: gated delta-rule associative state, surprise gating, bounded erase
  embedded_rtu.py      optional offline byte-level micro-model (ported Model/Layer/losses/optimizer)
  plasticity.py        FAST tier (no-grad writes) + SLOW tier (replay-trained meta-weights, MAS, decreasing LR)
  consolidation.py     SleepConsolidation, replay scheduling, validation gate, rollback, auto-sleep worker
  forgetting.py        strength model, decay, K_max eviction, consolidate-then-forget, cold archive
  retrieval.py         two-channel fusion (plastic prior + episodic verbatim) + reranker + writeback
  episodic.py          EpisodicStore: SQLite FTS5 + RRF + ANN index, versioned schema
  engine.py            MemoryEngine — orchestration only (target <400 lines)
  registry.py          AgentRegistry (~/.recamem/agents/<name>/)
  lifecycle.py         snapshots, merge, semantic distill, bandit policy, continuous stream
  backends.py          GgufBackend, EmbeddedRTUBackend, OpenAICompatibleBackend + registry
  teacher.py           stdlib urllib distillation / synthesis (unchanged semantics)
  propositions.py  confidence.py  decision_head.py  injection.py
  package.py           AgentPackage: zip export/import + SHA-256 manifest
  bench/               download.py locomo.py longmemeval.py dmr.py lifelong.py report.py
```

`src/model/droid.py` (1,417 lines) decomposes into `engine.py` + `plasticity.py` + `retrieval.py` +
`consolidation.py` + `forgetting.py`. `src/model/rtu.py` keeps the delta-rule RTU and gains the slow
tier; the byte-level `Model`/`Layer`/`losses`/`CustomAdamW` move to `embedded_rtu.py`.

## Ordered tasks

### Phase 0 — Correctness fixes before any rename
Land these first so the rest of the work builds on a green suite.

1. Add `clear_memory()` to `DroidEngine` (`knowledge.clear_all()` + `memory.reset()` + `step_count = 0`),
   plus a `clear_knowledge = clear_memory` alias. Unblocks `app.py:198` and
   `scripts/verify_remote_sensing_e2e.py:18`.
2. Make `effective_lr` load-bearing: remove it from the cosmetic result dict and wire it into the
   SLOW-tier LR schedule so `base_lr / sqrt(1 + steps/20)` actually governs replay training.
3. Add `psutil` to `requirements.txt`; add a `pyproject.toml` stub so pytest/ruff/mypy config has a home.
4. Write the missing tests **first** as behavioral pins: associative `retrieve()`, `merge_memory()`,
   `consolidate_memory()` pruning/regularization, `SnapshotManager.restore_snapshot`,
   `KnowledgeStore` RRF fusion, `DroidPackage` round-trip. Target ≥15 new tests before Phase 1.

### Phase 1 — Package skeleton and rename
1. `git mv src recamem`; `pyproject.toml` with console entry points `recamem` and `recamem-mcp`.
2. Add a compatibility shim so `from src.model.droid import DroidEngine` still imports with a
   `DeprecationWarning` — keeps `benchmark.py`, `main.py`, and downstream imports working during the
   transition. Delete the shim in Phase 9.
3. Class renames with deprecation aliases: `DroidEngine`→`MemoryEngine`, `DroidManager`→`AgentRegistry`,
   `DroidPackage`→`AgentPackage`, `OnnxMiniLM`→`MiniLMAnchor`, `KnowledgeStore`→`EpisodicStore`,
   `PlasticAssociativeRTU`→`RTUBlock`. `PlasticAdapter` stays as-is.
4. Decompose the god class into `plasticity.py`, `retrieval.py`, `consolidation.py`, `forgetting.py`;
   `engine.py` keeps orchestration. Delete dead `RTUMemoryBlock` (`droid.py:163-203`) and the
   `error_vec` triple-assignment (`rtu.py:524-529`).
5. Replace the ~102 bare `except:` clauses and the defensive import chains in `src/model/__init__.py`
   and `rtu.py:22-27` with real exception types. No silent degradation.

### Phase 2 — Embedder honesty
1. `MiniLMAnchor` raises on load failure; `MockAnchor` is constructed explicitly. Both satisfy one
   `Anchor` protocol (`embed`, `dim`, `mode`).
2. `health()` reports `{"anchor": "real"|"mock", "model_path":..., "load_ms":...}`; surface in
   `recamem doctor`, CLI `--json`, and every benchmark report.
3. Rewrite `tests/test_onnx_anchor.py` to construct `MockAnchor` explicitly; add a real-anchor test
   that skips loudly when the HF cache is absent rather than silently passing on the mock.

### Phase 3 — Two-tier plasticity (Decision 2)
1. **FAST tier** (unchanged semantics, cleaned): gated delta-rule write
   `S_t = S_{t-1}(1-β_t) + β_t·outer(K_t, V_t - K_tᵀS_{t-1})`, per-head bounded `β_t = sigmoid(W_β e_t)`,
   Householder-style bounded `erase()`, surprise-gated `h_trace` EMA. All under `no_grad`.
2. **SLOW tier** (new): replay-trained projections. Objective = associative reconstruction on replayed
   experience, optimized with the ported `CustomAdamW`, guarded by a MAS penalty
   `λ·Σ_i F_i(θ_i - θ*_i)²` (`F_i` = output-sensitivity importance from one backward pass per batch) and
   a decreasing schedule `lr_i = base_lr · decay^i` seeded from the now-load-bearing `effective_lr`.
   Persist MAS importance/masks and the schedule index across restarts.
3. **Schema-versioned state**: `state_dict()` gains `_schema_version`, `_adapter_type`, `dim`, `heads`.
   Fix `save_profile` adapter detection (currently reads a nonexistent attribute, `droid.py:1233-1244`)
   and stop filtering metadata out. Explicit loader for v1 `associative_rtu` states — no silent reshaping.

### Phase 4 — Consolidation / sleep (Decision 2)
1. `SleepConsolidation.run()` returns a structured report and replaces `consolidate_memory()`:
   surprise-prioritized replay selection, **interleaved** replay of novel + familiar traces (shown to
   prevent interference), capped replay steps, self-inhibition against attractor loops, and coverage
   reporting rather than silent "recovered" claims.
2. **Validation gate**: before accepting a sleep cycle, score a persisted per-domain probe set; on
   regression beyond tolerance, roll back to the pre-sleep snapshot and report which domains regressed.
   This is what makes "no catastrophic collapse" a measured property.
3. Auto-sleep worker: one cancellable worker per agent, guarded by a single engine lock, never
   duplicated on Streamlit reruns. Persisted sleep config.
4. Add a single-writer lock around `EpisodicStore` for consolidation/teaching/retrieval concurrency
   (currently `check_same_thread=False` with no lock, `knowledge_store.py:62`).

### Phase 5 — Forgetting (Decision 5)
1. Per-fact `strength` column with power-law decay over time-since-last-retrieval, reinforced by
   `access_count`, scaled by `novelty`, overridable by explicit pin/approve/reject.
2. Enforce `K_max`: when `active_count > K_max`, evict the weakest unpinned facts.
3. **Consolidate-then-forget**: each evicted fact is replayed into plastic state first, then moved to a
   `retired` cold tier (`deleted_at`, queryable with `--include-retired`) instead of hard-DELETEd.
4. Test: a taught-but-never-accessed, unpinned fact must become retrievable only from the plastic
   channel after eviction; pinned facts must never be evicted.

### Phase 6 — Two-channel recall (Decision 6)
1. Wire `retrieve()` into the engine. Channel A (plastic): query-conditioned read through `S`/`h_trace`
   → re-ranking prior + novelty/confidence. Channel B (episodic): hybrid BM25 + dense RRF → verbatim,
   auditable text with source attribution.
2. **Delete the tuned boosts** at `droid.py:542-580`; replace with a documented feature scorer whose
   weights are fit on a shipped labeled retrieval set. The E2E fixture stays green as a regression test,
   not via magic constants.
3. **Reconsolidation on recall**: a successful retrieval writes back to the trace and increments
   access/strength, so retrieval itself reinforces memory (ties into Phase 5).
4. Stop discarding the RRF rank signal (`droid.py:536` discards `fused_score`); replace the SQLite
   `FULL OUTER JOIN` emulation (`knowledge_store.py:369-476`) with the `UNION` + post-filter path that
   already exists.
5. Add an ANN dense-candidate index (HNSW or `sqlite-vec`) with an exact brute-force fallback; keep the
   `<5 ms @ 100k facts` target from the existing `knowledge-store` spec and benchmark before changing
   the default backend.

### Phase 7 — Benchmarks (Decision 7)
1. `recamem/bench/download.py` fetches LoCoMo, LongMemEval-S, and DMR once into
   `~/.recamem/benchmarks/`.
2. Select a deterministic N-question subset by seed; commit its hash/sample so runs are reproducible.
3. `lifelong.py`: sequential multi-domain teaching sequence reporting backward transfer, forgetting,
   and mean memory loss — the acceptance metric the existing `delta-plastic-memory` change targets.
4. `report.py` writes `reports/benchmarks/<date>-<commit>.md` including config, seed, anchor mode,
   backend, and ablations (plastic on/off, real vs mock anchor, consolidation on/off) so every number
   has provenance.
5. Offline fallback: if the cache is absent, run a bundled local fixture and mark
   `provenance=fixture`. Offline mode must never fail.

### Phase 8 — Surfaces and platform
1. CLI: `recamem chat|teach|recall|decide|serve|mcp|doctor|bench|migrate|package|rtu`. `rtu` runs the
   optional embedded micro-model (`main.py` / `main_mlx.py` behavior).
2. Move `main_mlx.py` and `benchmark_mlx.py` to `legacy/mlx/` with a README; keep `main.py` working via
   the `recamem rtu` subcommand.
3. MCP server: rebrand tool names to `recamem_*`, keep the JSON-RPC contract stable.
4. WebUI: rebrand "Droid" → agent, fix the broken Reset Memory button, expose anchor mode and memory
   health in the sidebar. Modularizing the 757-line `app.py` into `recamem/webui/` is recommended but
   may be deferred.
5. CI: GitHub Actions running `pytest tests` plus the offline fixture benchmark as a gate against a
   committed baseline file.

### Phase 9 — Specs and docs
1. OpenSpec: create one change `recamem-plastic-memory-engine` superseding the three active ones
   (`delta-plastic-memory`, `autonomous-sleep-consolidation`, `edge-slm-optimization`) — their content
   is the intended direction and is absorbed into Phases 3–6 — then archive them. Add new specs for
   `plasticity/two-tier`, `memory/forgetting`, `memory/consolidation`, `retrieval/two-channel`.
2. Rewrite `README.md` around the RecaMem description with a benchmark table sourced from
   `reports/benchmarks/`. Remove the "43 tests" and "zero catastrophic collapse" overstatements, or
   back them with measured numbers.
3. Rewrite `AGENTS.md` as the RecaMem runbook (new module boundaries, the `ai` conda env, the
   safetensors / in-place-parameter / HF-cache gotchas, plus new rules: never re-add a silent mock
   fallback, never hard-delete a fact without consolidate-then-forget).
4. Add this file to the read-order index in `plans/README.md`.

## Risks

- **Slow-tier instability.** Replay-trained projections can diverge on CPU. Mitigation: bounded `β`,
  L2-normalized keys/values, MAS penalty, decreasing LR, validation-gate rollback, hard cap on replay
  steps. The validation gate is the safety net, not an optimization.
- **Removing tuned recall boosts regresses the E2E fixture.** The E2E test must stay green through the
  Phase 6 rewrite; if the learned scorer cannot reproduce it, report that as a finding rather than
  restoring the constants.
- **Backward compatibility.** `droids/` profiles and v1 `associative_rtu` states must load unchanged via
  `recamem migrate`; incompatible shapes require explicit migration and a loud error.
- **Benchmark credibility.** LoCoMo is the most-quoted and least-discriminating memory benchmark (DMR
  scores already cluster in the nineties). Report per-question-type breakdowns and ablations; never
  publish a headline number without the protocol.
- **Licensing and size.** LongMemEval/LoCoMo redistribution is not assumed; download-on-cache with a
  committed deterministic subset avoids vendoring third-party data into an MIT repo.
- **Scope creep.** `app.py` modularization, `EmbeddedRTUBackend` polish, and multi-agent merging can
  slip; the memory engine must not.

## Out of scope

Cloud hosted service, multi-user auth/tenancy, rewriting the Streamlit UI into a SPA, training a new
foundation model, and full external-benchmark publication in the first delivery.

## Validation plan

1. `PYTHONPATH=. /home/asus/miniforge3/envs/ai/bin/pytest tests` — all pre-existing 57 tests green
   modulo intentional, documented changes (recall-boost removal, anchor opt-in), plus ≥15 new tests
   from Phase 0 covering `retrieve`, `merge`, `consolidate`, snapshots, and package round-trip.
2. `PYTHONPATH=. /home/asus/miniforge3/envs/ai/bin/python scripts/verify_remote_sensing_e2e.py` — must
   stop raising `AttributeError` and pass end to end.
3. `recamem doctor` reports real-anchor mode with model path and load time.
4. `recamem bench --fixture` runs fully offline and writes a report with `provenance=fixture`.
5. `recamem bench` (cache present) reports LoCoMo per-type F1, LongMemEval-S per-ability accuracy, DMR
   accuracy, and the lifelong sequence's backward transfer / mean memory loss.
6. Forgetting-curve test: consolidate-then-forget verified — knowledge survives as a plastic trace after
   episodic eviction, pinned facts never evicted, nothing hard-deleted.
7. Validation-gate test: teach domain A, sleep, then teach domains B–F; all A probes must remain within
   tolerance or the cycle rolls back and is reported.
8. CI green: pytest + offline fixture benchmark against the committed baseline.

## Open questions

None blocking. Deferred to implementation: exact labeled-retrieval-set construction and reranker feature
set (Phase 6); whether `app.py` is modularized in this change or a follow-up (Phase 8); whether the
`tmt.plastic_adapters` entry-point group is renamed to `recamem.plastic_adapters` with a compatibility
alias (recommend: yes, with alias).
