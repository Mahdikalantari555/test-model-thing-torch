# Design

## Context

See `proposal.md` for motivation and capability scope, and `specs/` for normative behavioral requirements.

Verified state on `Experiment` (`ed7ad9db`) shaping this design:
- **Existing `Experiment` primitives that must be tested rather than re-invented:**
  - `src/model/rtu.py` & `src/model/parallel_scan.py`: unstacked diagonal Byte-RTU (`C0`), custom AdamW, 4-term byte loss (`O2`), `--ce-only` (`O1`), and exact diagonal eligibility traces (`O4`).
  - `src/model/decision_head.py` & `src/model/confidence.py`: hyperspherical cosine prototype router (`D1`), closed-form Ridge probe (`D2`), normalized Shannon entropy confidence ($1 - H_{\text{norm}}$), and harmonic confidence fusion (`R2`).
  - `src/model/delta_memory.py`: `DeltaPlasticAdapter` (multi-head gated delta rule + Householder erase) and `SparseCoder` / `SparseDeltaAdapter` (`384 -> 2048`, TopK-128, 93.75% sparsity) (`M4`).
  - `src/model/knowledge_store.py` & `src/model/hybrid_retrieval.py`: SQLite WAL + FTS5 BM25 + dense vector RRF + belief revision (`M3`).
  - `src/model/edge_pretraining.py` & `src/model/benchmarks.py`: SHA-256 document-level splits, 64-byte window contamination checks, median-after-warmup timing, `VARIANCE_THRESHOLD = 0.10`, `LOAD_THRESHOLD = 0.40`, and `skipped`/`blocked` honesty rules.
- **Hardware & environment constraints:**
  - Target environments span multi-core developer laptops (8–24 GB RAM) down to this 2-vCPU Xeon @ 2.60 GHz / **3.8 GiB RAM (0 swap)** sandbox with restricted network (no Hugging Face TLS access; PyPI/GitHub/npm only).
  - Naive `batch = 32 × 512` fp32 activations will OOM on 3.8 GiB RAM, and external datasets (`TinyStories`, `BLiMP`, `CLINC150`, `Banking77`) may be unreachable offline.
- **Coexistence with active OpenSpec changes:**
  - `post-merge-completion` owns release-evidence ledger scripts; we do not modify its inventory or weaken any threshold.
  - `recamem-plastic-memory-engine` transforms `src/` into `recamem/` without training/selecting a new model core; this change runs cleanly on `Experiment` (`src/model/tournament/`, `src/model/cores/`, `src/model/recamem_core.py`, `configs/recamem_selected_core.json`) so `recamem` imports the empirically crowned winner as its canonical offline core.

## Goals / Non-Goals

**Goals:**
- Isolate all **24 candidate arms across 5 orthogonal tracks** behind zero-overhead Python protocols so every single design axis is tested ceteris paribus.
- Guarantee training and benchmarking fit comfortably inside **< 2.5 GiB peak RSS** on 2-vCPU hardware while scaling linearly when run on larger multi-core CPUs.
- Eliminate researcher degrees of freedom via **pre-registered Stage 1–4 numeric gates**, separate `val_cal` vs. `test_eval` splits, per-arm LR sweeps, and 3/5-seed paired bootstrap 95% CIs.
- Emit a single declarative winner (`configs/recamem_selected_core.json`) and factory (`build_recamem_core`) plus full audit reports (`reports/recamem-core-selection.md`, `reports/evidence/recamem-core-tournament.json`).

**Non-Goals:**
- Claiming novel foundational mechanisms (all candidates—Gated DeltaNet, Mamba-style SSM, RetNet, Needle SAN, Engram, JEPA/VICReg—are controlled implementations of prior art).
- Custom CUDA/Triton kernels (everything must run in pure PyTorch CPU fp32 + `torch.ao.quantization` dynamic INT8).
- Modifying `droids/` profile persistence formats or breaking any of the 912 existing pytest tests.

## Decisions

### D-1. Five-slot orthogonal plug-in architecture (`src/model/cores/` & `src/model/tournament/`)

Instead of building three monolithic `CandidateA`, `CandidateB`, and `CandidateC` classes, every model is assembled from five orthogonal protocol slots:

```
Tokens/Bytes ──► [Slot 1: TokenizerAdapter (V=256 Bytes | V=4096 BPE)]
                      │
                      ▼
               [Slot 2: SequenceCore (C0 | C1 | C2 | C3 | C4 | C5 | C6)]
                      │  ▲
                      │  └── [Slot 5: MemoryModule (None | M0 | M1 | M2 | M3 | M4 | M5)]
                      ▼
         ┌────────────┼────────────────────┐
         ▼            ▼                    ▼
   [Tied LM Head]  [Slot 3: DecisionHead] [Slot 4: ActionRouter + Guards]
   + [AuxLoss O1..3]  (D1 | D2 | D3 | D4)    (R1 | R2 | R3 | R4)
```

- **Why over monolithic candidate classes:** Candidate C = `C4 + M1 + M2 + D3 + R4`; Candidate A = `C4 + D3 + R4`; Candidate B = `C1 + O3 + D3`. Monolithic classes duplicate shared code and make it impossible to tell whether Candidate C's score comes from `C4`, `M1`, or `M2`. Slot composition guarantees identical initialization, optimizer, batch order, and evaluator around whichever single slot is varied.
- **Alternative considered:** Subclassing `src.model.rtu.Model`. Rejected because `rtu.Model` hardcodes `vocab_size=256`, per-byte state buffers, and unstacked layers; wrapping `rtu.Model` as `C0` inside the new `SequenceCore` protocol keeps 100% MLX parity untouched while keeping `C1..C6` clean.

### D-2. Pure-PyTorch CPU Chunkwise WY / Parallel Scans for Matrix & Diagonal Recurrences

- For **`C4` (`gated_delta_matrix`)** and **`C3` (`retnet_decay_matrix`)**, token-by-token Python loops over $T=512$ are 10–30× too slow on CPU, while materializing $T \times T$ matrices at $T=4096$ destroys linear scaling. We implement **chunkwise parallel form (`chunk_size = 64`)**: intra-chunk parallel matmuls (using BLAS/AVX-512 on $64 \times 64$ blocks) combined with inter-chunk recurrent state updates on $S \in \mathbb{R}^{h \times d_v \times d_k}$, plus an exact step-by-step recurrent path (`step()`) for $O(1)$ autoregressive decode and numerical equivalence testing (`atol=1e-4`).
- For **`C1` (`stacked_gated_rtu`)** and **`C2` (`selective_ssm`)**, reuse and extend `src/model/parallel_scan.py`'s associative binary operator `(a_2, b_2) ∘ (a_1, b_1) = (a_2 a_1, a_2 b_1 + b_2)` over chunks.
- **Alternative considered:** Custom C++ CPU extensions. Rejected: complicates portability and hides algorithmic fairness behind compiler flags.

### D-3. Micro-Batched Gradient Accumulation & Multi-Fidelity Compute Tiers for 3.8 GiB CPU Safety

- Effective batch sizes (e.g. `16_384` tokens = `32 × 512`) are executed via `micro_batch_size = 4` (or `2` at Tier-Target) × `grad_accum_steps = 8` (or `16`), verified by a unit test asserting `< 1e-5` relative gradient difference vs. single-pass batching. Peak training RSS stays under **1.2 GiB** at Tier-Small (~6M) and **2.2 GiB** at Tier-Target (~17M).
- Three parameter-matched profiles (`±2%` budget verified automatically via solver over `d_model` and `d_ff`):
  1. **`micro` (~2.0M params):** Stage-1 MQAR/Copy falsification, objective sanity check, and 3-point LR sweep (`{3e-4, 1e-3, 3e-3}`). Runs in minutes on 2 vCPUs.
  2. **`small` (~6.0M params):** Stage-2 isolated track tournaments and Stage-3 Shared-vs-Split interference gate (3 seeds).
  3. **`target` (~17.0M params):** Stage-4 forward assembly, Leave-One-Out pruning, and dynamic INT8 validation (5 seeds; supports `--profile ci-fixture` for deterministic sandbox verification and `--profile full-scale` for multi-hour 30M–100M token runs, stamping the exact profile and token count in the JSON ledger).

### D-4. Deterministic Self-Contained BPE (`V=4096`) + Strict Bits-Per-Byte (`BPB`) Accounting

- Implement a minimal, zero-network byte-level BPE trainer/encoder (`src/model/cores/tokenizer.py`, compatible with `tiktoken`/`tokenizers` if installed, pure Python/PyTorch fallback otherwise) trained deterministically on the training split only (never touching `val_cal` or `test_eval`), reserving IDs `0..7` for `PAD`, `EOS`, `OPT`, `SEP`, `ROUTE_GENERATE`, `ROUTE_CLASSIFY`, `ROUTE_RETRIEVE`, `ROUTE_TOOL`, `ROUTE_ABSTAIN` (`9` control IDs total).
- Every sequence batch records `raw_utf8_bytes` alongside token IDs so `BPB = total_nll_nats / (raw_utf8_bytes * math.log(2.0))` is exact across `V=256` and `V=4096`.

### D-5. Two-Stage Calibration & Guard Separation (`val_cal` vs. `test_eval`)

- Held-out data is split deterministically (`sha256(seed:doc_id)`) into **50% `val_cal`** and **50% `test_eval`**.
- Only `val_cal` is used to fit L-BFGS scalar temperature $T_{\text{cal}}$ and select risk-coverage abstention thresholds $\tau_{\text{abstain}}$ at target risk ceilings (e.g. `5%` selective error).
- Only `test_eval` produces the tournament leaderboard numbers (`BPB`, `Macro-F1`, `Unseen-Option Acc`, `Adaptive ECE`, `Brier`, `AURC`, `Cost-Weighted Routing Regret`), including an out-of-distribution/label-renamed shift slice to catch fragile temperature scaling.

### D-6. Canonical RecaMem Core Export (`configs/recamem_selected_core.json` + `src/model/recamem_core.py`)

When Stage 4 finishes, the tournament runner writes:
1. `configs/recamem_selected_core.json`: declarative spec of the winning topology (`topology: "unified_shared" | "split_dual"`, `core_id`, `tokenizer`, `d_model`, `n_layers`, `n_heads`, `conv_kernel`, `decision_head_id`, `router_id`, `memory_arms`, `calibration_temp`, `guard_thresholds`, `quantization`).
2. `src/model/recamem_core.py`: exposes `RecaMemCoreBundle` and `build_recamem_core(manifest_path=...)` implementing:
   - `generate(...)` + `capabilities` (`BackendCapabilities` contract for `DroidEngine` and `recamem/backends.py`),
   - `decide(query, options, ...)` (`DecisionHead` contract + dynamic options + null class),
   - `route(query, context, schema, ...)` (5-way guarded router),
   - `memory` (`PlasticAdapter` / episodic two-channel contract for `recamem/plasticity.py` and `recamem/retrieval.py`).

This guarantees zero manual translation bugs between the tournament winner on `Experiment` and the `recamem` package rewrite.

## Risks / Trade-offs

- **[Risk] 2-vCPU sandbox wall-clock limits full 100M-token × 24-arm sweeps in a single interactive session** → Mitigation: Resumable atomic stage checkpoints (`edge_pretraining.py` pattern) + explicit `--scale-tier {smoke,micro,small,target}` CLI flags; every JSON report records exact tokens/bytes consumed and marks unrun full-scale tiers as `skipped`/`blocked` rather than claiming full-scale convergence from a micro run.
- **[Risk] External Hugging Face datasets (`TinyStories`, `CLINC150`, `Banking77`, `BLiMP`) blocked by sandbox network rules** → Mitigation: Ship deterministic synthetic generators (`MQAR`, `Copy`, `Extrapolation`) plus committed license-clean mini-corpora and multi-domain intent/routing/contradiction fixtures under `evals/tournament_fixtures/`, stamping `dataset_source="fixture"` vs. `"real"` in every report.
- **[Risk] Option-conditioned head (`D3`) underfits unseen option text at `< 20M` parameters** → Mitigation: Stage 2 explicitly tests held-out renamed labels; if zero-shot text scoring lags, `D3` supports hybrid prototype + text option embeddings (`D1 + D3`) and lets the tournament data decide cleanly.
- **[Risk] CPU chunkwise scan overhead erases theoretical $O(T)$ advantage over Softmax Transformer at $T \le 1024$** → Mitigation: Pre-registered Stage-2 Edge Advantage Gate measures real CPU `ttft_ms`, `decode_bytes/s`, and `peak_rss_mb` against `C5` (Needle SAN) and `C6` (Transformer); if attention wins on real CPU metrics within the target context window, the funnel reports that honestly rather than forcing a slower recurrent winner.

## Migration Plan

1. **Additive implementation on `Experiment`:** All candidate modules land under `src/model/cores/`, `src/model/tournament/`, and `src/model/recamem_core.py`, with CLI entry point `scripts/run_core_tournament.py`. Existing `src/model/*` files stay backward-compatible.
2. **Execute tournament stages 0 → 4:** Produce `reports/evidence/recamem-core-tournament.json`, `reports/recamem-core-selection.md`, and `configs/recamem_selected_core.json`.
3. **Bridge into `recamem`:** During `recamem-plastic-memory-engine` apply (Phase 1 / Phase 3 / Phase 6), `recamem/backends.py`, `recamem/rtu.py`, and `recamem/decision_head.py` import/adopt `RecaMemCoreBundle` configured by `configs/recamem_selected_core.json` instead of the legacy unstacked byte-RTU toy.
4. **Rollback:** Because all new modules and reports are additive and existing `DroidEngine` / `ByteRtuBackend` defaults are untouched unless `engine="recamem_core"` is requested, reverting this change is a clean directory removal with zero profile migration rollback needed.
