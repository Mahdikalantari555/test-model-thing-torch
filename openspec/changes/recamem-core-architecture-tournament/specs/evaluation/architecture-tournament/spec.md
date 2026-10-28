# Spec Delta

## Purpose

Establishes a reproducible, ceteris-paribus benchmark suite and a pre-registered 4-stage elimination, interference, and leave-one-out pruning tournament that ranks all 24 candidate arms fairly and extracts one empirically superior, minimal architecture.

## ADDED Requirements

### Requirement: Multi-dimensional capability and calibration benchmark suite
The tournament harness SHALL evaluate candidates across six standardized workload families with deterministic seeds and contamination checks:
1. **Associative recall & copying:** synthetic Multi-Query Associative Recall (`MQAR` at 4, 16, 32, and 64 key-value pairs across context lengths 128–1024) and exact string copying;
2. **Language modeling & length extrapolation:** held-out Bits-Per-Byte (`BPB`), contrastive syntax probes (`evals/grammar_probes.jsonl` and BLiMP phenomena), and length extrapolation trained at $T=256$ and evaluated at $T \in \{512, 1024, 2048, 4096\}$;
3. **Dynamic System-1 decisions:** closed-option classification with variable option counts ($K \in [2, 20]$), shuffled option orders, held-out unseen option label phrasings, and out-of-scope rejection (`AUROC`);
4. **Probabilistic calibration & selective prediction:** post-temperature-scaling Adaptive ECE (15 equal-mass bins), Brier score, and Area Under the Risk-Coverage Curve (`AURC`) on both in-domain and distribution-shifted splits;
5. **Five-way agent routing:** cost-weighted action regret (penalizing false tool execution or ungrounded generation strictly higher than safe abstention), schema compliance rate, and p95 routing latency;
6. **Continual learning & edge efficiency:** multi-domain retention, non-monotonic contradiction updates, peak RSS, disk size, prefill TTFT, and fp32 / dynamic-INT8 decode throughput.

#### Scenario: Calibration parameters never leak into evaluation splits
- **WHEN** temperature scaling ($T_{\text{cal}}$) or risk-coverage abstention thresholds are fit during evaluation
- **THEN** they SHALL be fit exclusively on a disjoint `val_cal` split
- **THEN** reported `ECE`, `Brier`, `AURC`, and routing regret SHALL be computed exclusively on an untouched `test_eval` split with zero document or query overlap

#### Scenario: Offline deterministic fixtures run when external datasets are unreachable
- **WHEN** external corpora or benchmark datasets (such as TinyStories, BLiMP, CLINC150, or Banking77) are absent from cache and unreachable offline
- **THEN** the harness SHALL execute bundled deterministic synthetic/fixture workloads and stamp every output record with `dataset_source="fixture"` (or report `status="blocked"` when `--require-real-data` is passed)
- **THEN** no fixture score SHALL ever be labelled `dataset_source="real"`

### Requirement: Per-arm learning-rate fairness and bootstrap statistical rigor
Before locking multi-seed track comparisons, the tournament SHALL execute a short per-candidate learning-rate micro-sweep (`{3e-4, 1e-3, 3e-3}`) at matched tokens so no candidate loses due to a mis-scaled default learning rate. All Stage-2/3 comparisons SHALL run at least 3 seeds and Stage-4 finality checks at least 5 seeds, reporting paired bootstrap 95% confidence intervals (10,000 resamples).

#### Scenario: Candidate victory requires statistical significance
- **WHEN** the tournament compares Candidate $X$ against Baseline/Candidate $Y$ on a primary metric
- **THEN** $X$ SHALL be marked statistically superior only when the paired bootstrap 95% confidence interval of their difference excludes zero and meets the pre-registered minimum practical effect threshold
- **THEN** ties within confidence intervals SHALL be broken deterministically in favor of fewer parameters and lower CPU latency/RSS

### Requirement: Four-stage elimination, interference, and leave-one-out pruning funnel
The tournament runner SHALL execute four sequential, auditable stages to extract a single winning architecture:
- **Stage 1 (Micro Falsification Gates):** eliminate any primary sequence core scoring `< 75%` on 16-pair MQAR (provided the Transformer upper bound exceeds `95%`) and drop any auxiliary training objective failing to improve held-out BPB by `≥ 0.015` (or downstream Macro-F1 by `≥ 1.0` point) at equal wall-clock;
- **Stage 2 (Isolated Track Tournaments):** select single winners (`Core*`, `Tokenizer*`, `Objective*`, `Decision*`, `Router*`, `Memory*`) within their isolated tracks under both Equal-Byte and Equal-Wall-Clock tables;
- **Stage 3 (Shared-Core vs. Split-Specialist Interference Gate):** select the **Unified Shared Core** if held-out LM BPB degrades by `≤ 3.0%` and Decision Macro-F1 drops by `≤ 2.0` points relative to single-task / Split-Specialist (`D4`) models at lower or equal RSS; automatically select the **Split Dual-Pathway** topology if LM BPB degrades by `> 5.0%` or Decision loses `> 5.0` Macro-F1 points;
- **Stage 4 (Earn-Your-Place Assembly & Leave-One-Out Pruning):** greedily add surviving optional enhancements to the Stage-3 winner only when 5-seed bootstrap 95% CIs prove non-regressing net gain within edge RAM budgets, followed by a Leave-One-Out ablation pass that permanently strips any component whose removal causes no statistically significant metric loss.

#### Scenario: Stage 1 eliminates non-recalling cores automatically
- **WHEN** a candidate core scores below `0.75` exact match on `MQAR-16` while `C6` (`softmax_transformer`) exceeds `0.95`
- **THEN** Stage 1 SHALL record the candidate as `eliminated` with reason `failed_mqar16_gate` and exclude it from primary backbone promotion in Stage 2

#### Scenario: Stage 3 enforces the shared-core interference rejection rule
- **WHEN** joint multi-task training on the winning core degrades held-out BPB by more than `5.0%` or reduces Macro-F1 by more than `5.0` points compared to `D4` (`Split-Specialist`)
- **THEN** Stage 3 SHALL reject the single-shared-core hypothesis, select the Split Dual-Pathway topology, and record the measured interference delta in the tournament ledger

#### Scenario: Stage 4 prunes redundant components via Leave-One-Out
- **WHEN** removing an added component from the assembled candidate produces a 5-seed paired bootstrap 95% CI that includes zero degradation across primary metrics
- **THEN** Stage 4 SHALL prune that component from the final selected architecture and log it under `pruned_components`
