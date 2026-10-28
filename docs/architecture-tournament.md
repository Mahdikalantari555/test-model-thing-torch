# RecaMem Core Architecture Tournament (`recamem-core-architecture-tournament`)

`scripts/run_core_tournament.py` executes the controlled 4-stage elimination, interference, and Leave-One-Out (`LOO`) pruning tournament across all 24+ candidate arms across five orthogonal design tracks (`C0`–`C6`, `O1`–`O4` + `Tok`, `D1`–`D4`, `R1`–`R4`, `M0`–`M5`).

## 1. Stage Gates & Evaluation Scale

1. **Stage 0 (Milestone M0)**: Verifies `±2.0%` parameter parity across all seven cores (`C0`–`C6`), lossless UTF-8 round-trips (`V=256` bytes and `V=4096` compact BPE), empirical byte-unigram and token-unigram BPB baselines, and zero 64-byte `train`/`val_cal`/`test_eval` split contamination.
2. **Stage 1 (Mechanistic Falsification)**:
   - Cores must solve `MQAR-4` and `MQAR-16` (`>= 0.75` on `1,024` queries/setting across `256` sequences) without NaN divergence; per-core MQAR LR selection (`{1e-3, 3e-3}`) prevents premature rejection of `C6` softmax attention.
   - Auxiliary objectives (`O2`, `O3` VICReg + EMA-JEPA on un-normalized core hidden states) record nonzero `mean_aux_loss` and `aux_grad_norm`, and must beat pure `O1` (`ce_eos`) by `ΔBPB >= 0.015` (or `+2.0` pts shifted-probe Macro-F1 without BPB loss).
   - `O4` (`online_rtrl`) is evaluated on both held-out BPB and `D2` shifted-probe Macro-F1 and must match `chunked_bptt` while supporting the winning core class.
   - Memory arms (`M0`–`M5`) are evaluated on `1,000` propositions (`5` domains × `200` facts) + `250` interleaved distractors + `200` contradiction updates using paraphrased natural-language queries without oracle `entity_key` metadata leakage (`ret_acc >= 0.40` and `contra_acc >= 0.50`).
3. **Stage 2 (Multi-Seed Per-Track Ranking)**:
   - **Track 2b Tokenizers (`Tok_bytes_256` vs `Tok_bpe_4096`)**: Matched at **equal non-embedding parameters** (`solve_core_config_non_embedding`) across **four independent runs** (two equal-UTF8-byte runs and two equal-wall-clock runs — never reusing a run).
   - **Track 3 Decisions (`D1`–`D4`)**: Evaluated on `1,100` held-out items (`500` in-domain + `600` unseen shifted + OOS) with calibrated null thresholds on `D1`/`D2`; ranked by `shifted_macro_f1`, `oos_f1`, `aurc`, and `adaptive_ece` (never in-domain F1).
   - **Track 4 Routers (`R1`–`R4`)**: Evaluated on `1,000` held-out items; ranked by cost-weighted `normalized_utility`, `false_tool_rate`, `unblocked_invalid_tool_rate`, and `adaptive_ece`.
4. **Stage 3 (Shared-vs-Split Interference Gate)**: Compares unified multi-task training (`interleaved 0.80/0.15/0.05` vs `summed` loss) against standalone models and Equal-Budget `D4` (`SplitSpecialistSystem`, `±2.0%` total parameter parity).
5. **Stage 4 (Leave-One-Out Pruning)**: Removes any optional complexity component whose marginal utility contribution is `< 0.005` (`0.5%`).

## 2. Provenance & Provisional Core Status Rule

`src/model/recamem_core.py` enforces strict honesty between trained tournament tiers and solved production shapes:
- Whenever `target_17m_trained` is `false` (e.g., when `0.35M` and `3.0M` tiers are trained on CPU while the `17.0M` configuration is architecturally solved but untrained), `selection_status` **must** be `"provisional_core"` (`validate_selected_core_manifest` raises `ValueError` if `"verified_winner"` is claimed with `target_17m_trained=False`).
- Every report and evidence ledger stamps exact `dataset_source`, corpus SHA-256 provenance, unigram BPB baselines, trained parameter counts, and held-out evaluation counts (`N`).

## 3. CLI Usage

```bash
# Dry-run parameter matching & split verification on real TinyStories (>=1,000 eval items)
PYTHONPATH=. python scripts/run_core_tournament.py --dry-run --use-tinystories --scale large --budget 350000

# Full real-corpus tournament at 0.35M + 3.0M scale-up across C4, C3, and C6 (resumable per core)
PYTHONPATH=. python scripts/run_core_tournament.py \
  --use-tinystories --scale large --budget 350000 \
  --seeds 1337,2026,4242 --mqar-steps 600 --copy-steps 120 --lm-steps 160 \
  --include-scaleup --scaleup-budget 3000000 --scaleup-lm-steps 240 --resume
```

## 4. Canonical RecaMem Core Integration

Downstream `recamem` code (`plans/09_recamem_plan.md`) loads the selected architecture (`provisional_core` or `verified_winner`) through a single factory call:

```python
from src.model.recamem_core import build_recamem_core

bundle = build_recamem_core("configs/recamem_selected_core.json")
bundle.teach("Sentinel-2 revisit period is 5 days", metadata={"entity_key": "sentinel2 revisit"})
recalled = bundle.recall("sentinel2 revisit")
decision = bundle.decide("Adjust perigee altitude", ["Orbital maneuver burn", "Mask clouds"])
routing = bundle.route("Compute 144 * 29 with calculator")
reply = bundle.generate("Summarize orbital radiometry:", max_new_tokens=24)
```

## 7. Hardware Throughput & Wall-Clock Scaling Reality Check

When extrapolating CPU training and inference budgets from `0.35M` and `3.0M` runs to the `17.0M` production target on a 2-vCPU host:
- **Effective Training Compute (`GFLOP/s`)**: At `N ≈ 3.5 × 10^5` parameters and `~2,400–11,600 tok/s` (`6 · N · tok/s`), effective CPU throughput is **`~5–25 GFLOP/s`** (rising to **`~50–70 GFLOP/s`** on `3.0M`-parameter dense matrix chunks where BLAS matmuls amortize Python dispatch, still far below theoretical AVX-512 peak).
- **Realistic `17M` × `30M`-Token Training Wall-Clock**: Training a `17M`-parameter model on `30M` tokens requires `6 × 1.7 × 10^7 × 3.0 × 10^7 ≈ 3.06 × 10^15` FLOPs. At realistic sustained `10–50 GFLOP/s` on 2 CPU vCPUs, a single `30M`-token run takes **`17 hours` (`50 GFLOP/s`) to `3.5 days` (`10 GFLOP/s`)** — multi-day CPU compute or a single GPU run is required before promoting `C4` from `provisional_core` to `verified_winner`.
- **Single-Stream Autoregressive Decode (`batch=1`)**: Eager Python `step()` loops at `~400–850 bytes/s` (`~100–250 tok/s`) are dominated by per-token PyTorch/Python kernel dispatch overhead (`~4–8 ms/step` across 4–8 layers), not FLOP capacity; production CPU deployment requires C++/ONNX/GGUF graph fusion to remove Python step overhead.
