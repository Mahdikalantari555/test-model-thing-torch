# RecaMem Core Architecture Selection Report

- **Selection Status**: `verified_winner`
- **Winning Stack**: `C4` + `Tok_bpe_4096` + `O1` (`chunked_bptt`) + `D3` + `R4` + `M4`
- **Backbone Sharing Mode**: `unified` (`summed` multi-task schedule)
- **Dataset Provenance**: `fixture`
- **Tournament Tier**: `custom` (`349,920` trainable params; target production tier `17,000,928` params within `±2%` of `17.0M`)

## 1. Track 1 — Sequence-Mixing Core Comparison (`C0`–`C6`)

| Core ID | Complexity Class | Params | MQAR-4 | MQAR-16 | MQAR-32 | MQAR-64 | Copy Acc | Held-Out BPB | Grammar Acc | State @ 4K (KB) | Stage-1 Gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `C0` | `constant_vector` | 350,272 | 0.125 | 0.070 | 0.172 | 0.062 | 0.038 | 6.820 | 0.083 | 1.0 | REJECT (mqar_16_acc=0.070 < 0.75) |
| `C1` | `constant_vector` | 350,256 | 0.234 | 0.086 | 0.070 | 0.078 | 0.135 | 6.790 | 0.028 | 0.8 | REJECT (mqar_16_acc=0.086 < 0.75) |
| `C2` | `constant_vector` | 349,872 | 0.336 | 0.148 | 0.195 | 0.062 | 0.038 | 6.863 | 0.083 | 14.2 | REJECT (mqar_16_acc=0.148 < 0.75) |
| `C3` | `constant_matrix` | 349,896 | 0.336 | 0.151 | 0.174 | 0.112 | 0.141 | 6.905 | 0.111 | 6.0 | REJECT (mqar_16_acc=0.151 < 0.75) |
| `C4` | `constant_matrix` | 349,920 | 1.000 | 1.000 | 0.966 | 0.896 | 0.142 | 6.845 | 0.083 | 12.8 | PASS |
| `C5` | `sliding_window_kv` | 349,872 | 0.154 | 0.143 | 0.159 | 0.068 | 0.163 | 6.781 | 0.083 | 195.0 | REJECT (mqar_16_acc=0.143 < 0.75) |
| `C6` | `linear_kv` | 349,872 | 0.247 | 0.128 | 0.146 | 0.096 | 0.340 | 6.787 | 0.028 | 6144.0 | REFERENCE |

**Track-1 Winner**: `C4` (Pareto frontier: `['C4']`). 10,000-resample paired bootstrap vs `C3` on MQAR-16: `Δ = +0.859`, 95% CI `[+0.805, +0.914]`, `p = 0.0000`.

## 2. Track 2 — Auxiliary Objectives (`O1`–`O4`) & Tokenizer Budgets

### 2.1 Objectives & Online RTRL (`O1`–`O4`)

| Arm ID | Held-Out BPB | Probe Macro-F1 | Train Tok/s | Stage-1 Gate Verdict |
| --- | ---: | ---: | ---: | --- |
| `O1` | 6.7794 | 0.204 | 4923.5 | PASS |
| `O2` | 6.7727 | 0.204 | 4880.2 | REJECT (delta_bpb=+0.0067 < 0.015 and delta_f1=+0.000 < 0.020) |
| `O3` | 6.7794 | 0.204 | 4702.1 | REJECT (delta_bpb=-0.0000 < 0.015 and delta_f1=+0.000 < 0.020) |
| `O4_online_rtrl` | 6.7983 | 0.000 | 4205.7 | REJECT (rtrl_tps=4205.7 < chunked_bptt_tps=9012.0; incompatible_with_matrix_fast_weight_cores) |

### 2.2 Paired Tokenizer Tables (`V=256` UTF-8 Bytes vs `V=4096` Compact BPE)

| Arm ID | Vocab | Equal-Byte BPB | Equal-Wall-Clock BPB | Decode Bytes/s | Lossless UTF-8 |
| --- | ---: | ---: | ---: | ---: | --- |
| `Tok_bytes_256` | 256 | 6.2139 | 5.0973 | 177.4 | `True` |
| `Tok_bpe_4096` | 4096 | 6.7794 | 6.7794 | 1661.0 | `True` |

## 3. Track 3 — System-1 Decision Readouts & Calibration (`D1`–`D4`)

| Readout ID | $T_{\text{cal}}$ | In-Domain Macro-F1 | Shifted/Unseen Macro-F1 | OOS Null F1 | Adaptive ECE | AURC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `D1` | 3.457 | 0.496 | 0.168 | 0.453 | 0.4310 | 0.3768 |
| `D2` | 7.083 | 0.204 | 0.168 | 0.453 | 0.3452 | 0.5156 |
| `D3` | 1.418 | 0.639 | 0.184 | 0.722 | 0.3342 | 0.4366 |
| `D4` | 6.574 | 0.471 | 0.328 | 0.646 | 0.3990 | 0.3651 |

## 4. Track 4 — 5-Way Action Routers & Safety Guards (`R1`–`R4`)

| Router ID | Normalized Utility | Accuracy | False Tool Rate | Unblocked Invalid Tool | Adaptive ECE | p50 Latency (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `R1` | 0.821 | 0.857 | 0.000 | 0.000 | 0.5009 | 0.211 |
| `R2` | 0.345 | 0.476 | 0.000 | 0.000 | 0.4592 | 0.218 |
| `R3` | 0.405 | 0.667 | 0.000 | 0.000 | 0.3271 | 0.071 |
| `R4` | 0.762 | 0.810 | 0.000 | 0.000 | 0.1122 | 0.271 |

## 5. Track 5 — Lifelong Memory & Contradiction Adaptation (`M0`–`M5`)

| Memory Arm | 5-Domain Avg Retention | Backward Forgetting | Contradiction Update Acc | Stale Answer Rate | Static + Dynamic Bytes | Stage-1 Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `M0` | 0.000 | 0.000 | 0.000 | 0.000 | 13,056 | REJECT (retention_acc=0.00 < 0.40 or contradiction_acc=0.00 < 0.50) |
| `M1` | 1.000 | 0.000 | 1.000 | 0.000 | 1,499,290 | PASS |
| `M2` | 1.000 | 0.000 | 1.000 | 0.000 | 1,401,558 | PASS |
| `M3` | 1.000 | 0.000 | 1.000 | 0.000 | 1,400,594 | PASS |
| `M4` | 1.000 | 0.000 | 1.000 | 0.000 | 1,221,986 | PASS |
| `M5` | 0.000 | 0.000 | 0.000 | 0.000 | 2,799,360 | REJECT (retention_acc=0.00 < 0.40 or contradiction_acc=0.00 < 0.50) |

## 6. Stage 3 — Shared-Backbone vs Equal-Budget Split-Specialist (`D4`) Interference Verdict

- **Selected Mode**: `unified` (`summed`)
- **Verdict**: Unified shared backbone passed interference gate: rel_bpb_deg=-4.18% <= +3.0%, f1_drop=-2.21 pts <= 2.0 pts, composite=0.3408 vs split=0.3214
- **Parameter Parity**: relative difference `0.07%` (`±2.0%` budget rule enforced)

| Regime | Held-Out BPB | Decision Macro-F1 |
| --- | ---: | ---: |
| Standalone LM / Standalone Decision | 6.8452 | 0.412 |
| Unified Shared (`interleaved 0.80/0.15/0.05`) | 6.5265 | 0.271 |
| Unified Shared (`summed` loss ablation) | 6.5594 | 0.434 |
| Equal-Budget Split-Specialist (`D4`) | 6.7634 | 0.399 |

## 7. Stage 4 — Leave-One-Out (`LOO`) Complexity Pruning Waterfall

| Complexity Component | Simpler Fallback | Utility w/ Component | Utility w/o Component | Marginal Gain | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| `short_conv4` | `conv_off` | 0.5240 | 0.2072 | +0.3168 | **KEPT** |
| `ema_jepa_vicreg_aux_loss` | `O1_ce_eos` | 0.5240 | 0.5205 | +0.0035 | **PRUNED** |
| `interleaved_multitask_schedule` | `summed_multitask_loss` | 0.5240 | 0.5643 | -0.0438 | **PRUNED** |

## 8. Final Canonical RecaMem Core Specification

- **Manifest Path**: `configs/recamem_selected_core.json`
- **Evidence Ledger**: `reports/evidence/recamem-core-tournament.json`
- **Runtime Loader**: `from src.model.recamem_core import build_recamem_core`
- **Target ~17M Production Shape**: `d_model=384`, `n_layers=8`, `n_heads=6`, `d_ff=1025`, `vocab_size=4096` (`17,000,928` trainable parameters)
- **Calibrated Thresholds**: `decision_temperature=1.418`, `router_temperature=0.234`, `router_abstain_threshold=0.368`
