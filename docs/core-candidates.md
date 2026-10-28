# Candidate Sequence-Mixing Cores (`C0`–`C6`)

All seven Track-1 sequence backbones implement `src.model.cores.contracts.SequenceCore` and plug into `CandidateLM` without altering tokenization, loss computation, System-1 decision readouts, routing, or memory modules.

## Summary Matrix

| ID | Slug | State Update Formula | State Complexity | Online RTRL |
| --- | --- | --- | --- | --- |
| `C0` | `unstacked_rtu` | $s_t^{(l)} = \sigma(\delta_l) \odot s_{t-1}^{(l)} + e_t$ (all layers read raw $e_t$) | `constant_vector` $O(L d)$ | Yes |
| `C1` | `stacked_gated_rtu` | $s_t^{(l)} = g_t \odot s_{t-1}^{(l)} + (1 - g_t) \odot (W_{\text{in}} u_t^{(l-1)})$ | `constant_vector` $O(L d)$ | Yes |
| `C2` | `selective_ssm` | $h_t = \exp(-\Delta_t A) \odot h_{t-1} + \Delta_t B_t x_t$, $y_t = C_t^\top h_t + D x_t$ | `constant_vector` $O(L d N)$ | No |
| `C3` | `retnet_decay_matrix` | $S_t = \gamma_h S_{t-1} + v_t k_t^\top$, $o_t = S_t q_t$ | `constant_matrix` $O(L d^2 / H)$ | No |
| `C4` | `gated_delta_matrix` | $S_t = \alpha_t S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top$, $o_t = S_t q_t$ | `constant_matrix` $O(L d^2 / H)$ | No |
| `C5` | `needle_san` | Sliding-window GQA ($W=256$) + QK-RMSNorm + 4 pinned sink tokens + Hadamard MLP | `sliding_window_kv` $O(L (W + S) d_{\text{kv}})$ | No |
| `C6` | `softmax_transformer` | Full causal MHA + RoPE + QK-RMSNorm + SwiGLU | `linear_kv` $O(L T d)$ | No |

## Parameter Budget Matching

`src/model/cores/param_matcher.py` solves `(d_model, n_layers, n_heads, d_ff)` for every core at shared `vocab_size` so that all seven backbones stay within `±2.0%` of the requested tier (`micro = 2M`, `small = 6M`, `target = 17M`).

```bash
PYTHONPATH=. python -c "
from src.model.cores.param_matcher import solve_all_cores_for_tier
from src.model.cores.registry import build_candidate_lm
for cid, cfg in solve_all_cores_for_tier('small', vocab_size=4096).items():
    lm = build_candidate_lm(cfg)
    actual = sum(p.numel() for p in lm.parameters() if p.requires_grad)
    print(f'{cid}: actual={actual:,} target={cfg.target_params:,} err={abs(actual-cfg.target_params)/cfg.target_params:.2%}')
"
```
