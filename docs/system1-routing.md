# System-1 Decision Readouts (`D1`–`D4`), Calibration, and Action Routers (`R1`–`R4`)

## Decision Readout Arms (Track 3)

| ID | Module | Description | Dynamic $K \in [2, 255]$ | Null/OOS Support |
| --- | --- | --- | --- | --- |
| `D1` | `ZeroShotPrototypeReadout` | Cosine similarity over mean-pooled query & option states (`DecisionHead` baseline) | Yes | Cosine threshold |
| `D2` | `ClosedFormRidgeReadout` | Closed-form Ridge regression $(X^\top X + \lambda I)^{-1} X^\top Y_{\text{opt}}$ | Yes | Cosine threshold |
| `D3` | `OptionConditionedReadout` | Attention-pooled query + hybrid core-encoded option vectors + learned null $v_0$ | Yes (permutation-invariant) | Learned $v_0, b_0$ |
| `D4` | `SplitSpecialistSystem` | Equal-budget (`±2.0%`) split LM + dedicated specialist encoder + `D3` head | Yes | Learned $v_0, b_0$ |

## Calibration Protocol (`src/model/cores/calibration.py`)

- Post-hoc scalar temperature $T_{\text{cal}} > 0$ is fitted strictly on `val_cal` using L-BFGS and never touches `test_eval` (`assert_disjoint_splits` raises `ValueError` on ID overlap).
- Reported metrics on `test_eval` (both `in_domain` and `shifted_unseen` slices): `NLL`, multi-class `Brier`, 15-bin equal-mass `adaptive_ece`, Risk-Coverage curves, and `AURC`.

## 5-Way Action Routers & Deterministic Guards (Track 4)

All four routers (`R1`–`R4`) map `(query, hidden_states)` into `("generate", "classify", "retrieve", "tool", "abstain")` and pass through two deterministic safety guards before execution:

1. **Risk-Coverage Abstention Guard**: If calibrated confidence $p_{\max} < \tau_{\text{abstain}}$ (selected on `val_cal`), the router overrides the action to `"abstain"` with `guard_triggered=True`.
2. **Tool Schema & Permission Guard (`validate_tool_call_guard`)**: Any `"tool"` route with missing required arguments, type mismatches, unknown tools, or unpermitted tool names is deterministically blocked and downgraded to `"abstain"`.
