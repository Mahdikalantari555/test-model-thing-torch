# Spec Delta

## ADDED Requirements

### Requirement: Option-conditioned scoring over dynamic closed option sets with null option
In addition to static cosine prototype routing (`D1`) and closed-form Ridge probes (`D2`), the system SHALL support scoring arbitrary per-request closed option sets ($K \in [2, 255]$) supplied as natural-language option descriptions (`D3`), encoding the query via attention pooling and each option via the shared sequence core while appending a learned null/out-of-scope option vector $v_0$ to represent "none of the supplied options apply".

#### Scenario: Variable option set size and order invariance
- **WHEN** a query is evaluated against $K$ dynamic textual options in two different permutations
- **THEN** the option-conditioned head SHALL produce identical per-option probabilities (up to permutation and `1e-5` floating-point tolerance) in a single non-autoregressive pass without token decoding
- **WHEN** none of the supplied options match an out-of-scope query
- **THEN** the learned null option $v_0$ SHALL be selectable and map deterministically to abstention / out-of-scope rejection

#### Scenario: Generalization fallback on unseen option label phrasings
- **WHEN** evaluation on held-out unseen option phrasings drops near chance relative to few-shot prototypes
- **THEN** the decision head SHALL support hybrid prototype-anchored option representations (combining option text encoding with few-shot support states) and report which mode ran

### Requirement: Post-hoc temperature calibration and selective risk-coverage metrics
The system SHALL fit post-hoc scalar temperature scaling ($T_{\text{cal}} > 0$) on a held-out calibration split separate from model-selection and test splits, and SHALL evaluate probabilistic calibration separately from accuracy using Negative Log-Likelihood (`NLL`), Brier score, 15-bin equal-mass Adaptive Expected Calibration Error (`adaptive_ece`), and Area Under the Risk-Coverage Curve (`AURC`).

#### Scenario: Temperature scaling preserves argmax while improving calibration
- **WHEN** post-hoc temperature $T_{\text{cal}}$ is fit on `val_cal` and applied to `test_eval` logits
- **THEN** top-1 option argmax predictions SHALL remain unchanged
- **THEN** the report SHALL record pre-scaling and post-scaling `NLL`, `Brier`, `adaptive_ece` (with bin counts), and `AURC` on both in-domain and out-of-distribution splits
