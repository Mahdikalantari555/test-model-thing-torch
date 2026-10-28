"""Statistical comparison primitives: LR micro-sweep, 10k paired bootstrap CIs, and Pareto frontier."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

DEFAULT_LR_GRID: Tuple[float, ...] = (3e-4, 1e-3, 3e-3)


@dataclass(frozen=True)
class BootstrapCI:
    metric: str
    arm_a: str
    arm_b: str
    mean_a: float
    mean_b: float
    mean_diff: float  # A - B
    ci_low: float
    ci_high: float
    p_value_two_sided: float
    significant_95: bool
    num_examples: int
    num_resamples: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def paired_bootstrap_ci(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    *,
    arm_a: str = "arm_a",
    arm_b: str = "arm_b",
    metric: str = "score",
    num_resamples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 1337,
) -> BootstrapCI:
    """10,000-resample paired bootstrap confidence interval for `mean(A - B)`."""
    arr_a = np.asarray(scores_a, dtype=np.float64)
    arr_b = np.asarray(scores_b, dtype=np.float64)
    if arr_a.shape != arr_b.shape or arr_a.ndim != 1 or arr_a.size == 0:
        raise ValueError(
            f"Paired bootstrap requires equal non-empty 1-D arrays, got {arr_a.shape} vs {arr_b.shape}"
        )
    diffs = arr_a - arr_b
    n = diffs.size
    mean_a = float(arr_a.mean())
    mean_b = float(arr_b.mean())
    mean_diff = float(diffs.mean())

    if np.allclose(diffs, 0.0):
        return BootstrapCI(
            metric=metric,
            arm_a=arm_a,
            arm_b=arm_b,
            mean_a=mean_a,
            mean_b=mean_b,
            mean_diff=0.0,
            ci_low=0.0,
            ci_high=0.0,
            p_value_two_sided=1.0,
            significant_95=False,
            num_examples=int(n),
            num_resamples=int(num_resamples),
        )

    rng = np.random.default_rng(int(seed))
    # Chunked resampling keeps peak memory tiny (< 8 MB) even at 10,000 resamples
    boot_means = np.empty(int(num_resamples), dtype=np.float64)
    chunk = 1000
    for start in range(0, int(num_resamples), chunk):
        stop = min(start + chunk, int(num_resamples))
        idx = rng.integers(0, n, size=(stop - start, n))
        boot_means[start:stop] = diffs[idx].mean(axis=1)

    alpha = 1.0 - float(confidence_level)
    ci_low = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    ci_high = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))

    le_zero = float(np.mean(boot_means <= 0.0))
    ge_zero = float(np.mean(boot_means >= 0.0))
    p_two = min(1.0, 2.0 * min(le_zero, ge_zero))
    significant = bool(ci_low > 0.0 or ci_high < 0.0)

    return BootstrapCI(
        metric=metric,
        arm_a=arm_a,
        arm_b=arm_b,
        mean_a=mean_a,
        mean_b=mean_b,
        mean_diff=mean_diff,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value_two_sided=p_two,
        significant_95=significant,
        num_examples=int(n),
        num_resamples=int(num_resamples),
    )


def select_best_lr(
    evaluate_lr_on_val: Callable[[float], Dict[str, Any]],
    *,
    lr_grid: Sequence[float] = DEFAULT_LR_GRID,
    metric_key: str = "val_loss",
    higher_is_better: bool = False,
) -> Dict[str, Any]:
    """Run a 3-point LR micro-sweep strictly on `val_cal`, rejecting NaN/Inf runs."""
    sweep_log: List[Dict[str, Any]] = []
    best_lr: Optional[float] = None
    best_val: Optional[float] = None
    best_payload: Dict[str, Any] = {}

    for lr in lr_grid:
        res = dict(evaluate_lr_on_val(float(lr)))
        raw_val = res.get(metric_key)
        finite = raw_val is not None and math.isfinite(float(raw_val))
        sweep_log.append({"lr": float(lr), metric_key: raw_val, "finite": finite})
        if not finite:
            continue
        val = float(raw_val)
        if best_val is None or (val > best_val if higher_is_better else val < best_val):
            best_val = val
            best_lr = float(lr)
            best_payload = res

    if best_lr is None:
        raise RuntimeError(f"All candidate learning rates in {list(lr_grid)} diverged on val_cal")

    return {
        "selected_lr": best_lr,
        "selected_metric": best_val,
        "metric_key": metric_key,
        "sweep": sweep_log,
        "best_run": best_payload,
    }


def compute_pareto_frontier(
    records: Sequence[Mapping[str, Any]],
    objectives: Mapping[str, str],  # metric_name -> "max" | "min"
    id_key: str = "arm_id",
) -> List[str]:
    """Return IDs of Pareto-optimal (non-dominated) candidate arms."""
    items = list(records)
    frontier: List[str] = []
    for i, cand in enumerate(items):
        dominated = False
        for j, other in enumerate(items):
            if i == j:
                continue
            better_or_equal_all = True
            strictly_better_one = False
            for m_key, direction in objectives.items():
                vc = float(cand[m_key])
                vo = float(other[m_key])
                if direction == "max":
                    if vo < vc - 1e-12:
                        better_or_equal_all = False
                        break
                    if vo > vc + 1e-12:
                        strictly_better_one = True
                elif direction == "min":
                    if vo > vc + 1e-12:
                        better_or_equal_all = False
                        break
                    if vo < vc - 1e-12:
                        strictly_better_one = True
                else:
                    raise ValueError(f"Invalid objective direction {direction!r}")
            if better_or_equal_all and strictly_better_one:
                dominated = True
                break
        if not dominated:
            frontier.append(str(cand[id_key]))
    return frontier
