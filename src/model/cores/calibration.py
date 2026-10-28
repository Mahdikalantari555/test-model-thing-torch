"""Post-hoc temperature calibration (`T_cal`), adaptive 15-bin ECE, Brier, NLL, and AURC."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CalibrationReport:
    temperature: float
    nll: float
    brier: float
    adaptive_ece: float
    aurc: float
    accuracy: float
    risk_at_80_coverage: float
    num_examples: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "temperature": self.temperature,
            "nll": self.nll,
            "brier": self.brier,
            "adaptive_ece": self.adaptive_ece,
            "aurc": self.aurc,
            "accuracy": self.accuracy,
            "risk_at_80_coverage": self.risk_at_80_coverage,
            "num_examples": self.num_examples,
        }


def assert_disjoint_splits(
    val_cal_ids: Sequence[Any],
    test_eval_ids: Sequence[Any],
) -> None:
    """Enforce zero sample overlap between `val_cal` (temperature fitting) and `test_eval`."""
    overlap = set(val_cal_ids).intersection(set(test_eval_ids))
    if overlap:
        raise ValueError(
            f"Data contamination detected between val_cal and test_eval ({len(overlap)} overlapping IDs)"
        )


class TemperatureCalibrator:
    """Fits scalar post-hoc temperature `T_cal > 0` strictly on held-out `val_cal` logits."""

    def __init__(self, init_temperature: float = 1.0, min_temp: float = 0.05, max_temp: float = 20.0) -> None:
        self.temperature = float(init_temperature)
        self.min_temp = float(min_temp)
        self.max_temp = float(max_temp)
        self.fitted_split_ids: Optional[frozenset[Any]] = None

    def fit(
        self,
        val_logits: torch.Tensor,
        val_labels: torch.Tensor,
        val_ids: Optional[Sequence[Any]] = None,
        max_iter: int = 100,
    ) -> float:
        if val_logits.dim() != 2:
            raise ValueError(f"val_logits must be 2-D (N, C), got {tuple(val_logits.shape)}")
        if val_logits.shape[0] == 0:
            raise ValueError("val_logits cannot be empty")
        if val_ids is not None:
            self.fitted_split_ids = frozenset(val_ids)

        logits = val_logits.detach().float()
        labels = val_labels.detach().long()
        log_t = torch.nn.Parameter(torch.tensor(math.log(self.temperature), dtype=torch.float32))
        optimizer = torch.optim.LBFGS([log_t], lr=0.25, max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            t = log_t.exp().clamp(self.min_temp, self.max_temp)
            loss = F.cross_entropy(logits / t, labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature = float(log_t.detach().exp().clamp(self.min_temp, self.max_temp).item())
        return self.temperature

    def scale_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def calibrate_probs(self, logits: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.scale_logits(logits), dim=-1)

    def evaluate(
        self,
        test_logits: torch.Tensor,
        test_labels: torch.Tensor,
        test_ids: Optional[Sequence[Any]] = None,
        n_bins: int = 15,
    ) -> CalibrationReport:
        if test_ids is not None and self.fitted_split_ids is not None:
            assert_disjoint_splits(list(self.fitted_split_ids), test_ids)
        return compute_calibration_metrics(
            logits=test_logits,
            labels=test_labels,
            temperature=self.temperature,
            n_bins=n_bins,
        )


def compute_adaptive_ece(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
) -> float:
    """15-bin equal-mass (adaptive quantile) Expected Calibration Error."""
    confidences, predictions = torch.max(probs.detach().float(), dim=-1)
    accuracies = (predictions == labels.detach().long()).float()
    n = confidences.numel()
    if n == 0:
        return 0.0

    order = torch.argsort(confidences)
    conf_sorted = confidences[order].cpu().numpy()
    acc_sorted = accuracies[order].cpu().numpy()

    chunks = np.array_split(np.arange(n), min(int(n_bins), n))
    ece = 0.0
    for idx_arr in chunks:
        if idx_arr.size == 0:
            continue
        bin_conf = float(conf_sorted[idx_arr].mean())
        bin_acc = float(acc_sorted[idx_arr].mean())
        ece += (idx_arr.size / float(n)) * abs(bin_acc - bin_conf)
    return float(ece)


def compute_risk_coverage_curve(
    probs: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[List[float], List[float], float]:
    """Compute Risk-Coverage curve sorted by descending confidence and integrate AURC."""
    confidences, predictions = torch.max(probs.detach().float(), dim=-1)
    errors = (predictions != labels.detach().long()).float()
    n = confidences.numel()
    if n == 0:
        return [], [], 0.0

    order = torch.argsort(confidences, descending=True)
    err_sorted = errors[order].cpu().numpy()
    cum_errors = np.cumsum(err_sorted)
    counts = np.arange(1, n + 1, dtype=np.float64)
    coverages = counts / float(n)
    risks = cum_errors / counts

    if n == 1:
        aurc = float(risks[0])
    else:
        # Left-anchored Riemann/trapezoidal area over coverage in [1/N .. 1.0]
        trap = float(np.trapezoid(risks, coverages))
        aurc = trap + float(risks[0]) * float(coverages[0])
    return coverages.tolist(), risks.tolist(), float(aurc)


def select_abstention_threshold(
    val_probs: torch.Tensor,
    val_labels: torch.Tensor,
    target_max_risk: float = 0.15,
    fallback_threshold: float = 0.55,
) -> float:
    """Pick the lowest confidence threshold `tau` on `val_cal` achieving risk <= `target_max_risk`."""
    confidences, predictions = torch.max(val_probs.detach().float(), dim=-1)
    errors = (predictions != val_labels.detach().long()).float()
    n = confidences.numel()
    if n == 0:
        return float(fallback_threshold)

    order = torch.argsort(confidences, descending=True)
    conf_sorted = confidences[order].cpu().numpy()
    err_sorted = errors[order].cpu().numpy()
    cum_risk = np.cumsum(err_sorted) / np.arange(1, n + 1, dtype=np.float64)

    feasible = np.where(cum_risk <= float(target_max_risk))[0]
    if feasible.size == 0:
        return float(fallback_threshold)
    best_k = int(feasible[-1])  # maximum coverage meeting target risk
    return float(conf_sorted[best_k])


def compute_calibration_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
    n_bins: int = 15,
) -> CalibrationReport:
    t = max(1e-4, float(temperature))
    scaled = logits.detach().float() / t
    y = labels.detach().long()
    n, num_classes = scaled.shape
    probs = F.softmax(scaled, dim=-1)

    nll = float(F.cross_entropy(scaled, y).item())
    y_onehot = F.one_hot(y, num_classes=num_classes).float()
    brier = float(((probs - y_onehot) ** 2).sum(dim=-1).mean().item())
    ece = compute_adaptive_ece(probs, y, n_bins=n_bins)
    coverages, risks, aurc = compute_risk_coverage_curve(probs, y)
    acc = float((probs.argmax(dim=-1) == y).float().mean().item())

    # Risk at 80% coverage
    idx_80 = min(max(0, int(round(0.8 * n)) - 1), n - 1)
    risk_80 = float(risks[idx_80]) if risks else 0.0

    return CalibrationReport(
        temperature=t,
        nll=nll,
        brier=brier,
        adaptive_ece=ece,
        aurc=aurc,
        accuracy=acc,
        risk_at_80_coverage=risk_80,
        num_examples=n,
    )
