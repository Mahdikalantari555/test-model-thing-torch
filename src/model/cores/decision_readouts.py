"""Track-3 System-1 decision readout arms (`D1`..`D4`) with dynamic option sets and calibrated `null`.

Arms:
- `D1` (`zero_shot_prototype`): zero-gradient centered/whitened pooled representation cosine readout + calibrated `null`.
- `D2` (`closed_form_ridge`): exact closed-form Cholesky `(X^T X + lambda I)^{-1} X^T Y` linear/pairwise ridge readout (zero SGD steps).
- `D3` (`option_conditioned`): parameter-light cross-option bilinear + interaction scoring head for arbitrary `K` + `null`.
- `D4` (`split_specialist`): equal-total-parameter baseline allocating budget across `(smaller CandidateLM + specialist encoder + D3 head)`.
"""

from __future__ import annotations

import time
from typing import Any, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.cores.contracts import CandidateLM, DecisionResult
from src.model.cores.param_matcher import TIER_BUDGETS, solve_core_config
from src.model.cores.registry import build_candidate_lm

NULL_OPTION = "null"


def _make_decision_result(
    *,
    names: List[str],
    logits_t: torch.Tensor,
    probs_t: torch.Tensor,
    t0: float,
    mode: str,
) -> DecisionResult:
    idx = int(probs_t.argmax().item())
    sel = names[idx]
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return DecisionResult(
        selected_index=idx,
        selected_option=sel,
        probabilities=[float(p) for p in probs_t.detach().cpu().tolist()],
        logits=[float(z) for z in logits_t.detach().cpu().tolist()],
        confidence=float(probs_t[idx].item()),
        is_null_or_abstain=(sel == NULL_OPTION),
        latency_ms=elapsed_ms,
        mode=mode,
        metadata={"option_names": list(names)},
    )


def _pool_hidden(h: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Mean-pool `(B, T, d)` or `(T, d)` sequence representations cleanly."""
    if h.dim() == 2:
        h = h.unsqueeze(0)
    if mask is None:
        return h.mean(dim=1)
    weights = mask.to(h.dtype).unsqueeze(-1)
    return (h * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)


def _optimal_null_threshold(max_non_null_scores: List[float], is_null_flags: List[bool]) -> Optional[float]:
    """Grid-search threshold `tau` on `val_cal` maximizing balanced F1 between `null` and non-`null`."""
    if not max_non_null_scores or len(max_non_null_scores) != len(is_null_flags):
        return None
    has_pos = any(is_null_flags)
    has_neg = any(not f for f in is_null_flags)
    if not (has_pos and has_neg):
        return None
    vals = sorted(set(round(float(x), 5) for x in max_non_null_scores))
    candidates = [vals[0] - 0.05] + [(vals[i] + vals[i + 1]) * 0.5 for i in range(len(vals) - 1)] + [vals[-1] + 0.05]
    best_tau = candidates[len(candidates) // 2]
    best_score = -1.0
    for tau in candidates:
        tp_n = fp_n = fn_n = tp_v = fp_v = fn_v = 0
        for s, is_n in zip(max_non_null_scores, is_null_flags):
            pred_n = s < tau
            if pred_n and is_n:
                tp_n += 1
            elif pred_n and not is_n:
                fp_n += 1
            elif (not pred_n) and is_n:
                fn_n += 1
            if (not pred_n) and (not is_n):
                tp_v += 1
            elif (not pred_n) and is_n:
                fp_v += 1
            elif pred_n and (not is_n):
                fn_v += 1
        f1_n = (2.0 * tp_n) / max(1.0, 2.0 * tp_n + fp_n + fn_n)
        f1_v = (2.0 * tp_v) / max(1.0, 2.0 * tp_v + fp_v + fn_v)
        macro = 0.5 * (f1_n + f1_v)
        if macro > best_score:
            best_score = macro
            best_tau = tau
    return float(best_tau)


class ZeroShotPrototypeReadout(nn.Module):
    """`D1`: zero-parameter centered/whitened cosine similarity against arbitrary option strings + `null`."""

    readout_id = "D1"

    def __init__(self, d_model: int, logit_scale: float = 8.0, null_threshold: float = 0.20) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.logit_scale = float(logit_scale)
        self.null_threshold = float(null_threshold)
        self.register_buffer("center", torch.zeros(self.d_model))
        self.register_buffer("inv_std", torch.ones(self.d_model))
        self.has_stats = False

    @torch.no_grad()
    def fit_corpus_stats(self, vectors: torch.Tensor) -> None:
        """Estimate non-parametric centering & whitening buffers on training representations (zero trainable params)."""
        if vectors.dim() != 2 or vectors.shape[0] < 2:
            return
        mu = vectors.mean(dim=0)
        std = torch.sqrt(vectors.var(dim=0, unbiased=False) + 1e-4)
        self.center.copy_(mu)
        self.inv_std.copy_(1.0 / std)
        self.has_stats = True

    def _whiten(self, v: torch.Tensor) -> torch.Tensor:
        if self.has_stats:
            v = (v - self.center.to(v.device, v.dtype)) * self.inv_std.to(v.device, v.dtype)
        return F.normalize(v, dim=-1)

    @torch.no_grad()
    def calibrate_null_threshold(
        self,
        val_queries: Sequence[torch.Tensor],
        val_options_list: Sequence[Sequence[torch.Tensor]],
        is_null_flags: Sequence[bool],
    ) -> float:
        max_sims: List[float] = []
        for q_h, opt_hs in zip(val_queries, val_options_list):
            q_n = self._whiten(_pool_hidden(q_h))[0]
            opt_n = self._whiten(torch.cat([_pool_hidden(o) for o in opt_hs], dim=0))
            sims = opt_n @ q_n
            max_sims.append(float(sims.max().item()))
        tau = _optimal_null_threshold(max_sims, list(is_null_flags))
        if tau is not None:
            self.null_threshold = tau
        return self.null_threshold

    def score_options(
        self,
        query_hidden: torch.Tensor,
        option_hiddens: Sequence[torch.Tensor],
        option_names: Sequence[str],
        include_null: bool = True,
        temperature: float = 1.0,
    ) -> DecisionResult:
        t0 = time.perf_counter()
        q_vec = self._whiten(_pool_hidden(query_hidden))  # (1, d)
        opt_vecs = self._whiten(torch.cat([_pool_hidden(o) for o in option_hiddens], dim=0))  # (K, d)
        sims = (q_vec @ opt_vecs.transpose(0, 1)).squeeze(0)  # (K,)
        names = list(option_names)
        if include_null:
            null_score = torch.tensor([self.null_threshold], device=sims.device, dtype=sims.dtype)
            raw_scores = torch.cat([sims, null_score], dim=0)
            names.append(NULL_OPTION)
        else:
            raw_scores = sims
        logits = (raw_scores * self.logit_scale) / max(1e-4, float(temperature))
        probs = F.softmax(logits, dim=-1)
        return _make_decision_result(names=names, logits_t=logits, probs_t=probs, t0=t0, mode="D1")


class ClosedFormRidgeReadout(nn.Module):
    """`D2`: closed-form `(X^T X + lambda I)^{-1} X^T Y` ridge readout without gradient steps.

    Supports both fixed-class ridge (`fit_ridge`) and pairwise query-option interaction ridge
    (`fit_pairwise_ridge` on `[q_w * c_w, |q_w - c_w|]`) so unseen shifted option sets are
    scored with an exact Cholesky linear regressor rather than uncalibrated cosine fallback.
    """

    readout_id = "D2"

    def __init__(self, d_model: int, reg_lambda: float = 1.0, null_threshold: float = 0.0) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.reg_lambda = float(reg_lambda)
        self.null_threshold = float(null_threshold)
        self.logit_scale = 6.0
        self.register_buffer("weights", torch.zeros(self.d_model, 1))
        self.register_buffer("bias", torch.zeros(1))
        self.register_buffer("center", torch.zeros(self.d_model))
        self.register_buffer("inv_std", torch.ones(self.d_model))
        self.register_buffer("pair_weights", torch.zeros(2 * self.d_model, 1))
        self.register_buffer("pair_bias", torch.zeros(1))
        self.fitted_labels: List[str] = []
        self.is_fitted = False
        self.has_pairwise = False
        self.has_vector_ridge = False

    @torch.no_grad()
    def fit_ridge(
        self,
        features: torch.Tensor,
        target_indices: torch.Tensor,
        label_names: Optional[Sequence[str]] = None,
        lambda_reg: Optional[float] = None,
    ) -> None:
        """Fit closed-form ridge regression on `(N, d)` frozen features (either class-index or target-vector)."""
        if lambda_reg is not None:
            self.reg_lambda = float(lambda_reg)
        n_samples, d = features.shape
        if d != self.d_model:
            raise ValueError(f"Expected d_model={self.d_model}, got {d}")
        ones = torch.ones(n_samples, 1, device=features.device, dtype=features.dtype)
        x_aug = torch.cat([features, ones], dim=1)  # (N, d + 1)
        gram = x_aug.transpose(0, 1) @ x_aug
        reg = torch.eye(d + 1, device=features.device, dtype=features.dtype) * self.reg_lambda
        reg[-1, -1] = 1e-5

        if target_indices.dim() == 2:
            rhs = x_aug.transpose(0, 1) @ target_indices.to(features.dtype)
            sol = torch.linalg.solve(gram + reg, rhs)
            self.weights = sol[:-1, :].contiguous()
            self.bias = sol[-1, :].contiguous()
            self.has_vector_ridge = True
            self.is_fitted = True
            return

        if label_names is None:
            raise ValueError("label_names is required when target_indices is 1D class indices")
        num_classes = len(label_names)
        y_onehot = F.one_hot(target_indices.long(), num_classes=num_classes).to(features.dtype) * 2.0 - 1.0
        rhs = x_aug.transpose(0, 1) @ y_onehot
        sol = torch.linalg.solve(gram + reg, rhs)
        self.weights = sol[:-1, :].contiguous()
        self.bias = sol[-1, :].contiguous()
        self.fitted_labels = list(label_names)
        self.has_vector_ridge = False
        self.is_fitted = True

    def _whiten(self, v: torch.Tensor) -> torch.Tensor:
        v_c = (v - self.center.to(v.device, v.dtype)) * self.inv_std.to(v.device, v.dtype)
        return F.normalize(v_c, dim=-1)

    def _pair_features(self, q_vec: torch.Tensor, opt_vecs: torch.Tensor) -> torch.Tensor:
        q_n = self._whiten(q_vec).expand_as(opt_vecs)
        o_n = self._whiten(opt_vecs)
        return torch.cat([q_n * o_n, (q_n - o_n).abs()], dim=-1)  # (K, 2*d)

    @torch.no_grad()
    def fit_pairwise_ridge(
        self,
        query_hiddens: Sequence[torch.Tensor],
        option_hiddens_list: Sequence[Sequence[torch.Tensor]],
        gold_indices: Sequence[Optional[int]],
    ) -> None:
        """Fit closed-form Cholesky ridge on `(q, option_k)` interaction features (zero SGD steps)."""
        all_vecs: List[torch.Tensor] = []
        for q_h, opt_hs in zip(query_hiddens, option_hiddens_list):
            all_vecs.append(_pool_hidden(q_h))
            for o_h in opt_hs:
                all_vecs.append(_pool_hidden(o_h))
        stacked = torch.cat(all_vecs, dim=0)
        self.center.copy_(stacked.mean(dim=0))
        self.inv_std.copy_(1.0 / torch.sqrt(stacked.var(dim=0, unbiased=False) + 1e-4))

        rows: List[torch.Tensor] = []
        ys: List[float] = []
        for q_h, opt_hs, gold_idx in zip(query_hiddens, option_hiddens_list, gold_indices):
            q_p = _pool_hidden(q_h)
            o_p = torch.cat([_pool_hidden(o) for o in opt_hs], dim=0)
            pf = self._pair_features(q_p, o_p)  # (K, 2d)
            rows.append(pf)
            for k_i in range(pf.shape[0]):
                ys.append(1.0 if (gold_idx is not None and k_i == gold_idx) else -1.0)

        x_mat = torch.cat(rows, dim=0)  # (M, 2d)
        m_rows, p_dim = x_mat.shape
        ones = torch.ones(m_rows, 1, device=x_mat.device, dtype=x_mat.dtype)
        x_aug = torch.cat([x_mat, ones], dim=1)  # (M, 2d + 1)
        y_col = torch.tensor(ys, device=x_mat.device, dtype=x_mat.dtype).unsqueeze(1)

        gram = x_aug.transpose(0, 1) @ x_aug
        reg = torch.eye(p_dim + 1, device=x_mat.device, dtype=x_mat.dtype) * self.reg_lambda
        reg[-1, -1] = 1e-5
        sol = torch.linalg.solve(gram + reg, x_aug.transpose(0, 1) @ y_col)
        self.pair_weights = sol[:-1, :].contiguous()
        self.pair_bias = sol[-1, :].contiguous()
        self.has_pairwise = True
        self.is_fitted = True

    @torch.no_grad()
    def calibrate_null_threshold(
        self,
        val_queries: Sequence[torch.Tensor],
        val_options_list: Sequence[Sequence[torch.Tensor]],
        val_option_names_list: Sequence[Sequence[str]],
        is_null_flags: Sequence[bool],
    ) -> float:
        max_scores: List[float] = []
        for q_h, opt_hs, opt_names in zip(val_queries, val_options_list, val_option_names_list):
            res = self.score_options(q_h, opt_hs, opt_names, include_null=False)
            max_scores.append(max(res.logits) / max(1e-4, self.logit_scale))
        tau = _optimal_null_threshold(max_scores, list(is_null_flags))
        if tau is not None:
            self.null_threshold = tau
        return self.null_threshold

    def score_options(
        self,
        query_hidden: torch.Tensor,
        option_hiddens: Sequence[torch.Tensor],
        option_names: Sequence[str],
        include_null: bool = True,
        temperature: float = 1.0,
    ) -> DecisionResult:
        t0 = time.perf_counter()
        q_vec = _pool_hidden(query_hidden)  # (1, d)
        names = list(option_names)

        if self.has_pairwise:
            opt_vecs = torch.cat([_pool_hidden(o) for o in option_hiddens], dim=0)
            pf = self._pair_features(q_vec, opt_vecs)
            raw_opt = (pf @ self.pair_weights + self.pair_bias).squeeze(-1)
            scale = self.logit_scale
        elif self.is_fitted and self.has_vector_ridge:
            q_hat = F.normalize(q_vec @ self.weights + self.bias, dim=-1)
            opt_vecs = F.normalize(torch.cat([_pool_hidden(o) for o in option_hiddens], dim=0), dim=-1)
            raw_opt = (q_hat @ opt_vecs.transpose(0, 1)).squeeze(0)
            scale = self.logit_scale
        elif self.is_fitted and set(names).issubset(set(self.fitted_labels)):
            all_scores = (q_vec @ self.weights + self.bias).squeeze(0)
            idx_map = {lab: i for i, lab in enumerate(self.fitted_labels)}
            raw_opt = torch.stack([all_scores[idx_map[name]] for name in names])
            scale = 1.0
        else:
            q_norm = F.normalize(q_vec, dim=-1)
            opt_vecs = F.normalize(torch.cat([_pool_hidden(o) for o in option_hiddens], dim=0), dim=-1)
            raw_opt = (q_norm @ opt_vecs.transpose(0, 1)).squeeze(0)
            scale = self.logit_scale

        if include_null:
            null_s = torch.tensor([self.null_threshold], device=raw_opt.device, dtype=raw_opt.dtype)
            raw_scores = torch.cat([raw_opt, null_s], dim=0) * scale
            names.append(NULL_OPTION)
        else:
            raw_scores = raw_opt * scale

        logits = raw_scores / max(1e-4, float(temperature))
        probs = F.softmax(logits, dim=-1)
        return _make_decision_result(names=names, logits_t=logits, probs_t=probs, t0=t0, mode="D2")


class _NullBiasHolder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.null_bias = nn.Parameter(torch.zeros(1))


class OptionConditionedReadout(nn.Module):
    """`D3`: trainable option-conditioned bilinear + interaction head with calibrated `null` logit."""

    readout_id = "D3"

    def __init__(self, d_model: int, proj_dim: Optional[int] = None) -> None:
        super().__init__()
        self.d_model = int(d_model)
        p_dim = int(proj_dim) if proj_dim is not None else max(32, self.d_model // 2)
        self.proj_dim = p_dim
        self.q_proj = nn.Linear(self.d_model, p_dim, bias=False)
        self.opt_proj = nn.Linear(self.d_model, p_dim, bias=False)
        with torch.no_grad():
            self.opt_proj.weight.copy_(self.q_proj.weight)
        self.pair_scorer = nn.Sequential(
            nn.Linear(p_dim * 3, p_dim),
            nn.SiLU(),
            nn.Linear(p_dim, 1),
        )
        self.null_query_head = nn.Sequential(
            nn.Linear(p_dim * 2, p_dim // 2),
            nn.SiLU(),
            nn.Linear(p_dim // 2, 1),
        )
        with torch.no_grad():
            nn.init.zeros_(self.pair_scorer[-1].weight)
            nn.init.zeros_(self.pair_scorer[-1].bias)
            nn.init.zeros_(self.null_query_head[-1].weight)
            nn.init.zeros_(self.null_query_head[-1].bias)
        self.head = _NullBiasHolder()

    def compute_logits(
        self,
        query_hidden: torch.Tensor,
        option_hiddens: Sequence[torch.Tensor],
        include_null: bool = True,
    ) -> torch.Tensor:
        q_raw = _pool_hidden(query_hidden)  # (1, d)
        opts_raw = torch.cat([_pool_hidden(o) for o in option_hiddens], dim=0)  # (K, d)
        z_q = F.normalize(self.q_proj(q_raw), dim=-1)  # (1, p)
        z_o = F.normalize(self.opt_proj(opts_raw), dim=-1)  # (K, p)

        z_q_exp = z_q.expand_as(z_o)
        pair_feat = torch.cat([z_q_exp * z_o, (z_q_exp - z_o).abs(), z_o], dim=-1)
        bilinear = (z_q @ z_o.transpose(0, 1)).squeeze(0) * (self.proj_dim ** 0.5)
        mlp_score = self.pair_scorer(pair_feat).squeeze(-1)
        opt_logits = bilinear + mlp_score

        if not include_null:
            return opt_logits

        max_opt = z_o.max(dim=0, keepdim=True).values
        null_feat = torch.cat([z_q, (z_q - max_opt).abs()], dim=-1)
        null_logit = self.null_query_head(null_feat).squeeze(0) + self.head.null_bias
        return torch.cat([opt_logits, null_logit], dim=0)

    def forward_logits(
        self,
        query_hidden: torch.Tensor,
        option_hiddens: Sequence[torch.Tensor],
        include_null: bool = True,
    ) -> torch.Tensor:
        return self.compute_logits(query_hidden, option_hiddens, include_null=include_null).unsqueeze(0)

    def score_options(
        self,
        query_hidden: torch.Tensor,
        option_hiddens: Sequence[torch.Tensor],
        option_names: Sequence[str],
        include_null: bool = True,
        temperature: float = 1.0,
    ) -> DecisionResult:
        t0 = time.perf_counter()
        names = list(option_names) + ([NULL_OPTION] if include_null else [])
        raw_logits = self.compute_logits(query_hidden, option_hiddens, include_null=include_null)
        logits = raw_logits / max(1e-4, float(temperature))
        probs = F.softmax(logits, dim=-1)
        return _make_decision_result(names=names, logits_t=logits, probs_t=probs, t0=t0, mode="D3")


class SplitSpecialistSystem(nn.Module):
    """`D4`: equal-total-parameter baseline splitting budget into `(LM + Specialist Encoder + D3 Head)`."""

    readout_id = "D4"

    def __init__(
        self,
        core_id: str,
        target_total_params: int | str,
        vocab_size: int = 4096,
        lm_budget_share: float = 0.78,
        tolerance: float = 0.02,
        chunk_size: int = 64,
    ) -> None:
        super().__init__()
        if isinstance(target_total_params, str):
            total_budget = TIER_BUDGETS[target_total_params]
        else:
            total_budget = int(target_total_params)
        if not (0.25 <= lm_budget_share <= 0.90):
            raise ValueError(f"lm_budget_share={lm_budget_share} must lie in [0.25, 0.90]")

        share_grid = [lm_budget_share] + [s for s in (0.72, 0.65, 0.58, 0.50) if abs(s - lm_budget_share) > 1e-4]
        best_pair: Optional[Tuple[float, CandidateLM, CandidateLM, OptionConditionedReadout]] = None

        for share in share_grid:
            lm_target = int(round(total_budget * share))
            try:
                lm_cfg = solve_core_config(core_id, tier_or_budget=lm_target, vocab_size=vocab_size, tolerance=0.08)
            except ValueError:
                continue
            cand_lm: CandidateLM = build_candidate_lm(lm_cfg, chunk_size=chunk_size)
            lm_actual = sum(p.numel() for p in cand_lm.parameters() if p.requires_grad)

            remaining = max(32_000, total_budget - lm_actual)
            for head_reserve_frac in (0.02, 0.04, 0.08, 0.12, 0.16, 0.20):
                spec_target = max(24_000, int(round(remaining * (1.0 - head_reserve_frac))))
                try:
                    spec_cfg = solve_core_config(
                        core_id,
                        tier_or_budget=spec_target,
                        vocab_size=vocab_size,
                        tolerance=0.15,
                    )
                except ValueError:
                    continue
                spec_lm = build_candidate_lm(spec_cfg, chunk_size=chunk_size)
                readout = OptionConditionedReadout(d_model=spec_cfg.d_model)
                tot = (
                    lm_actual
                    + sum(p.numel() for p in spec_lm.parameters() if p.requires_grad)
                    + sum(p.numel() for p in readout.parameters() if p.requires_grad)
                )
                rel_err = abs(tot - total_budget) / float(total_budget)
                if best_pair is None or rel_err < best_pair[0]:
                    best_pair = (rel_err, cand_lm, spec_lm, readout)
            if best_pair is not None and best_pair[0] <= tolerance:
                break

        if best_pair is None or best_pair[0] > tolerance:
            best_err = best_pair[0] if best_pair is not None else float("inf")
            raise ValueError(
                f"SplitSpecialistSystem could not match budget={total_budget} within ±{tolerance:.1%} "
                f"(best error={best_err:.2%})"
            )

        self.relative_budget_error = best_pair[0]
        self.target_total_params = total_budget
        self.lm_model: CandidateLM = best_pair[1]
        self.specialist_model: CandidateLM = best_pair[2]
        self.readout: OptionConditionedReadout = best_pair[3]

    def total_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def encode_for_decision(self, token_ids: torch.Tensor) -> torch.Tensor:
        details = self.specialist_model(token_ids, return_details=True)
        return details["hidden_states"]

    def score_token_options(
        self,
        query_ids: torch.Tensor,
        option_id_list: Sequence[torch.Tensor],
        option_names: Sequence[str],
        include_null: bool = True,
        temperature: float = 1.0,
    ) -> DecisionResult:
        q_h = self.encode_for_decision(query_ids)
        opt_h = [self.encode_for_decision(oid) for oid in option_id_list]
        return self.readout.score_options(
            q_h,
            opt_h,
            option_names=option_names,
            include_null=include_null,
            temperature=temperature,
        )

    def score_options(
        self,
        query_hidden: torch.Tensor,
        option_hiddens: Sequence[torch.Tensor],
        option_names: Sequence[str],
        include_null: bool = True,
        temperature: float = 1.0,
    ) -> DecisionResult:
        return self.readout.score_options(
            query_hidden,
            option_hiddens,
            option_names=option_names,
            include_null=include_null,
            temperature=temperature,
        )


def build_decision_readout(readout_id: str, d_model: int, **kwargs: Any) -> nn.Module:
    key = readout_id.strip().upper()
    if key in {"D1", "ZERO_SHOT_PROTOTYPE"}:
        return ZeroShotPrototypeReadout(d_model=d_model, **kwargs)
    if key in {"D2", "CLOSED_FORM_RIDGE"}:
        return ClosedFormRidgeReadout(d_model=d_model, **kwargs)
    if key in {"D3", "OPTION_CONDITIONED"}:
        return OptionConditionedReadout(d_model=d_model, **kwargs)
    raise ValueError(f"Unknown readout_id {readout_id!r}; use SplitSpecialistSystem directly for D4")
