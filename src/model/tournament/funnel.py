"""4-Stage Elimination, Interference, and Leave-One-Out Pruning Funnel.

Implements the quantitative decision rules specified in `recamem-core-architecture-tournament`:
- Stage 0 (Milestone M0): Parameter budget parity (`±2%`), tokenizer lossless UTF-8 checks,
  scan-vs-step numerical parity (`< 1e-4`), and split disjointness verification.
- Stage 1: Single-seed mechanistic falsification gates (`mqar_16_acc >= 0.75` for cores,
  `ΔBPB >= 0.015` or `+2.0` Macro-F1 pts for auxiliary objectives `O2`/`O3` vs `O1`,
  `O4` RTRL throughput + BPB parity gate).
- Stage 2: Multi-seed per-track ranking with 10,000-resample paired bootstrap 95% CIs and
  Pareto frontier analysis across Tracks 1..5 (preserving `C6` Transformer as reference anchor
  while selecting the edge streaming core with sub-quadratic `O(1)`/`O(W)` state complexity).
- Stage 3: Shared-backbone vs Equal-Budget Split-Specialist (`D4`) interference gate
  (`ΔBPB <= +3.0%` and `ΔMacro-F1 >= -2.0` pts -> unified; `> +5.0%` or `> 5.0` pts -> split).
- Stage 4: Greedy forward assembly + Leave-One-Out (`LOO`) complexity pruning (`< 0.5%`
  utility drop -> prune to simpler component).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from src.model.tournament.stats import compute_pareto_frontier, paired_bootstrap_ci

STAGE1_MQAR16_MIN = 0.75
STAGE1_AUX_DELTA_BPB_MIN = 0.015
STAGE1_AUX_DELTA_F1_MIN = 0.020

STAGE3_UNIFIED_MAX_BPB_REL_DEG = 0.03
STAGE3_UNIFIED_MAX_F1_DROP_PTS = 2.0
STAGE3_HARD_REJECT_BPB_REL_DEG = 0.05
STAGE3_HARD_REJECT_F1_DROP_PTS = 5.0

STAGE4_LOO_MIN_UTILITY_GAIN = 0.005


@dataclass
class Stage1GateVerdict:
    arm_id: str
    track: str
    passed: bool
    is_reference_only: bool
    rejection_reason: Optional[str]
    key_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_stage1_gates(
    *,
    core_records: Sequence[Mapping[str, Any]],
    objective_records: Sequence[Mapping[str, Any]],
    tokenizer_records: Sequence[Mapping[str, Any]],
    decision_records: Sequence[Mapping[str, Any]],
    router_records: Sequence[Mapping[str, Any]],
    memory_records: Sequence[Mapping[str, Any]],
    mqar16_threshold: float = STAGE1_MQAR16_MIN,
    aux_delta_bpb_min: float = STAGE1_AUX_DELTA_BPB_MIN,
) -> Dict[str, Any]:
    """Apply Stage-1 mechanistic falsification gates to all candidate arms."""
    verdicts: List[Stage1GateVerdict] = []

    # Track 1: Sequence-mixing cores C0..C6
    for rec in core_records:
        cid = str(rec["arm_id"])
        diverged = bool(rec.get("diverged", False))
        mqar16 = float(rec.get("mqar_16_acc", 0.0))
        complexity = str(rec.get("complexity_class", ""))
        is_ref = complexity == "linear_kv"  # C6 Softmax Transformer is an O(T) KV reference anchor
        reasons: List[str] = []
        if diverged or not math.isfinite(float(rec.get("held_out_bpb", float("nan")))):
            reasons.append("diverged_or_non_finite_bpb")
        if mqar16 < mqar16_threshold:
            reasons.append(f"mqar_16_acc={mqar16:.3f} < {mqar16_threshold:.2f}")
        verdicts.append(
            Stage1GateVerdict(
                arm_id=cid,
                track="track1_core",
                passed=(len(reasons) == 0),
                is_reference_only=is_ref,
                rejection_reason="; ".join(reasons) if reasons else None,
                key_metrics={
                    "mqar_16_acc": mqar16,
                    "mqar_64_acc": float(rec.get("mqar_64_acc", 0.0)),
                    "copy_acc": float(rec.get("copy_acc", 0.0)),
                    "held_out_bpb": float(rec.get("held_out_bpb", 0.0)),
                    "complexity_class": complexity,
                },
            )
        )

    # Track 2: Auxiliary objectives O1..O3 + O4 RTRL + Tokenizers
    o1_rec = next((r for r in objective_records if str(r["arm_id"]) == "O1"), None)
    o1_bpb = float(o1_rec["held_out_bpb"]) if o1_rec is not None else None
    o1_f1 = float(o1_rec.get("probe_macro_f1", 0.0)) if o1_rec is not None else 0.0
    o1_tps = float(o1_rec.get("train_tokens_per_sec", 0.0)) if o1_rec is not None else 0.0

    for rec in objective_records:
        oid = str(rec["arm_id"])
        bpb = float(rec["held_out_bpb"])
        f1 = float(rec.get("probe_macro_f1", 0.0))
        tps = float(rec.get("train_tokens_per_sec", 0.0))
        if oid == "O1":
            verdicts.append(
                Stage1GateVerdict(
                    arm_id=oid,
                    track="track2_objective",
                    passed=True,
                    is_reference_only=False,
                    rejection_reason=None,
                    key_metrics={"held_out_bpb": bpb, "probe_macro_f1": f1},
                )
            )
        elif oid.startswith("O4"):
            # O4 online_rtrl must match chunked_bptt BPB within 0.02 AND achieve higher throughput
            bptt_bpb = float(rec.get("paired_bptt_bpb", o1_bpb if o1_bpb is not None else bpb))
            bptt_tps = float(rec.get("paired_bptt_tps", o1_tps))
            reasons = []
            if bpb - bptt_bpb > 0.02:
                reasons.append(f"rtrl_bpb_penalty={bpb - bptt_bpb:+.3f} > +0.020")
            if tps < bptt_tps:
                reasons.append(f"rtrl_tps={tps:.1f} < chunked_bptt_tps={bptt_tps:.1f}")
            if not bool(rec.get("supports_matrix_cores", False)):
                reasons.append("incompatible_with_matrix_fast_weight_cores")
            verdicts.append(
                Stage1GateVerdict(
                    arm_id=oid,
                    track="track2_gradient",
                    passed=(len(reasons) == 0),
                    is_reference_only=False,
                    rejection_reason="; ".join(reasons) if reasons else None,
                    key_metrics={"held_out_bpb": bpb, "train_tokens_per_sec": tps},
                )
            )
        else:
            delta_bpb = (o1_bpb - bpb) if o1_bpb is not None else 0.0
            delta_f1 = f1 - o1_f1
            passed = (delta_bpb >= aux_delta_bpb_min) or (
                delta_f1 >= STAGE1_AUX_DELTA_F1_MIN and delta_bpb >= -0.005
            )
            reason = (
                None
                if passed
                else f"delta_bpb={delta_bpb:+.4f} < {aux_delta_bpb_min:.3f} and delta_f1={delta_f1:+.3f} < {STAGE1_AUX_DELTA_F1_MIN:.3f}"
            )
            verdicts.append(
                Stage1GateVerdict(
                    arm_id=oid,
                    track="track2_objective",
                    passed=passed,
                    is_reference_only=False,
                    rejection_reason=reason,
                    key_metrics={"held_out_bpb": bpb, "delta_bpb_vs_o1": delta_bpb, "delta_f1_vs_o1": delta_f1},
                )
            )

    for rec in tokenizer_records:
        tid = str(rec["arm_id"])
        lossless = bool(rec.get("lossless_roundtrip", True))
        verdicts.append(
            Stage1GateVerdict(
                arm_id=tid,
                track="track2_tokenizer",
                passed=lossless,
                is_reference_only=False,
                rejection_reason=None if lossless else "failed_utf8_roundtrip",
                key_metrics={
                    "equal_byte_bpb": float(rec.get("equal_byte_bpb", rec.get("held_out_bpb", 0.0))),
                    "equal_clock_bpb": float(rec.get("equal_clock_bpb", rec.get("held_out_bpb", 0.0))),
                    "int8_bytes_per_second": float(rec.get("int8_bytes_per_second", 0.0)),
                },
            )
        )

    for rec in decision_records:
        did = str(rec["arm_id"])
        perm_ok = bool(rec.get("permutation_invariant", True))
        verdicts.append(
            Stage1GateVerdict(
                arm_id=did,
                track="track3_decision",
                passed=perm_ok,
                is_reference_only=(did == "D4"),
                rejection_reason=None if perm_ok else "violated_option_permutation_invariance",
                key_metrics={
                    "in_domain_macro_f1": float(rec.get("in_domain_macro_f1", 0.0)),
                    "shifted_macro_f1": float(rec.get("shifted_macro_f1", 0.0)),
                    "oos_f1": float(rec.get("oos_f1", 0.0)),
                    "adaptive_ece": float(rec.get("adaptive_ece", 1.0)),
                },
            )
        )

    for rec in router_records:
        rid = str(rec["arm_id"])
        unsafe_rate = float(rec.get("unblocked_invalid_tool_rate", 0.0))
        passed = unsafe_rate == 0.0
        verdicts.append(
            Stage1GateVerdict(
                arm_id=rid,
                track="track4_router",
                passed=passed,
                is_reference_only=False,
                rejection_reason=None if passed else f"unblocked_invalid_tool_rate={unsafe_rate:.3f} > 0",
                key_metrics={
                    "normalized_utility": float(rec.get("normalized_utility", 0.0)),
                    "false_tool_rate": float(rec.get("false_tool_rate", 0.0)),
                    "p50_latency_ms": float(rec.get("p50_latency_ms", 0.0)),
                },
            )
        )

    for rec in memory_records:
        mid = str(rec["arm_id"])
        ret_acc = float(rec.get("average_final_accuracy", 0.0))
        contra_acc = float(rec.get("contradiction_update_accuracy", 0.0))
        passed = ret_acc >= 0.40 and contra_acc >= 0.50
        reason = (
            None
            if passed
            else f"retention_acc={ret_acc:.2f} < 0.40 or contradiction_acc={contra_acc:.2f} < 0.50"
        )
        verdicts.append(
            Stage1GateVerdict(
                arm_id=mid,
                track="track5_memory",
                passed=passed,
                is_reference_only=False,
                rejection_reason=reason,
                key_metrics={
                    "average_final_accuracy": ret_acc,
                    "backward_forgetting": float(rec.get("backward_forgetting", 0.0)),
                    "contradiction_update_accuracy": contra_acc,
                    "stale_answer_rate": float(rec.get("stale_answer_rate", 0.0)),
                },
            )
        )

    surviving_by_track: Dict[str, List[str]] = {}
    for v in verdicts:
        if v.passed and not v.is_reference_only:
            surviving_by_track.setdefault(v.track, []).append(v.arm_id)

    return {
        "verdicts": [v.to_dict() for v in verdicts],
        "surviving_by_track": surviving_by_track,
    }


def _score_core_utility(rec: Mapping[str, Any]) -> float:
    mqar_mean = 0.25 * (
        float(rec.get("mqar_4_acc", 0.0))
        + float(rec.get("mqar_16_acc", 0.0))
        + float(rec.get("mqar_32_acc", 0.0))
        + float(rec.get("mqar_64_acc", 0.0))
    )
    copy_acc = float(rec.get("copy_acc", 0.0))
    bpb = max(0.25, float(rec.get("held_out_bpb", 8.0)))
    extrap_ratio = min(2.0, max(1.0, float(rec.get("extrapolation_bpb_ratio_4k_vs_256", 1.25))))
    grammar_acc = float(rec.get("grammar_acc", 0.5))
    # Sub-quadratic streaming efficiency bonus (constant state vs growing KV cache)
    state_kb = max(1.0, float(rec.get("state_bytes_at_4k", 65536)) / 1024.0)
    state_eff = 1.0 / (1.0 + 0.05 * math.log2(state_kb))
    return (
        0.35 * mqar_mean
        + 0.15 * copy_acc
        + 0.25 * (2.0 / (1.0 + bpb))
        + 0.10 * grammar_acc
        + 0.10 * (1.0 / extrap_ratio)
        + 0.05 * state_eff
    )


def _score_decision_utility(rec: Mapping[str, Any]) -> float:
    """Rank decision arms strictly by unseen/shifted generalization and risk-calibration (`shifted_macro_f1`, `oos_f1`, `aurc`, `adaptive_ece`).

    `in_domain_macro_f1` is intentionally excluded from selection utility (reported only as
    a diagnostic) so memorizing seen option labels never overrides shifted/OOS reliability.
    """
    sh_f1 = float(rec.get("shifted_macro_f1", 0.0))
    oos_f1 = float(rec.get("oos_f1", 0.0))
    aurc = float(rec.get("aurc", 0.5))
    ece = float(rec.get("adaptive_ece", 0.5))
    return 0.40 * sh_f1 + 0.20 * oos_f1 + 0.20 * (1.0 - aurc) + 0.20 * (1.0 - ece)


def _score_router_utility(rec: Mapping[str, Any]) -> float:
    """Rank router arms by cost-weighted utility, AURC, calibration ECE, and false-tool safety."""
    util = float(rec.get("normalized_utility", 0.0))
    false_tool = float(rec.get("false_tool_rate", 0.0))
    ece = float(rec.get("adaptive_ece", 0.5))
    aurc = float(rec.get("aurc", ece))
    return 0.40 * util + 0.20 * (1.0 - false_tool) + 0.20 * (1.0 - aurc) + 0.20 * (1.0 - ece)


def _score_memory_utility(rec: Mapping[str, Any]) -> float:
    ret = float(rec.get("average_final_accuracy", 0.0))
    forget = float(rec.get("backward_forgetting", 1.0))
    contra = float(rec.get("contradiction_update_accuracy", 0.0))
    stale = float(rec.get("stale_answer_rate", 1.0))
    total_mb = float(rec.get("total_footprint_bytes", 0)) / (1024.0 * 1024.0)
    mem_eff = 1.0 / (1.0 + 0.10 * total_mb)
    return 0.35 * ret + 0.20 * (1.0 - forget) + 0.25 * contra + 0.10 * (1.0 - stale) + 0.10 * mem_eff


def evaluate_stage2_tracks(
    *,
    stage1_result: Mapping[str, Any],
    core_records: Sequence[Mapping[str, Any]],
    objective_records: Sequence[Mapping[str, Any]],
    tokenizer_records: Sequence[Mapping[str, Any]],
    decision_records: Sequence[Mapping[str, Any]],
    router_records: Sequence[Mapping[str, Any]],
    memory_records: Sequence[Mapping[str, Any]],
    seed: int = 1337,
) -> Dict[str, Any]:
    """Rank surviving arms per track using composite utility, Pareto frontiers, and 10k bootstrap CIs."""
    surviving = stage1_result["surviving_by_track"]

    def _rank_track(
        track_key: str,
        records: Sequence[Mapping[str, Any]],
        scorer,
        pareto_spec: Mapping[str, str],
        per_item_key: str,
        allow_fallback_all: bool = False,
    ) -> Dict[str, Any]:
        allowed_ids = set(surviving.get(track_key, []))
        pool = [dict(r) for r in records if str(r["arm_id"]) in allowed_ids]
        if not pool and allow_fallback_all:
            pool = [dict(r) for r in records if not str(r["arm_id"]).startswith("D4")]
        if not pool:
            raise ValueError(f"Zero candidates survived Stage 1 in {track_key}!")

        for r in pool:
            r["composite_score"] = float(scorer(r))
        pool.sort(key=lambda r: r["composite_score"], reverse=True)
        winner = pool[0]
        pareto = compute_pareto_frontier(pool, pareto_spec)

        bootstrap_vs_runner_up = None
        runner_up = pool[1] if len(pool) >= 2 else None
        if runner_up is None:
            others = [dict(r) for r in records if str(r["arm_id"]) != str(winner["arm_id"])]
            if others:
                for o in others:
                    o["composite_score"] = float(scorer(o))
                others.sort(key=lambda r: r["composite_score"], reverse=True)
                runner_up = others[0]
        if runner_up is not None and per_item_key in winner and per_item_key in runner_up:
            s_a = winner[per_item_key]
            s_b = runner_up[per_item_key]
            if len(s_a) == len(s_b) and len(s_a) > 0:
                bootstrap_vs_runner_up = paired_bootstrap_ci(
                    s_a,
                    s_b,
                    arm_a=str(winner["arm_id"]),
                    arm_b=str(runner_up["arm_id"]),
                    metric=per_item_key,
                    seed=seed,
                ).to_dict()

        return {
            "winner_id": str(winner["arm_id"]),
            "winner_score": winner["composite_score"],
            "pareto_frontier": pareto,
            "ranking": [
                {"arm_id": str(r["arm_id"]), "composite_score": r["composite_score"]}
                for r in pool
            ],
            "bootstrap_vs_runner_up": bootstrap_vs_runner_up,
        }

    core_res = _rank_track(
        "track1_core",
        core_records,
        _score_core_utility,
        {"mqar_16_acc": "max", "held_out_bpb": "min", "state_bytes_at_4k": "min"},
        per_item_key="per_item_mqar_scores",
    )

    # Objective winner (surviving O1..O3; defaults to O1 if O2/O3 failed Stage-1 gate)
    obj_res = _rank_track(
        "track2_objective",
        objective_records,
        lambda r: 2.0 / (1.0 + float(r["held_out_bpb"])) + 0.2 * float(r.get("probe_macro_f1", 0.0)),
        {"held_out_bpb": "min"},
        per_item_key="per_doc_neg_bpb",
        allow_fallback_all=True,
    )

    # Tokenizer winner: Equal-Byte BPB + Equal-Wall-Clock BPB + INT8 bytes/sec
    tok_res = _rank_track(
        "track2_tokenizer",
        tokenizer_records,
        lambda r: (
            0.40 * (2.0 / (1.0 + float(r.get("equal_byte_bpb", r.get("held_out_bpb", 4.0)))))
            + 0.45 * (2.0 / (1.0 + float(r.get("equal_clock_bpb", r.get("held_out_bpb", 4.0)))))
            + 0.15 * min(1.0, float(r.get("int8_bytes_per_second", 100.0)) / 2000.0)
        ),
        {"equal_clock_bpb": "min", "int8_bytes_per_second": "max"},
        per_item_key="per_doc_neg_bpb",
    )

    dec_res = _rank_track(
        "track3_decision",
        decision_records,
        _score_decision_utility,
        {"shifted_macro_f1": "max", "adaptive_ece": "min"},
        per_item_key="per_item_correct",
    )

    rt_res = _rank_track(
        "track4_router",
        router_records,
        _score_router_utility,
        {"normalized_utility": "max", "p50_latency_ms": "min"},
        per_item_key="per_item_utility",
    )

    mem_res = _rank_track(
        "track5_memory",
        memory_records,
        _score_memory_utility,
        {"average_final_accuracy": "max", "contradiction_update_accuracy": "max", "total_footprint_bytes": "min"},
        per_item_key="per_item_recall",
    )

    return {
        "track1_core": core_res,
        "track2_objective": obj_res,
        "track2_tokenizer": tok_res,
        "track3_decision": dec_res,
        "track4_router": rt_res,
        "track5_memory": mem_res,
        "selected_gradient_mode": (
            surviving["track2_gradient"][0] if surviving.get("track2_gradient") else "chunked_bptt"
        ),
    }


def evaluate_stage3_interference(
    *,
    standalone_lm_bpb: float,
    standalone_decision_macro_f1: float,
    unified_interleaved_bpb: float,
    unified_interleaved_macro_f1: float,
    unified_summed_bpb: float,
    unified_summed_macro_f1: float,
    split_d4_bpb: float,
    split_d4_macro_f1: float,
    unified_total_params: int,
    split_d4_total_params: int,
) -> Dict[str, Any]:
    """Evaluate Stage-3 Shared-Backbone vs Equal-Budget Split-Specialist (`D4`) interference gate."""
    param_rel_diff = abs(unified_total_params - split_d4_total_params) / float(max(1, unified_total_params))
    if param_rel_diff > 0.04:
        raise ValueError(
            f"Shared vs Split parameter parity violated: unified={unified_total_params} vs split={split_d4_total_params}"
        )

    # Pick better multi-task batching schedule between interleaved and summed
    interleaved_score = (2.0 / (1.0 + unified_interleaved_bpb)) + unified_interleaved_macro_f1
    summed_score = (2.0 / (1.0 + unified_summed_bpb)) + unified_summed_macro_f1
    if interleaved_score >= summed_score:
        best_batching = "interleaved"
        u_bpb = float(unified_interleaved_bpb)
        u_f1 = float(unified_interleaved_macro_f1)
    else:
        best_batching = "summed"
        u_bpb = float(unified_summed_bpb)
        u_f1 = float(unified_summed_macro_f1)

    rel_bpb_deg = (u_bpb - standalone_lm_bpb) / max(1e-6, float(standalone_lm_bpb))
    f1_drop_pts = (standalone_decision_macro_f1 - u_f1) * 100.0

    unified_composite = 0.55 * (2.0 / (1.0 + u_bpb)) + 0.45 * u_f1
    split_composite = 0.55 * (2.0 / (1.0 + split_d4_bpb)) + 0.45 * split_d4_macro_f1

    if (
        rel_bpb_deg > STAGE3_HARD_REJECT_BPB_REL_DEG
        or f1_drop_pts > STAGE3_HARD_REJECT_F1_DROP_PTS
    ):
        architecture_mode = "split_specialist"
        verdict_reason = (
            f"Hard interference rejection triggered: rel_bpb_deg={rel_bpb_deg:+.2%} "
            f"(limit +{STAGE3_HARD_REJECT_BPB_REL_DEG:.0%}), f1_drop={f1_drop_pts:+.2f} pts "
            f"(limit {STAGE3_HARD_REJECT_F1_DROP_PTS:.1f} pts)"
        )
    elif (
        rel_bpb_deg <= STAGE3_UNIFIED_MAX_BPB_REL_DEG
        and f1_drop_pts <= STAGE3_UNIFIED_MAX_F1_DROP_PTS
        and unified_composite >= split_composite - 0.01
    ):
        architecture_mode = "unified"
        verdict_reason = (
            f"Unified shared backbone passed interference gate: rel_bpb_deg={rel_bpb_deg:+.2%} <= +3.0%, "
            f"f1_drop={f1_drop_pts:+.2f} pts <= 2.0 pts, composite={unified_composite:.4f} vs split={split_composite:.4f}"
        )
    elif unified_composite >= split_composite:
        architecture_mode = "unified"
        verdict_reason = (
            f"Moderate interference zone resolved in favor of unified (composite {unified_composite:.4f} >= {split_composite:.4f})"
        )
    else:
        architecture_mode = "split_specialist"
        verdict_reason = (
            f"Equal-budget Split-Specialist (D4) outperformed unified ({split_composite:.4f} > {unified_composite:.4f})"
        )

    return {
        "selected_architecture_mode": architecture_mode,
        "selected_batching_mode": best_batching,
        "verdict_reason": verdict_reason,
        "relative_bpb_degradation": rel_bpb_deg,
        "macro_f1_drop_pts": f1_drop_pts,
        "unified_composite": unified_composite,
        "split_d4_composite": split_composite,
        "param_relative_diff": param_rel_diff,
        "comparison": {
            "standalone_lm_bpb": standalone_lm_bpb,
            "standalone_decision_macro_f1": standalone_decision_macro_f1,
            "unified_interleaved_bpb": unified_interleaved_bpb,
            "unified_interleaved_macro_f1": unified_interleaved_macro_f1,
            "unified_summed_bpb": unified_summed_bpb,
            "unified_summed_macro_f1": unified_summed_macro_f1,
            "split_d4_bpb": split_d4_bpb,
            "split_d4_macro_f1": split_d4_macro_f1,
        },
    }


def evaluate_stage4_loo_pruning(
    full_stack_utility: float,
    ablations: Sequence[Mapping[str, Any]],
    *,
    min_gain_to_keep: float = STAGE4_LOO_MIN_UTILITY_GAIN,
    seed: int = 1337,
) -> Dict[str, Any]:
    """Leave-One-Out (`LOO`) Occam's Razor pruning waterfall.

    Each item in `ablations` describes removing one optional complexity component:
      {
        "component": "short_conv4",
        "simpler_fallback": "conv_off",
        "ablated_utility": 0.812,
        "full_per_item": [...],      # optional paired scores
        "ablated_per_item": [...],   # optional paired scores
      }
    If `full_stack_utility - ablated_utility < min_gain_to_keep` (the component adds
    `< 0.5%` utility), the component is PRUNED in favor of `simpler_fallback`.
    """
    current_utility = float(full_stack_utility)
    waterfall: List[Dict[str, Any]] = []
    kept_components: List[str] = []
    pruned_components: Dict[str, str] = {}

    for item in ablations:
        comp = str(item["component"])
        fallback = str(item["simpler_fallback"])
        abl_u = float(item["ablated_utility"])
        gain_from_component = current_utility - abl_u

        ci_dict = None
        ci_includes_zero = False
        if "full_per_item" in item and "ablated_per_item" in item:
            f_arr = item["full_per_item"]
            a_arr = item["ablated_per_item"]
            if len(f_arr) == len(a_arr) and len(f_arr) > 0:
                ci = paired_bootstrap_ci(
                    f_arr,
                    a_arr,
                    arm_a=f"with_{comp}",
                    arm_b=fallback,
                    metric="utility",
                    seed=seed,
                )
                ci_dict = ci.to_dict()
                ci_includes_zero = ci.ci_low <= 0.0

        # Prune if marginal utility gain is below 0.5% threshold
        if gain_from_component < min_gain_to_keep:
            decision = "pruned"
            pruned_components[comp] = fallback
            current_utility = abl_u
            reason = (
                f"Marginal utility gain {gain_from_component:+.4f} < {min_gain_to_keep:.4f}; "
                f"pruned {comp!r} -> {fallback!r}"
            )
        else:
            decision = "kept"
            kept_components.append(comp)
            reason = (
                f"Component {comp!r} contributed {gain_from_component:+.4f} >= {min_gain_to_keep:.4f} "
                f"(kept over {fallback!r})"
            )

        waterfall.append(
            {
                "component": comp,
                "simpler_fallback": fallback,
                "utility_with_component": float(full_stack_utility),
                "utility_without_component": abl_u,
                "marginal_gain": gain_from_component,
                "ci_includes_zero": ci_includes_zero,
                "bootstrap_ci": ci_dict,
                "decision": decision,
                "reason": reason,
            }
        )

    return {
        "initial_full_utility": float(full_stack_utility),
        "final_pruned_utility": current_utility,
        "kept_components": kept_components,
        "pruned_components": pruned_components,
        "waterfall": waterfall,
    }
