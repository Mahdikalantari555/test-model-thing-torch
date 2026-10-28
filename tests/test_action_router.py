"""Tests for Track-3 decision readouts (D1..D4), calibration, and Track-4 routers (R1..R4)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import src.model.cores.calibration as calibration_mod
import src.model.cores.decision_readouts as decision_readouts
import src.model.cores.routers as routers_mod
import src.model.cores.tokenizer as tokenizer_mod


def test_decision_readouts_d1_d2_d3_permutation_invariance_null_and_d4_parity() -> None:
    torch.manual_seed(11)
    d = 48
    query_h = torch.randn(1, 12, d)
    opt_a = query_h + 0.05 * torch.randn(1, 12, d)  # strongly aligned with query
    opt_b = -query_h
    opt_c = torch.randn(1, 12, d)

    d1 = decision_readouts.build_decision_readout("D1", d_model=d)
    d2 = decision_readouts.build_decision_readout("D2", d_model=d)
    d3 = decision_readouts.build_decision_readout("D3", d_model=d)

    # Fit D2 Ridge probe on 8 synthetic identity-aligned pairs
    x_train = F.normalize(torch.randn(8, d), dim=-1)
    d2.fit_ridge(x_train, x_train, lambda_reg=0.1)

    for readout in (d1, d2, d3):
        r_abc = readout.score_options(query_h, [opt_a, opt_b, opt_c], ["alpha", "beta", "gamma"])
        r_cab = readout.score_options(query_h, [opt_c, opt_a, opt_b], ["gamma", "alpha", "beta"])
        assert r_abc.selected_option == "alpha"
        assert r_cab.selected_option == "alpha"
        # Permutation invariance of option probabilities (null stays at index -1)
        probs_abc = {
            "alpha": r_abc.probabilities[0],
            "beta": r_abc.probabilities[1],
            "gamma": r_abc.probabilities[2],
            "null": r_abc.probabilities[3],
        }
        probs_cab = {
            "gamma": r_cab.probabilities[0],
            "alpha": r_cab.probabilities[1],
            "beta": r_cab.probabilities[2],
            "null": r_cab.probabilities[3],
        }
        for key in probs_abc:
            assert abs(probs_abc[key] - probs_cab[key]) < 1e-5

    # Null / out-of-scope selection when all candidate options oppose query or null bias dominates
    d1_oos = d1.score_options(query_h, [-query_h, -0.9 * query_h], ["opt1", "opt2"], include_null=True)
    assert d1_oos.is_null and d1_oos.selected_option == "null"

    with torch.no_grad():
        d3.head.null_bias.fill_(25.0)
    d3_oos = d3.score_options(query_h, [opt_b, opt_c], ["opt1", "opt2"], include_null=True)
    assert d3_oos.is_null and d3_oos.selected_option == "null"

    # D4 Split-Specialist total parameter budget parity within ±2.0%
    split_sys = decision_readouts.SplitSpecialistSystem(
        core_id="C4",
        target_total_params="micro",
        vocab_size=256,
        tolerance=0.02,
    )
    total_p = split_sys.total_trainable_params()
    assert abs(total_p - 2_000_000) / 2_000_000 <= 0.02
    res_d4 = split_sys.score_token_options(
        torch.randint(4, 200, (1, 10)),
        [torch.randint(4, 200, (1, 6)), torch.randint(4, 200, (1, 6))],
        ["intent_a", "intent_b"],
    )
    assert res_d4.selected_option in {"intent_a", "intent_b", "null"}


def test_calibration_temperature_scaling_argmax_disjointness_and_ece_bounds() -> None:
    torch.manual_seed(19)
    n_val, n_test, num_classes = 120, 120, 4
    raw_val = torch.randn(n_val, num_classes)
    raw_test = torch.randn(n_test, num_classes)
    # Sample ground-truth labels from true softmax(raw), then multiply logits by 6x to make them severely overconfident!
    val_labels = torch.multinomial(F.softmax(raw_val, dim=-1), num_samples=1).squeeze(-1)
    test_labels = torch.multinomial(F.softmax(raw_test, dim=-1), num_samples=1).squeeze(-1)

    overconf_val = raw_val * 6.0
    overconf_test = raw_test * 6.0

    uncal_report = calibration_mod.compute_calibration_metrics(overconf_test, test_labels, temperature=1.0)
    cal = calibration_mod.TemperatureCalibrator(init_temperature=1.0)
    t_cal = cal.fit(overconf_val, val_labels, val_ids=[f"val_{i}" for i in range(n_val)])
    assert t_cal > 2.0, f"Expected T_cal > 2.0 for 6x overconfident logits, got {t_cal}"

    cal_report = cal.evaluate(
        overconf_test,
        test_labels,
        test_ids=[f"test_{i}" for i in range(n_test)],
    )
    # Post-hoc temperature scaling strictly preserves argmax accuracy while reducing NLL & ECE!
    assert abs(cal_report.accuracy - uncal_report.accuracy) < 1e-6
    assert cal_report.nll < uncal_report.nll
    assert cal_report.adaptive_ece < uncal_report.adaptive_ece
    assert 0.0 <= cal_report.aurc <= 1.0

    # Contaminating val_cal and test_eval IDs must raise ValueError
    with pytest.raises(ValueError, match="contamination"):
        cal.evaluate(overconf_test[:5], test_labels[:5], test_ids=["val_0", "test_1", "test_2", "test_3", "test_4"])


def test_routers_r1_r2_r3_r4_uniform_actions_guards_and_latency() -> None:
    torch.manual_seed(23)
    d = 48
    tok = tokenizer_mod.build_tokenizer("bytes", vocab_size=256)
    hidden = torch.randn(1, 20, d)
    lm_logits_last = torch.randn(1, 256)

    r1 = routers_mod.build_router("R1", d_model=d, abstain_threshold=0.10)
    r2 = routers_mod.build_router("R2", d_model=d, abstain_threshold=0.10)
    r3 = routers_mod.build_router("R3", d_model=d, route_token_ids=tok.route_token_ids, abstain_threshold=0.10)
    r4 = routers_mod.build_router("R4", d_model=d, abstain_threshold=0.10)

    latencies = []
    for router in (r1, r2, r3, r4):
        out = router.route(
            "Explain how gated delta fast weights work.",
            hidden_states=hidden,
            lm_logits_last=lm_logits_last,
            retrieval_similarity=0.80,
        )
        assert out.action in routers_mod.ACTION_NAMES
        assert set(out.probabilities.keys()) == set(routers_mod.ACTION_NAMES)
        assert abs(sum(out.probabilities.values()) - 1.0) < 1e-4
        assert out.latency_ms >= 0.0
        latencies.append(out.latency_ms)
    assert float(np.median(latencies)) < 50.0

    # Low-confidence guard forces non-abstain argmax to 'abstain'
    low_conf_probs = torch.tensor([0.55, 0.15, 0.10, 0.10, 0.10])  # raw argmax == 'generate'
    res_low = routers_mod.apply_safety_guards(low_conf_probs, abstain_threshold=0.80)
    assert res_low.raw_action == "generate"
    assert res_low.action == "abstain" and res_low.guard_triggered

    # Tool-schema & permission guard forces 'abstain' on missing/mistyped/unpermitted tool calls
    schema = {
        "allowed_tools": ["calculator"],
        "tools": {
            "calculator": {
                "required": ["expression", "precision"],
                "properties": {
                    "expression": {"type": "string"},
                    "precision": {"type": "integer"},
                },
                "additionalProperties": False,
            }
        },
    }
    tool_probs = torch.tensor([0.02, 0.02, 0.02, 0.92, 0.02])  # raw argmax == 'tool'
    # 1. Valid call passes guard
    ok_res = routers_mod.apply_safety_guards(
        tool_probs,
        abstain_threshold=0.40,
        tool_schema=schema,
        proposed_tool_call={"name": "calculator", "arguments": {"expression": "2+2", "precision": 2}},
    )
    assert ok_res.action == "tool" and not ok_res.guard_triggered

    # 2. Boolean passed where integer expected -> blocked!
    bad_type = routers_mod.apply_safety_guards(
        tool_probs,
        abstain_threshold=0.40,
        tool_schema=schema,
        proposed_tool_call={"name": "calculator", "arguments": {"expression": "2+2", "precision": True}},
    )
    assert bad_type.action == "abstain" and bad_type.guard_triggered
    assert "type_mismatch" in (bad_type.guard_reason or "")

    # 3. Unpermitted tool -> blocked!
    denied = routers_mod.apply_safety_guards(
        tool_probs,
        abstain_threshold=0.40,
        tool_schema=schema,
        proposed_tool_call={"name": "shell_exec", "arguments": {"cmd": "rm -rf /"}},
    )
    assert denied.action == "abstain" and denied.guard_triggered
    assert "permission_denied" in (denied.guard_reason or "")

    # Cost-weighted utility heavily penalizes false tool calls relative to safe abstention
    u_safe = routers_mod.compute_routing_utility(["abstain", "generate"], ["retrieve", "generate"])
    u_reckless = routers_mod.compute_routing_utility(["tool", "generate"], ["retrieve", "generate"])
    assert u_safe["normalized_utility"] > u_reckless["normalized_utility"]
