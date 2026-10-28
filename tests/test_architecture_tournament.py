"""Tests for benchmark workloads (`workloads.py`), statistics (`stats.py`), and 4-stage funnel (`funnel.py`)."""

from __future__ import annotations

import pytest

import src.model.tournament.funnel as funnel_mod
import src.model.tournament.stats as stats_mod
import src.model.tournament.workloads as workloads_mod


def test_workloads_seed_reproducibility_zero_contamination_and_offline_block() -> None:
    b1 = workloads_mod.load_tournament_workloads(seed=1337)
    b2 = workloads_mod.load_tournament_workloads(seed=1337)
    assert b1.status == "ready"
    assert b1.dataset_source == "fixture"
    assert set(b1.mqar_splits.keys()) == {4, 16, 32, 64}
    assert len(b1.continual_domains) == 5
    assert len(b1.contradiction_pairs) >= 4
    assert len(b1.grammar_probes) >= 32

    s1, p1 = workloads_mod.mqar_example_to_lm_sequence(b1.mqar_splits[16]["eval"][0])
    s2, p2 = workloads_mod.mqar_example_to_lm_sequence(b2.mqar_splits[16]["eval"][0])
    assert s1 == s2 and p1 == p2

    c_seq, c_pos = workloads_mod.copy_example_to_lm_sequence(b1.copy_splits["eval"][0])
    assert len(c_pos) == len(b1.copy_splits["eval"][0].source_tokens)
    assert len(c_seq) > len(c_pos)

    blocked = workloads_mod.load_tournament_workloads(require_real_data=True)
    assert blocked.status == "blocked"
    assert blocked.dataset_source == "missing_real"
    assert blocked.reason


def test_funnel_lr_sweep_bootstrap_stage1_stage2_stage3_and_stage4_loo() -> None:
    # 1. 3-point LR micro-sweep ignores NaNs and picks minimum val_loss
    lr_res = stats_mod.select_best_lr(
        lambda lr: {"val_loss": float("nan") if lr > 2e-3 else abs(lr - 1e-3) + 1.2}
    )
    assert lr_res["selected_lr"] == 1e-3

    # 2. 10,000-resample paired bootstrap CI
    ci = stats_mod.paired_bootstrap_ci(
        [0.95] * 30 + [0.85] * 10,
        [0.50] * 30 + [0.40] * 10,
        arm_a="C4",
        arm_b="C0",
        num_resamples=10_000,
    )
    assert ci.significant_95 and ci.ci_low > 0.35

    # 3. Stage-1 falsification gates + Stage-2 ranking
    cores = [
        {
            "arm_id": "C0",
            "complexity_class": "constant_vector",
            "mqar_4_acc": 0.45,
            "mqar_16_acc": 0.25,
            "mqar_32_acc": 0.15,
            "mqar_64_acc": 0.10,
            "copy_acc": 0.30,
            "held_out_bpb": 2.15,
            "state_bytes_at_4k": 4096,
            "per_item_mqar_scores": [0.25] * 20,
        },
        {
            "arm_id": "C4",
            "complexity_class": "constant_matrix",
            "mqar_4_acc": 0.98,
            "mqar_16_acc": 0.94,
            "mqar_32_acc": 0.89,
            "mqar_64_acc": 0.84,
            "copy_acc": 0.96,
            "held_out_bpb": 1.38,
            "grammar_acc": 0.82,
            "extrapolation_bpb_ratio_4k_vs_256": 1.03,
            "state_bytes_at_4k": 32768,
            "per_item_mqar_scores": [0.95] * 20,
        },
        {
            "arm_id": "C2",
            "complexity_class": "constant_vector",
            "mqar_4_acc": 0.90,
            "mqar_16_acc": 0.78,
            "mqar_32_acc": 0.62,
            "mqar_64_acc": 0.48,
            "copy_acc": 0.82,
            "held_out_bpb": 1.47,
            "grammar_acc": 0.76,
            "extrapolation_bpb_ratio_4k_vs_256": 1.08,
            "state_bytes_at_4k": 16384,
            "per_item_mqar_scores": [0.78] * 20,
        },
        {
            "arm_id": "C6",
            "complexity_class": "linear_kv",
            "mqar_4_acc": 0.99,
            "mqar_16_acc": 0.97,
            "mqar_32_acc": 0.95,
            "mqar_64_acc": 0.92,
            "copy_acc": 0.98,
            "held_out_bpb": 1.36,
            "state_bytes_at_4k": 4_194_304,
            "per_item_mqar_scores": [0.97] * 20,
        },
    ]
    objs = [
        {"arm_id": "O1", "held_out_bpb": 1.40, "probe_macro_f1": 0.80, "train_tokens_per_sec": 1200.0},
        {"arm_id": "O2", "held_out_bpb": 1.41, "probe_macro_f1": 0.79, "train_tokens_per_sec": 1050.0},
        {"arm_id": "O3", "held_out_bpb": 1.395, "probe_macro_f1": 0.805, "train_tokens_per_sec": 950.0},
        {"arm_id": "O4_online_rtrl", "held_out_bpb": 1.48, "train_tokens_per_sec": 800.0, "supports_matrix_cores": False},
    ]
    toks = [
        {"arm_id": "Tok_bytes_256", "equal_byte_bpb": 1.46, "equal_clock_bpb": 1.58, "int8_bytes_per_second": 420.0},
        {"arm_id": "Tok_bpe_4096", "equal_byte_bpb": 1.38, "equal_clock_bpb": 1.34, "int8_bytes_per_second": 1350.0},
    ]
    decs = [
        {"arm_id": "D1", "in_domain_macro_f1": 0.68, "shifted_macro_f1": 0.61, "oos_f1": 0.55, "adaptive_ece": 0.14, "aurc": 0.22},
        {"arm_id": "D3", "in_domain_macro_f1": 0.89, "shifted_macro_f1": 0.84, "oos_f1": 0.86, "adaptive_ece": 0.05, "aurc": 0.08},
    ]
    routers = [
        {"arm_id": "R1", "normalized_utility": 0.54, "false_tool_rate": 0.08, "adaptive_ece": 0.16, "p50_latency_ms": 0.3},
        {"arm_id": "R4", "normalized_utility": 0.88, "false_tool_rate": 0.00, "adaptive_ece": 0.05, "p50_latency_ms": 0.6},
    ]
    mems = [
        {"arm_id": "M0", "average_final_accuracy": 0.20, "backward_forgetting": 0.65, "contradiction_update_accuracy": 0.25, "stale_answer_rate": 0.50},
        {"arm_id": "M3", "average_final_accuracy": 0.96, "backward_forgetting": 0.02, "contradiction_update_accuracy": 1.00, "stale_answer_rate": 0.00, "total_footprint_bytes": 32768},
    ]

    s1 = funnel_mod.evaluate_stage1_gates(
        core_records=cores,
        objective_records=objs,
        tokenizer_records=toks,
        decision_records=decs,
        router_records=routers,
        memory_records=mems,
    )
    assert "C0" not in s1["surviving_by_track"]["track1_core"]
    assert "C4" in s1["surviving_by_track"]["track1_core"]
    # O2 and O3 both fail ΔBPB >= 0.015 vs O1, leaving O1 as clean survivor!
    assert s1["surviving_by_track"]["track2_objective"] == ["O1"]

    s2 = funnel_mod.evaluate_stage2_tracks(
        stage1_result=s1,
        core_records=cores,
        objective_records=objs,
        tokenizer_records=toks,
        decision_records=decs,
        router_records=routers,
        memory_records=mems,
    )
    assert s2["track1_core"]["winner_id"] == "C4"
    assert s2["track2_tokenizer"]["winner_id"] == "Tok_bpe_4096"
    assert s2["track3_decision"]["winner_id"] == "D3"
    assert s2["track4_router"]["winner_id"] == "R4"
    assert s2["track5_memory"]["winner_id"] == "M3"

    # 4. Stage-3 Shared-vs-Split interference gate: test BOTH unified-win and split-fallback!
    s3_unified = funnel_mod.evaluate_stage3_interference(
        standalone_lm_bpb=1.38,
        standalone_decision_macro_f1=0.88,
        unified_interleaved_bpb=1.39,
        unified_interleaved_macro_f1=0.875,
        unified_summed_bpb=1.43,
        unified_summed_macro_f1=0.84,
        split_d4_bpb=1.44,
        split_d4_macro_f1=0.83,
        unified_total_params=2_000_000,
        split_d4_total_params=1_995_000,
    )
    assert s3_unified["selected_architecture_mode"] == "unified"
    assert s3_unified["selected_batching_mode"] == "interleaved"

    s3_split = funnel_mod.evaluate_stage3_interference(
        standalone_lm_bpb=1.38,
        standalone_decision_macro_f1=0.88,
        unified_interleaved_bpb=1.52,  # > +5% degradation -> hard reject to split_specialist!
        unified_interleaved_macro_f1=0.79,
        unified_summed_bpb=1.55,
        unified_summed_macro_f1=0.77,
        split_d4_bpb=1.42,
        split_d4_macro_f1=0.86,
        unified_total_params=2_000_000,
        split_d4_total_params=2_010_000,
    )
    assert s3_split["selected_architecture_mode"] == "split_specialist"

    # 5. Stage-4 Leave-One-Out complexity pruning waterfall
    s4 = funnel_mod.evaluate_stage4_loo_pruning(
        full_stack_utility=0.860,
        ablations=[
            {"component": "short_conv4", "simpler_fallback": "conv_off", "ablated_utility": 0.835},  # +0.025 -> kept!
            {"component": "aux_ema_predictor", "simpler_fallback": "O1_pure_ce", "ablated_utility": 0.859},  # +0.001 < 0.005 -> pruned!
        ],
    )
    assert "short_conv4" in s4["kept_components"]
    assert s4["pruned_components"]["aux_ema_predictor"] == "O1_pure_ce"


def test_large_workloads_item_counts_and_equal_non_embedding_param_matching(tmp_path):
    """Verify scale='large' suites satisfy >=1,000 item rules and equal-non-embedding tokenizer matching."""
    from src.model.cores.param_matcher import solve_core_config_non_embedding
    from src.model.cores.registry import build_candidate_lm
    from src.model.tournament.large_workloads import compute_corpus_unigram_bpb
    from src.model.tournament.workloads import load_tournament_workloads

    w = load_tournament_workloads(seed=1337, scale="large", fixture_dir=tmp_path)
    assert w.item_counts["mqar_eval_queries_per_kv"] >= 1000
    assert w.item_counts["decision_test_total"] >= 1000
    assert w.item_counts["routing_test_eval"] >= 1000
    assert w.item_counts["continual_facts_total"] >= 1000
    assert w.item_counts["continual_distractors_total"] >= 200
    assert w.item_counts["contradiction_pairs"] >= 200

    bpb_stats = compute_corpus_unigram_bpb(
        {"d1": "once upon a time there was a little cat who lived in a small house " * 8},
        {"e1": "the little cat walked into the small house one sunny afternoon " * 4},
    )
    assert 3.0 <= bpb_stats["byte_unigram_bpb"] <= 8.0

    cfg_256 = solve_core_config_non_embedding(
        "C4", target_non_embedding_params=350_000, vocab_size=256
    )
    cfg_4096 = cfg_256.with_vocab_size(4096)
    assert cfg_256.non_embedding_params == cfg_4096.non_embedding_params
    m_256 = build_candidate_lm(cfg_256)
    m_4096 = build_candidate_lm(cfg_4096)
    ne_256 = sum(p.numel() for p in m_256.parameters()) - m_256.lm_head.weight.numel()
    ne_4096 = sum(p.numel() for p in m_4096.parameters()) - m_4096.lm_head.weight.numel()
    assert ne_256 == ne_4096 == cfg_256.non_embedding_params

