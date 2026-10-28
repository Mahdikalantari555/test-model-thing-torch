"""Contract tests for the versioned 20-task continual-learning benchmark.

Invariants (see openspec/changes/delta-plastic-memory/tasks.md 4.1-4.4 and
post-merge-completion 5.6):
- the benchmark protocol (seeds, task order, splits, scoring) is versioned and
  reproducible;
- every adapter sees the identical stream in the identical order;
- memory loss is the mean per-task drop between best and final score, which is
  pinned here against a hand-computed fixture;
- targets are compared to measurements and never hard-coded as achieved.
"""

import json

import pytest

from src.model.continual_bench import (
    BENCHMARK_VERSION,
    BenchmarkConfig,
    DeltaSystem,
    EWCSystem,
    build_task_stream,
    compute_memory_loss,
    run_benchmark,
)


TINY = BenchmarkConfig(tasks=4, patterns_per_task=8, eval_per_task=3, dim=16)


# --------------------------------------------------------------------------- #
# Protocol
# --------------------------------------------------------------------------- #
def test_config_records_version_seeds_and_splits():
    record = TINY.to_dict()
    assert record["version"] == BENCHMARK_VERSION
    assert record["seeds"] == {"data": 7, "task_order": 11}
    assert record["split"] == {"train_per_task": 8, "eval_per_task": 3,
                                "eval_pairs_with_train": True}
    assert record["dim"] == 16


def test_task_stream_is_reproducible_and_order_is_not_identity():
    first = build_task_stream(TINY)
    second = build_task_stream(TINY)
    assert len(first) == TINY.tasks
    assert [t["task_id"] for t in first] == [t["task_id"] for t in second]
    assert [t["task_id"] for t in first] != list(range(TINY.tasks))
    for left, right in zip(first, second):
        for key in ("train_keys", "train_values", "eval_keys", "eval_values"):
            assert left[key].shape == right[key].shape
            assert bool((left[key] == right[key]).all())


def test_task_splits_are_disjoint_and_sized():
    for task in build_task_stream(TINY):
        assert task["train_keys"].shape == (TINY.patterns_per_task, TINY.dim)
        assert task["eval_keys"].shape == (TINY.eval_per_task, TINY.dim)
        # Every row is unit length, as the adapters assume.
        assert float(task["train_keys"].norm(dim=1).min()) == pytest.approx(1.0, abs=1e-5)
        # Train and evaluation rows are distinct draws: no shared row.
        train_rows = {tuple(row.tolist()) for row in task["train_keys"]}
        eval_rows = {tuple(row.tolist()) for row in task["eval_keys"]}
        assert train_rows.isdisjoint(eval_rows)


# --------------------------------------------------------------------------- #
# Memory-loss definition
# --------------------------------------------------------------------------- #
def test_memory_loss_matches_hand_computed_fixture():
    # Task 0: 0.9 -> 0.5 (drop 0.4), task 1: 0.8 -> 0.8 (drop 0.0),
    # task 2: 0.7 -> 0.2 (drop 0.5)  =>  mean 0.3
    best = {0: 0.9, 1: 0.8, 2: 0.7}
    final = {0: 0.5, 1: 0.8, 2: 0.2}
    assert compute_memory_loss(best, final) == pytest.approx(0.3)


def test_memory_loss_ignores_improvement_and_is_zero_without_degradation():
    assert compute_memory_loss({0: 0.4}, {0: 0.9}) == 0.0
    assert compute_memory_loss({}, {}) == 0.0


def test_memory_loss_requires_best_scores():
    with pytest.raises(ValueError, match="missing best score"):
        compute_memory_loss({0: 0.5}, {0: 0.5, 1: 0.2})


# --------------------------------------------------------------------------- #
# Systems
# --------------------------------------------------------------------------- #
def test_delta_and_sparse_systems_train_and_read_back():
    tasks = build_task_stream(TINY)
    dense = DeltaSystem(TINY, sparse=False)
    for task in tasks[:2]:
        dense.train_task(task)
    score = dense.score(tasks[0]["eval_keys"], tasks[0]["eval_values"])
    assert -1.0 <= score <= 1.0
    assert len(dense.update_latencies) == 2 * TINY.patterns_per_task
    # The dense adapter activates every dimension; sparse codes do not.
    assert dense.active_set(tasks[0]["eval_keys"]).all()


def test_sparse_system_reports_a_sparse_active_set():
    tasks = build_task_stream(TINY)
    sparse = DeltaSystem(TINY, sparse=True)
    sparse.train_task(tasks[0])
    masks = sparse.active_set(tasks[0]["eval_keys"])
    # Sparse codes live in the 2048-D latent space, not the 384-D embedding.
    assert masks.shape == (TINY.eval_per_task, 2048)
    assert not masks.all()
    assert int(masks.sum(dim=1).max()) <= 128
    assert sparse.score(tasks[0]["eval_keys"], tasks[0]["eval_values"]) <= 1.000001


def test_ewc_reference_trains_under_the_same_budget():
    tasks = build_task_stream(TINY)
    ewc = EWCSystem(TINY)
    ewc.train_task(tasks[0])
    assert len(ewc.update_latencies) == TINY.patterns_per_task
    assert ewc.fisher.abs().sum() > 0
    assert -1.0 <= ewc.score(tasks[0]["eval_keys"], tasks[0]["eval_values"]) <= 1.0


# --------------------------------------------------------------------------- #
# Report contract
# --------------------------------------------------------------------------- #
def test_benchmark_report_contains_all_systems_and_task_detail():
    results = run_benchmark(TINY)
    assert results["tasks"] == TINY.tasks
    assert results["config"]["version"] == BENCHMARK_VERSION
    for name in ("delta_rtu", "sparse_delta", "ewc"):
        system = results[name]
        assert system["system"] == name
        assert len(system["final_scores"]) == TINY.tasks
        assert len(system["history"]) == TINY.tasks
        assert system["memory_loss"] >= 0.0
        assert set(system["per_task_drop"]) == set(system["final_scores"])
    summary = results["summary"]
    assert summary["delta_memory_loss"] == results["delta_rtu"]["memory_loss"]
    assert summary["delta_target_met"] == (summary["delta_memory_loss"] < 0.10)
    assert summary["ewc_reference_target"] == 0.24
    assert "interference" in summary and "target_met" in summary["interference"]


def test_domain_retention_matrix_and_contradiction_updates():
    from src.model.continual_bench import (
        evaluate_contradiction_updates,
        evaluate_domain_retention_matrix,
    )
    from src.model.cores.memory_arms import build_memory_arm
    from src.model.cores.param_matcher import solve_core_config
    from src.model.cores.registry import build_candidate_lm
    from src.model.cores.tokenizer import build_tokenizer
    from src.model.tournament.workloads import load_tournament_workloads

    bundle = load_tournament_workloads()
    tok = build_tokenizer("bytes", vocab_size=256)
    lm = build_candidate_lm(solve_core_config("C4", tier_or_budget=250_000, vocab_size=256), chunk_size=8)
    m3 = build_memory_arm("M3", lm, tok)

    ret = evaluate_domain_retention_matrix(m3, bundle.continual_domains)
    assert len(ret["retention_matrix"]) == 5
    assert all(len(row) == 5 for row in ret["retention_matrix"])
    assert ret["average_final_accuracy"] >= 0.75
    assert ret["backward_forgetting"] <= 0.25

    contra = evaluate_contradiction_updates(m3, bundle.contradiction_pairs)
    assert contra["contradiction_update_accuracy"] == 1.0
    assert contra["stale_answer_rate"] == 0.0



def test_report_is_json_serialisable():
    results = run_benchmark(TINY)
    payload = json.dumps(results)
    assert '"delta_rtu"' in payload
