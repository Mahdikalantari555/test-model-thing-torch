"""Contract tests for ``src/model/benchmarks.py``.

The harness is only useful if a reader can trust it, so these tests pin the
report shape, the skip path, the median-of-runs rule, the document split,
the contamination refusal, and both comparison guards. Everything runs on a
tiny in-process model; the full-size numbers belong to the harness.
"""

import json
import math
import time
from pathlib import Path

import torch

from src.model import benchmarks
from src.model.rtu import Model


# --------------------------------------------------------------------------- #
# Report writer and outcome vocabulary
# --------------------------------------------------------------------------- #
def _config(seed=7):
    return {"seed": seed, "chunk": 64, "dim": 8}


def test_report_round_trips_with_every_provenance_field(tmp_path):
    report = benchmarks.build_report(
        command="python -m src.cli bench --tokens 10",
        config=_config(),
        measurements={
            "params": benchmarks.measurement(1234, metric="params"),
            "rss": benchmarks.measurement(42.5, metric="rss_mb", artifacts=["x"]),
        },
        dependencies={"torch": "test"},
        hardware="test-cpu",
        commit="a" * 40,
    )
    path = benchmarks.write_report(report, tmp_path / "report.json")
    written = json.loads(open(path, encoding="utf-8").read())

    for field in ("schema_version", "commit", "command", "dependencies", "config",
                  "hardware", "measurements", "outcome"):
        assert field in written
    assert written["commit"] == "a" * 40
    assert written["hardware"] == "test-cpu"
    assert written["dependencies"] == {"torch": "test"}
    assert written["config"]["seed"] == 7
    assert written["measurements"]["params"]["value"] == 1234
    assert written["outcome"] in benchmarks.OUTCOMES
    assert written["schema_version"] == benchmarks.REPORT_SCHEMA_VERSION


def test_skipped_measurement_has_a_reason_and_no_value(tmp_path):
    measurements = {
        "throughput": benchmarks.skipped("no checkpoint supplied"),
        "params": benchmarks.measurement(10, metric="params"),
    }
    report = benchmarks.build_report(
        command="bench", config=_config(), measurements=measurements
    )
    skipped = report["measurements"]["throughput"]
    assert skipped["status"] == "skipped"
    assert "value" not in skipped
    assert skipped["reason"] == "no checkpoint supplied"
    assert report["outcome"] == "passed"


def test_outcome_vocabulary_is_closed():
    assert benchmarks.OUTCOMES == ("passed", "failed", "skipped", "blocked")
    try:
        benchmarks.build_report(
            command="bench", config={}, measurements={}, outcome="green"
        )
    except ValueError as exc:
        assert "green" in str(exc)
    else:  # pragma: no cover - the raise is the assertion
        raise AssertionError("an unknown outcome must be rejected")


def test_all_skipped_run_is_skipped_and_failure_outranks_skip():
    only_skips = {"a": benchmarks.skipped("why")}
    assert benchmarks.resolve_outcome(only_skips) == "skipped"
    assert benchmarks.resolve_outcome(
        {"a": benchmarks.measurement(1)}
    ) == "passed"
    assert benchmarks.resolve_outcome(
        {"a": benchmarks.measurement(1)}, failed=True
    ) == "failed"
    assert benchmarks.resolve_outcome(
        {"a": benchmarks.skipped("why")}, blocked=True
    ) == "blocked"


# --------------------------------------------------------------------------- #
# Throughput
# --------------------------------------------------------------------------- #
def test_throughput_reports_median_of_recorded_runs_and_every_run():
    calls = {"count": 0}

    def run():
        calls["count"] += 1
        time.sleep(0.001)

    result = benchmarks.measure_throughput(run, 100, runs=3, warmup=1, path="sequential")
    assert result["status"] == "passed"
    assert len(result["runs"]) == 3
    rates = [record["tokens_per_second"] for record in result["runs"]]
    assert math.isclose(result["value"], sorted(rates)[1], rel_tol=1e-9)
    assert result["tokens"] == 100
    assert result["path"] == "sequential"
    for key in ("elapsed_seconds", "batch_size", "sequence_length", "thread_count", "hardware"):
        assert key in result


def test_throughput_of_nothing_is_skipped():
    result = benchmarks.measure_throughput(lambda: None, 0, runs=2)
    assert result["status"] == "skipped"
    assert "value" not in result


# --------------------------------------------------------------------------- #
# Parameter and memory accounting
# --------------------------------------------------------------------------- #
def test_parameter_count_equals_trainable_elements():
    torch.manual_seed(0)
    model = Model(dim=8, layers=2, spread=4, vocab_size=16)
    expected = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert benchmarks.count_trainable_parameters(model) == expected
    assert expected == model.count()


def test_resident_memory_names_the_loaded_artifacts():
    result = benchmarks.measure_resident_memory(["rtu.safetensors", "knowledge.db"])
    assert result["status"] == "passed"
    assert result["value"] > 0
    assert result["artifacts"] == ["rtu.safetensors", "knowledge.db"]
    assert result["metric"] == "rss_mb"


# --------------------------------------------------------------------------- #
# Held-out split and perplexity
# --------------------------------------------------------------------------- #
def _documents(count=12):
    return {
        f"doc-{index}": (f"document {index} " * 40).encode()
        for index in range(count)
    }


def test_document_split_is_deterministic_and_disjoint():
    documents = _documents()
    first = benchmarks.document_split(documents, held_out_fraction=0.5, seed=3)
    second = benchmarks.document_split(documents, held_out_fraction=0.5, seed=3)
    assert first == second
    assert set(first["train_ids"]).isdisjoint(first["held_out_ids"])
    assert set(first["train_ids"]) | set(first["held_out_ids"]) == set(documents)

    # Insertion order must not change the split or the document order.
    shuffled = {key: documents[key] for key in reversed(list(documents))}
    third = benchmarks.document_split(shuffled, held_out_fraction=0.5, seed=3)
    assert third["train_ids"] == first["train_ids"]
    assert third["held_out_ids"] == first["held_out_ids"]


def test_perplexity_is_reported_with_its_split():
    documents = _documents()
    result = benchmarks.measure_held_out_perplexity(
        lambda held: (sum(len(data) for data in held.values()) * 1.0,
                      sum(len(data) for data in held.values())),
        documents,
        held_out_fraction=0.5,
        seed=5,
    )
    assert result["status"] == "passed"
    assert math.isclose(result["value"], math.e, rel_tol=1e-9)
    for key in ("corpus_id", "bytes_evaluated", "split_rule"):
        assert result[key]
    assert result["bytes_evaluated"] > 0


def test_contaminated_split_refuses_to_report_a_number():
    shared = b"the same passage appears on both sides of the split. " * 20
    documents = {f"doc-{index}": shared for index in range(8)}
    result = benchmarks.measure_held_out_perplexity(
        lambda held: (1.0, 10),
        documents,
        held_out_fraction=0.5,
        seed=1,
    )
    assert result["status"] == "skipped"
    assert "value" not in result
    assert "training corpus" in result["reason"]
    assert result.get("contaminated_documents")


def test_find_contamination_names_the_overlapping_document():
    held = {"held-doc": b"unique held-out bytes that also appear elsewhere"}
    train = b"prefix " + held["held-doc"] + b" suffix"
    offenders = benchmarks.find_contamination(train, held, window=16)
    assert offenders and offenders[0]["document_id"] == "held-doc"
    assert benchmarks.find_contamination(b"nothing in common here", held, window=16) == []


# --------------------------------------------------------------------------- #
# Reference path comparison and its two guards
# --------------------------------------------------------------------------- #
def test_ratio_is_computed_from_the_two_reported_timings():
    def sequential():
        time.sleep(0.004)

    def chunked():
        time.sleep(0.001)

    result = benchmarks.compare_reference_paths(
        sequential, chunked, 10, runs=3, warmup=0, load={"load_ratio": 0.0}
    )
    assert result["status"] == "passed"
    expected = result["sequential"]["elapsed_seconds"] / result["chunked"]["elapsed_seconds"]
    assert math.isclose(result["value"], expected, rel_tol=1e-6)
    assert result["hardware"]
    assert result["sequence_length"] == benchmarks.SPEED_FACTOR_SEQUENCE_LENGTH


def test_high_variance_measurement_is_skipped_without_a_ratio():
    durations = iter([0.004, 0.040, 0.004, 0.040, 0.004, 0.040])

    def jittery():
        time.sleep(next(durations, 0.004))

    result = benchmarks.compare_reference_paths(
        jittery, lambda: time.sleep(0.001), 10, runs=3, warmup=0,
        load={"load_ratio": 0.0},
    )
    assert result["status"] == "skipped"
    assert "value" not in result
    assert "variation" in result["reason"]
    assert result["cv"] > benchmarks.VARIANCE_THRESHOLD
    # The documented threshold is not lowered to make the machine pass.
    assert benchmarks.VARIANCE_THRESHOLD == 0.10


def test_busy_machine_is_skipped_with_the_measured_load():
    result = benchmarks.compare_reference_paths(
        lambda: time.sleep(0.001), lambda: time.sleep(0.001), 10,
        runs=2, warmup=0, load={"load_ratio": 0.95, "load_1m": 7.6, "cpu_count": 8},
    )
    assert result["status"] == "skipped"
    assert "value" not in result
    assert "load" in result["reason"]
    assert result["load"]["load_ratio"] == 0.95
    assert benchmarks.LOAD_THRESHOLD == 0.40


def test_speed_factor_threshold_matches_the_parallel_scan_requirement():
    spec = (
        Path(__file__).resolve().parents[1]
        / "openspec" / "specs" / "training" / "parallel-scan" / "spec.md"
    ).read_text(encoding="utf-8")
    assert "at least 10" in spec
    assert "length 512" in spec
    assert benchmarks.SPEED_FACTOR_THRESHOLD == 10.0
    assert benchmarks.SPEED_FACTOR_SEQUENCE_LENGTH == 512


def test_held_out_bpb_ttft_int8_decode_and_equal_budget_tables():
    import math
    import torch
    import torch.nn as nn

    train_docs = {"tr1": "Synthetic aperture radar measures interferometric backscatter." * 4}
    test_docs = {"te1": "Magnetotelluric sounding characterizes crustal conductivity." * 4}
    total_bytes = len(test_docs["te1"].encode("utf-8"))

    # Both a byte tokenizer (N tokens = total_bytes) and BPE tokenizer (N/3 tokens)
    # with identical total NLL nats MUST report identical cross-tokenizer BPB!
    bpb_byte = benchmarks.measure_held_out_bpb(
        train_docs, test_docs, lambda _: (total_bytes * math.log(2.0) * 1.5, total_bytes), vocab_size=256, tokenizer_id="bytes"
    )
    bpb_bpe = benchmarks.measure_held_out_bpb(
        train_docs, test_docs, lambda _: (total_bytes * math.log(2.0) * 1.5, total_bytes // 3), vocab_size=4096, tokenizer_id="bpe"
    )
    assert bpb_byte["status"] == "passed" and bpb_bpe["status"] == "passed"
    assert abs(bpb_byte["value"] - 1.5) < 1e-6
    assert abs(bpb_byte["value"] - bpb_bpe["value"]) < 1e-6

    # Load guard skips TTFT cleanly when machine load exceeds 0.40
    ttft_skip = benchmarks.measure_prefill_ttft(
        {128: lambda: None}, load={"load_1m": 4.0, "cpu_count": 4, "load_ratio": 0.90}
    )
    assert ttft_skip["status"] == "skipped" and "value" not in ttft_skip

    # Dynamic INT8 decode helper
    tiny = nn.Sequential(nn.Linear(32, 32), nn.SiLU(), nn.Linear(32, 32))
    x = torch.randn(1, 32)
    int8_res = benchmarks.measure_int8_decode_speed(
        tiny,
        lambda qm: (lambda: [qm(x) for _ in range(8)]),
        tokens_per_run=8,
        bytes_per_token=3.2,
        runs=3,
        warmup=1,
        variance_threshold=0.95,
        load={"load_1m": 0.1, "cpu_count": 4, "load_ratio": 0.025},
    )
    assert int8_res["status"] in {"passed", "skipped"}
    if int8_res["status"] == "passed":
        assert int8_res["value"]["int8_bytes_per_second"] > int8_res["value"]["int8_tokens_per_second"]

    tables = benchmarks.build_equal_budget_tables(
        [
            {"arm_id": "A_bytes", "core_id": "C4", "tokenizer_id": "bytes", "vocab_size": 256, "equal_byte_bpb": 1.42, "equal_clock_bpb": 1.55},
            {"arm_id": "A_bpe", "core_id": "C4", "tokenizer_id": "bpe", "vocab_size": 4096, "equal_byte_bpb": 1.38, "equal_clock_bpb": 1.31},
        ]
    )
    assert tables["equal_byte_exposure"][0]["arm_id"] == "A_bpe"
    assert tables["equal_wall_clock"][0]["arm_id"] == "A_bpe"

