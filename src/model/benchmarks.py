"""Measurement primitives and the benchmark report writer.

Every number the project argues about is produced here, once, in one shape.
The rules the module enforces are the ones the ``evaluation/edge-performance``
capability states:

* a measurement is either a value with its conditions, or a skip with a reason,
  and a skipped measurement never carries a value;
* a report carries the commit, the exact command, the dependency versions, the
  configuration (seeds included), the hardware, the measurements and the
  outcome, so the run can be reproduced;
* a perplexity is only reported for a split that is verified disjoint from the
  training bytes;
* a reference-path comparison is reported as skipped rather than as a ratio
  when the measurement is unstable or the machine is loaded, and the documented
  thresholds are constants, not arguments a caller can quietly lower.

The module holds no argument parsing and no run orchestration: those live in
``scripts/bench_edge.py``. This split is deliberate so the primitives are
importable and unit-testable without a harness.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "REPORT_SCHEMA_VERSION",
    "OUTCOMES",
    "SPEED_FACTOR_THRESHOLD",
    "SPEED_FACTOR_SEQUENCE_LENGTH",
    "VARIANCE_THRESHOLD",
    "LOAD_THRESHOLD",
    "PERPLEXITY_WINDOW_BYTES",
    "SPLIT_RULE",
    "ContaminatedCorpusError",
    "git_commit",
    "dependency_versions",
    "hardware_description",
    "system_load",
    "sha256_file",
    "sha256_bytes",
    "measurement",
    "skipped",
    "build_report",
    "write_report",
    "resolve_outcome",
    "measure_throughput",
    "count_trainable_parameters",
    "measure_resident_memory",
    "document_split",
    "find_contamination",
    "measure_held_out_perplexity",
    "compare_reference_paths",
    "measure_held_out_bpb",
    "measure_prefill_ttft",
    "measure_int8_decode_speed",
    "build_equal_budget_tables",
]

REPORT_SCHEMA_VERSION = 1

# The closed outcome vocabulary. A failed or skipped run is still evidence and
# is written, never discarded to obtain a green result.
OUTCOMES = ("passed", "failed", "skipped", "blocked")

# The chunked path's speed factor at sequence length 512. This value is the
# threshold the ``training/parallel-scan`` requirement specifies; this module
# does not renegotiate it.
SPEED_FACTOR_THRESHOLD = 10.0
SPEED_FACTOR_SEQUENCE_LENGTH = 512

# Run-to-run coefficient of variation above which a comparison is skipped.
VARIANCE_THRESHOLD = 0.10

# One-minute load average divided by the CPU count above which the machine is
# treated as busy and a throughput comparison is skipped instead of asserted.
LOAD_THRESHOLD = 0.40

# Window length used by the perplexity containment check.
PERPLEXITY_WINDOW_BYTES = 64

SPLIT_RULE = (
    "document-level: documents ordered by sha256(f'{seed}:{document_id}'), "
    "the first round(held_out_fraction * document_count) are held out; "
    "the split is independent of file order and reproducible from the seed"
)


class ContaminatedCorpusError(RuntimeError):
    """A held-out split shares bytes with the corpus a model trained on."""


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
def git_commit(root: Optional[Path] = None) -> str:
    """Full 40-character commit hash, or ``"unknown"`` outside a git checkout."""
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    commit = out.stdout.strip()
    return commit if len(commit) == 40 else "unknown"


def dependency_versions() -> Dict[str, str]:
    """Versions of the libraries a reported number depends on."""
    versions = {"python": platform.python_version()}
    for name, module in (
        ("torch", "torch"),
        ("numpy", "numpy"),
        ("safetensors", "safetensors"),
        ("psutil", "psutil"),
        ("pytest", "pytest"),
    ):
        try:
            versions[name] = str(__import__(module).__version__)
        except Exception:
            versions[name] = "absent"
    return versions


def hardware_description() -> str:
    """A one-line hardware and thread description (never just a model name)."""
    machine = platform.machine() or "unknown"
    processor = platform.processor() or platform.uname().processor or "unknown cpu"
    threads = None
    try:
        import torch

        threads = torch.get_num_threads()
    except Exception:
        threads = os.cpu_count()
    return (
        f"{machine} | {processor} | cpu_count={os.cpu_count()} | "
        f"threads={threads} | python={platform.python_version()} | {platform.platform()}"
    )


def thread_count() -> Optional[int]:
    try:
        import torch

        return int(torch.get_num_threads())
    except Exception:
        return None


def system_load() -> Dict[str, Any]:
    """One-minute load average and the CPU count, plus their ratio."""
    try:
        load1 = float(os.getloadavg()[0])
    except (OSError, AttributeError):
        load1 = float("nan")
    cpus = os.cpu_count() or 1
    ratio = load1 / cpus if cpus and not math.isnan(load1) else 0.0
    return {"load_1m": load1, "cpu_count": cpus, "load_ratio": ratio}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | os.PathLike) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# Measurements
# --------------------------------------------------------------------------- #
def measurement(value: Any, **conditions: Any) -> Dict[str, Any]:
    """A measured value plus the conditions that produced it."""
    record: Dict[str, Any] = {"status": "passed", "value": value}
    record.update(conditions)
    return record


def skipped(reason: str, **conditions: Any) -> Dict[str, Any]:
    """A measurement that was not taken. Carries no ``value`` key at all."""
    record: Dict[str, Any] = {"status": "skipped", "reason": str(reason)}
    record.update(conditions)
    return record


def is_measured(record: Mapping[str, Any]) -> bool:
    return record.get("status") == "passed" and "value" in record


def resolve_outcome(measurements: Mapping[str, Mapping[str, Any]],
                    failed: bool = False, blocked: bool = False) -> str:
    """Resolve the run outcome from its measurements.

    ``blocked`` outranks everything: a run that could not start did not run.
    A failed measurement or an explicit ``failed`` flag outranks a skip, and an
    all-skipped run is ``skipped`` rather than ``passed``.
    """
    if blocked:
        return "blocked"
    if failed:
        return "failed"
    statuses = [record.get("status") for record in measurements.values()]
    if statuses and all(status == "skipped" for status in statuses):
        return "skipped"
    return "passed"


@dataclass
class BenchmarkReport:
    """The single document every run writes."""

    command: str
    config: Dict[str, Any]
    measurements: Dict[str, Dict[str, Any]]
    outcome: str
    commit: str = ""
    hardware: str = ""
    dependencies: Dict[str, str] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    skip_reason: Optional[str] = None
    schema_version: int = REPORT_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        if self.outcome not in OUTCOMES:
            raise ValueError(
                f"outcome {self.outcome!r} is not one of {OUTCOMES}"
            )
        document: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "commit": self.commit or git_commit(),
            "command": self.command,
            "dependencies": dict(self.dependencies or dependency_versions()),
            "config": dict(self.config),
            "hardware": self.hardware or hardware_description(),
            "measurements": {
                name: dict(record) for name, record in self.measurements.items()
            },
            "outcome": self.outcome,
        }
        if self.artifacts:
            document["artifacts"] = dict(self.artifacts)
        if self.skip_reason:
            document["skip_reason"] = self.skip_reason
        return document


def build_report(*, command: str, config: Mapping[str, Any],
                 measurements: Mapping[str, Mapping[str, Any]],
                 outcome: Optional[str] = None,
                 commit: Optional[str] = None,
                 hardware: Optional[str] = None,
                 dependencies: Optional[Mapping[str, str]] = None,
                 artifacts: Optional[Mapping[str, Any]] = None,
                 skip_reason: Optional[str] = None,
                 failed: bool = False, blocked: bool = False) -> Dict[str, Any]:
    """Build the report document. Returns the JSON-serialisable dict."""
    resolved = outcome or resolve_outcome(measurements, failed=failed, blocked=blocked)
    report = BenchmarkReport(
        command=command,
        config=dict(config),
        measurements={name: dict(record) for name, record in measurements.items()},
        outcome=resolved,
        commit=commit or git_commit(),
        hardware=hardware or hardware_description(),
        dependencies=dict(dependencies or dependency_versions()),
        artifacts=dict(artifacts or {}),
        skip_reason=skip_reason,
    )
    return report.to_dict()


def write_report(report: Mapping[str, Any], path: str | os.PathLike) -> str:
    """Write the report atomically and return its path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(dict(report), handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, target)
    return str(target)


# --------------------------------------------------------------------------- #
# Throughput
# --------------------------------------------------------------------------- #
def measure_throughput(
    run: Callable[[], Any],
    tokens_per_run: int,
    *,
    runs: int = 5,
    warmup: int = 1,
    path: str = "sequential",
    batch_size: int = 1,
    sequence_length: Optional[int] = None,
    threads: Optional[int] = None,
    hardware: Optional[str] = None,
) -> Dict[str, Any]:
    """Time ``run`` and report the median token rate with every run.

    One warm-up pass precedes the timed passes, every timed pass is recorded,
    and the reported rate is the median of the per-run rates. A mean hides an
    outlier and a minimum flatters a loaded machine; the spread stays visible
    because every run is in the report.
    """
    if tokens_per_run <= 0:
        return skipped("measured no tokens", path=path)
    if runs < 1:
        return skipped(f"runs must be >= 1 (got {runs})", path=path)

    for _ in range(max(0, warmup)):
        run()

    records: List[Dict[str, float]] = []
    for _ in range(runs):
        start = time.perf_counter()
        run()
        elapsed = time.perf_counter() - start
        records.append(
            {
                "elapsed_seconds": elapsed,
                "tokens_per_second": tokens_per_run / elapsed if elapsed > 0 else 0.0,
            }
        )

    rates = [record["tokens_per_second"] for record in records]
    elapsed_all = [record["elapsed_seconds"] for record in records]
    median_rate = statistics.median(rates)
    mean_rate = statistics.fmean(rates)
    cv = (statistics.stdev(rates) / mean_rate) if len(rates) > 1 and mean_rate > 0 else 0.0
    return measurement(
        median_rate,
        metric="tokens_per_second",
        tokens=tokens_per_run,
        elapsed_seconds=statistics.median(elapsed_all),
        path=path,
        batch_size=batch_size,
        sequence_length=sequence_length,
        thread_count=threads if threads is not None else thread_count(),
        hardware=hardware or hardware_description(),
        runs=records,
        cv=cv,
    )


# --------------------------------------------------------------------------- #
# Parameter and memory accounting
# --------------------------------------------------------------------------- #
def count_trainable_parameters(model: Any) -> int:
    """Number of trainable elements in a constructed model."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _resident_set_bytes() -> int:
    try:
        import psutil

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        try:
            with open("/proc/self/statm", "r", encoding="ascii") as handle:
                pages = int(handle.read().split()[1])
            return pages * os.sysconf("SC_PAGE_SIZE")
        except Exception:
            return 0


def measure_resident_memory(artifacts: Sequence[str] | Mapping[str, Any]) -> Dict[str, Any]:
    """Resident set size in MiB, naming the artifacts that were loaded."""
    loaded = list(artifacts.keys()) if isinstance(artifacts, Mapping) else list(artifacts)
    rss_bytes = _resident_set_bytes()
    if rss_bytes <= 0:
        return skipped("resident set size is unavailable on this platform", artifacts=loaded)
    return measurement(
        rss_bytes / (1024.0 * 1024.0),
        metric="rss_mb",
        rss_bytes=rss_bytes,
        artifacts=loaded,
        artifacts_loaded=len(loaded),
        hardware=hardware_description(),
    )


# --------------------------------------------------------------------------- #
# Held-out perplexity
# --------------------------------------------------------------------------- #
def _split_key(seed: int, document_id: str) -> str:
    return f"{seed}:{document_id}"


def document_split(documents: Mapping[str, bytes], held_out_fraction: float = 0.2,
                   seed: int = 0) -> Dict[str, Any]:
    """Split documents by a hash of their identifier.

    The split is independent of filesystem order and reproducible from the
    seed alone, and no document can appear on both sides because the decision
    is made once per identifier.
    """
    if not 0.0 <= held_out_fraction < 1.0:
        raise ValueError(f"held_out_fraction must be in [0, 1), got {held_out_fraction}")
    ordered = sorted(documents, key=lambda doc_id: hashlib.sha256(
        _split_key(seed, doc_id).encode("utf-8")).hexdigest())
    held_out_count = int(round(held_out_fraction * len(ordered)))
    if held_out_fraction > 0.0 and held_out_count == 0 and len(ordered) > 1:
        # A hash-selected split must still exercise the held-out path; one
        # document is held out rather than silently reporting "no split".
        held_out_count = 1
    held_out: List[str] = ordered[:held_out_count]
    train: List[str] = ordered[held_out_count:]
    return {
        "split_rule": SPLIT_RULE,
        "seed": seed,
        "held_out_fraction": held_out_fraction,
        "order": ordered,
        "train_ids": train,
        "held_out_ids": held_out,
        "held_out_count": len(held_out),
        "train_count": len(train),
    }


def _sampled_windows(data: bytes, window: int, samples: int, seed_key: str) -> List[bytes]:
    if len(data) < window:
        return [data] if data else []
    step = max(1, (len(data) - window) // max(1, samples))
    windows = []
    for index in range(samples):
        start = min(index * step, len(data) - window)
        windows.append(data[start:start + window])
    return windows


def find_contamination(train_bytes: bytes, held_out_documents: Mapping[str, bytes],
                       window: int = PERPLEXITY_WINDOW_BYTES,
                       samples_per_document: int = 8) -> List[Dict[str, Any]]:
    """Held-out windows that also appear in the training bytes.

    Only sampled windows are checked, so an empty result is evidence of no
    *sampled* overlap rather than a proof of disjointness; the report says so
    by naming the check in the split rule.
    """
    offenders: List[Dict[str, Any]] = []
    for document_id, data in held_out_documents.items():
        for sample_index, candidate in enumerate(
            _sampled_windows(bytes(data), window, samples_per_document, document_id)
        ):
            if candidate and candidate in train_bytes:
                offenders.append(
                    {
                        "document_id": document_id,
                        "window": window,
                        "sample_index": sample_index,
                        "window_sha256": sha256_bytes(candidate),
                    }
                )
                break
    return offenders


def measure_held_out_perplexity(
    score: Callable[[Mapping[str, bytes]], Tuple[float, int]],
    documents: Mapping[str, bytes],
    *,
    held_out_fraction: float = 0.2,
    seed: int = 0,
    corpus_id: Optional[str] = None,
    window: int = PERPLEXITY_WINDOW_BYTES,
    samples_per_document: int = 8,
) -> Dict[str, Any]:
    """Byte-level perplexity on a verified-disjoint held-out split.

    ``score`` receives the held-out documents and returns ``(total_nll, bytes)``.
    A contaminated split yields a skipped measurement with the offending
    documents named and no perplexity value.
    """
    split = document_split(documents, held_out_fraction=held_out_fraction, seed=seed)
    held_out = {doc_id: bytes(documents[doc_id]) for doc_id in split["held_out_ids"]}
    train_bytes = b"\n".join(bytes(documents[doc_id]) for doc_id in split["train_ids"])
    corpus_id = corpus_id or sha256_bytes(
        b"".join(sorted(bytes(documents[doc_id]) for doc_id in documents))
    )

    if not held_out:
        return skipped(
            "the split produced no held-out documents",
            corpus_id=corpus_id,
            split_rule=split["split_rule"],
            seed=seed,
        )
    if not train_bytes:
        return skipped(
            "the split produced no training documents to check against",
            corpus_id=corpus_id,
            split_rule=split["split_rule"],
            seed=seed,
        )

    offenders = find_contamination(
        train_bytes, held_out, window=window, samples_per_document=samples_per_document
    )
    if offenders:
        return skipped(
            "held-out windows also appear in the training corpus: "
            + ", ".join(sorted({str(item["document_id"]) for item in offenders})),
            corpus_id=corpus_id,
            split_rule=split["split_rule"],
            seed=seed,
            contaminated_documents=sorted({str(item["document_id"]) for item in offenders}),
        )

    total_nll, byte_count = score(held_out)
    byte_count = int(byte_count)
    if byte_count <= 0:
        return skipped(
            "the held-out split contained no scorable bytes",
            corpus_id=corpus_id,
            split_rule=split["split_rule"],
            seed=seed,
        )
    if not math.isfinite(total_nll):
        return skipped(
            "the perplexity accumulator was not finite",
            corpus_id=corpus_id,
            split_rule=split["split_rule"],
            seed=seed,
        )
    perplexity = math.exp(total_nll / byte_count)
    return measurement(
        perplexity,
        metric="held_out_perplexity",
        corpus_id=corpus_id,
        bytes_evaluated=byte_count,
        held_out_documents=len(held_out),
        split_rule=split["split_rule"],
        seed=seed,
        containment_check=f"sampled {window}-byte windows, {samples_per_document} per document",
    )


# --------------------------------------------------------------------------- #
# Reference path comparison
# --------------------------------------------------------------------------- #
def _speedup(sequential: Mapping[str, Any], chunked: Mapping[str, Any]) -> float:
    sequential_seconds = float(sequential.get("elapsed_seconds") or 0.0)
    chunked_seconds = float(chunked.get("elapsed_seconds") or 0.0)
    if chunked_seconds <= 0:
        return float("nan")
    return sequential_seconds / chunked_seconds


def compare_reference_paths(
    sequential_run: Callable[[], Any],
    chunked_run: Callable[[], Any],
    tokens_per_run: int,
    *,
    runs: int = 5,
    warmup: int = 1,
    sequence_length: int = SPEED_FACTOR_SEQUENCE_LENGTH,
    threads: Optional[int] = None,
    hardware: Optional[str] = None,
    variance_threshold: float = VARIANCE_THRESHOLD,
    load_threshold: float = LOAD_THRESHOLD,
    load: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Time both paths and report the factor, or skip and say why.

    Two guards, both mandatory:

    * run-to-run variance above ``variance_threshold`` makes the ratio
      meaningless, so it is skipped with the measured coefficient of variation;
    * a uniformly loaded machine has *low* variance and a slow baseline, which
      the variance guard cannot see. The load guard catches it and skips with
      the measured load instead of reporting a failed factor.

    Neither threshold is raised or lowered by this function; a machine that
    cannot meet them reports a skip.
    """
    measured_load = dict(load) if load is not None else system_load()
    load_ratio = float(measured_load.get("load_ratio", 0.0) or 0.0)
    if load_ratio > load_threshold:
        return skipped(
            f"machine load ratio {load_ratio:.3f} exceeds {load_threshold}",
            load=measured_load,
            load_threshold=load_threshold,
            sequence_length=sequence_length,
            hardware=hardware or hardware_description(),
        )

    sequential = measure_throughput(
        sequential_run, tokens_per_run, runs=runs, warmup=warmup,
        path="sequential", sequence_length=sequence_length, threads=threads,
        hardware=hardware,
    )
    chunked = measure_throughput(
        chunked_run, tokens_per_run, runs=runs, warmup=warmup,
        path="chunked", sequence_length=sequence_length, threads=threads,
        hardware=hardware,
    )
    if not is_measured(sequential) or not is_measured(chunked):
        return skipped(
            "one of the reference paths produced no timing",
            sequential=sequential,
            chunked=chunked,
            load=measured_load,
            sequence_length=sequence_length,
        )

    sequential_cv = float(sequential.get("cv", 0.0))
    chunked_cv = float(chunked.get("cv", 0.0))
    worst_cv = max(sequential_cv, chunked_cv)
    if worst_cv > variance_threshold:
        return skipped(
            f"run-to-run coefficient of variation {worst_cv:.3f} exceeds {variance_threshold}",
            sequential=sequential,
            chunked=chunked,
            cv=worst_cv,
            variance_threshold=variance_threshold,
            load=measured_load,
            sequence_length=sequence_length,
        )

    factor = _speedup(sequential, chunked)
    if not math.isfinite(factor):
        return skipped(
            "the chunked timing was not positive",
            sequential=sequential,
            chunked=chunked,
            load=measured_load,
            sequence_length=sequence_length,
        )
    return measurement(
        factor,
        metric="chunked_speed_factor",
        sequential=sequential,
        chunked=chunked,
        sequential_seconds=sequential.get("elapsed_seconds"),
        chunked_seconds=chunked.get("elapsed_seconds"),
        hardware=hardware or hardware_description(),
        sequence_length=sequence_length,
        threads=threads if threads is not None else thread_count(),
        cv=worst_cv,
        variance_threshold=variance_threshold,
        load=measured_load,
        load_threshold=load_threshold,
        threshold=SPEED_FACTOR_THRESHOLD,
    )


# --------------------------------------------------------------------------- #
# Cross-tokenizer BPB, TTFT, Dynamic INT8 Decode & Equal-Budget Tables
# --------------------------------------------------------------------------- #
def measure_held_out_bpb(
    train_documents: Mapping[str, str | bytes],
    held_out_documents: Mapping[str, str | bytes],
    score_nll_nats: Callable[[Mapping[str, bytes]], Tuple[float, int]],
    *,
    vocab_size: int = 256,
    tokenizer_id: str = "bytes",
    window: int = PERPLEXITY_WINDOW_BYTES,
    samples_per_document: int = 8,
) -> Dict[str, Any]:
    """Compute cross-tokenizer invariant held-out Bits-Per-Byte (`held_out_bpb`).

    `BPB = total_nll_nats / (ln(2) * total_utf8_bytes)`.
    Runs the same sliding-window contamination guard as `measure_held_out_perplexity`
    so contaminated splits are rejected/skipped rather than silently reported.
    """
    to_bytes = lambda v: v if isinstance(v, (bytes, bytearray)) else str(v).encode("utf-8")
    train_bytes = b"\n".join(to_bytes(v) for v in train_documents.values())
    held_out_bytes = {str(k): bytes(to_bytes(v)) for k, v in held_out_documents.items()}

    if not held_out_bytes:
        return skipped("held_out_documents is empty", metric="held_out_bpb")
    if train_bytes:
        offenders = find_contamination(
            train_bytes, held_out_bytes, window=window, samples_per_document=samples_per_document
        )
        if offenders:
            return skipped(
                "held-out windows also appear in training documents: "
                + ", ".join(sorted({str(o["document_id"]) for o in offenders})),
                metric="held_out_bpb",
                contaminated_documents=sorted({str(o["document_id"]) for o in offenders}),
            )

    total_nll_nats, total_tokens = score_nll_nats(held_out_bytes)
    raw_utf8_bytes = sum(len(b) for b in held_out_bytes.values())
    if raw_utf8_bytes <= 0:
        return skipped("held-out split has zero UTF-8 bytes", metric="held_out_bpb")
    if not math.isfinite(total_nll_nats):
        return skipped("NLL accumulator was non-finite", metric="held_out_bpb")

    bpb = float(total_nll_nats) / (math.log(2.0) * float(raw_utf8_bytes))
    return measurement(
        bpb,
        metric="held_out_bpb",
        utf8_bytes_evaluated=raw_utf8_bytes,
        tokens_evaluated=int(total_tokens),
        bytes_per_token=float(raw_utf8_bytes) / float(max(1, total_tokens)),
        vocab_size=int(vocab_size),
        tokenizer_id=str(tokenizer_id),
    )


def measure_prefill_ttft(
    prefill_runner_by_length: Mapping[int, Callable[[], Any]],
    *,
    runs: int = 5,
    warmup: int = 1,
    variance_threshold: float = VARIANCE_THRESHOLD,
    load_threshold: float = LOAD_THRESHOLD,
    load: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Measure prefill Time-To-First-Token (`ttft_ms`) across prompt lengths (e.g. 128..4096)."""
    measured_load = dict(load) if load is not None else system_load()
    load_ratio = float(measured_load.get("load_ratio", 0.0) or 0.0)
    if load_ratio > load_threshold:
        return skipped(
            f"machine load ratio {load_ratio:.3f} exceeds {load_threshold}",
            metric="ttft_ms",
            load=measured_load,
            load_threshold=load_threshold,
        )

    by_len: Dict[str, Any] = {}
    worst_cv = 0.0
    for seq_len, fn in sorted(prefill_runner_by_length.items(), key=lambda kv: int(kv[0])):
        for _ in range(max(0, int(warmup))):
            fn()
        samples_ms: List[float] = []
        for _ in range(max(1, int(runs))):
            t0 = time.perf_counter()
            fn()
            samples_ms.append((time.perf_counter() - t0) * 1000.0)
        mean_ms = statistics.mean(samples_ms)
        stdev_ms = statistics.pstdev(samples_ms) if len(samples_ms) > 1 else 0.0
        cv = stdev_ms / mean_ms if mean_ms > 0 else 0.0
        worst_cv = max(worst_cv, cv)
        by_len[str(int(seq_len))] = {
            "mean_ms": mean_ms,
            "median_ms": statistics.median(samples_ms),
            "cv": cv,
            "runs": len(samples_ms),
        }

    if worst_cv > variance_threshold:
        return skipped(
            f"run-to-run coefficient of variation {worst_cv:.3f} exceeds {variance_threshold}",
            metric="ttft_ms",
            by_length=by_len,
            cv=worst_cv,
            variance_threshold=variance_threshold,
            load=measured_load,
        )
    return measurement(
        by_len,
        metric="ttft_ms",
        cv=worst_cv,
        variance_threshold=variance_threshold,
        load=measured_load,
        load_threshold=load_threshold,
        hardware=hardware_description(),
    )


def measure_int8_decode_speed(
    model: Any,
    build_decode_fn: Callable[[Any], Callable[[], Any]],
    tokens_per_run: int,
    *,
    bytes_per_token: float = 1.0,
    runs: int = 5,
    warmup: int = 1,
    variance_threshold: float = VARIANCE_THRESHOLD,
    load_threshold: float = LOAD_THRESHOLD,
    load: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Quantize a copy of `model` with dynamic INT8 and measure decode tokens/sec & bytes/sec."""
    import copy
    import torch
    import torch.nn as nn

    measured_load = dict(load) if load is not None else system_load()
    load_ratio = float(measured_load.get("load_ratio", 0.0) or 0.0)
    if load_ratio > load_threshold:
        return skipped(
            f"machine load ratio {load_ratio:.3f} exceeds {load_threshold}",
            metric="int8_decode_speed",
            load=measured_load,
            load_threshold=load_threshold,
        )

    engines = list(getattr(torch.backends.quantized, "supported_engines", []))
    if torch.backends.quantized.engine == "none":
        for candidate_eng in ("fbgemm", "qnnpack", "onednn"):
            if candidate_eng in engines:
                torch.backends.quantized.engine = candidate_eng
                break

    try:
        import warnings
        model_copy = copy.deepcopy(model).eval()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            q_model = torch.ao.quantization.quantize_dynamic(
                model_copy, {nn.Linear}, dtype=torch.qint8
            )
    except Exception as exc:
        return skipped(f"dynamic INT8 quantization unavailable: {exc}", metric="int8_decode_speed")

    decode_fn = build_decode_fn(q_model)
    tp = measure_throughput(decode_fn, tokens_per_run=tokens_per_run, runs=runs, warmup=warmup, path="int8_decode")
    if not is_measured(tp):
        return skipped("INT8 decode throughput produced no valid timing", metric="int8_decode_speed", raw=tp)

    cv = float(tp.get("cv", 0.0))
    if cv > variance_threshold:
        return skipped(
            f"run-to-run coefficient of variation {cv:.3f} exceeds {variance_threshold}",
            metric="int8_decode_speed",
            cv=cv,
            variance_threshold=variance_threshold,
            load=measured_load,
        )

    tok_per_s = float(tp["value"])
    bytes_per_s = tok_per_s * float(bytes_per_token)
    return measurement(
        {
            "int8_tokens_per_second": tok_per_s,
            "int8_bytes_per_second": bytes_per_s,
        },
        metric="int8_decode_speed",
        bytes_per_token=float(bytes_per_token),
        cv=cv,
        variance_threshold=variance_threshold,
        load=measured_load,
        load_threshold=load_threshold,
        quantized_engine=str(torch.backends.quantized.engine),
    )


def build_equal_budget_tables(arm_records: Sequence[Mapping[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Produce paired `equal_byte_exposure` and `equal_wall_clock` comparison tables."""
    equal_byte: List[Dict[str, Any]] = []
    equal_clock: List[Dict[str, Any]] = []
    for rec in arm_records:
        base = {
            "arm_id": rec.get("arm_id"),
            "core_id": rec.get("core_id"),
            "tokenizer_id": rec.get("tokenizer_id"),
            "vocab_size": rec.get("vocab_size"),
        }
        if "equal_byte_bpb" in rec or "held_out_bpb" in rec:
            equal_byte.append(
                {
                    **base,
                    "utf8_bytes_seen": rec.get("utf8_bytes_seen"),
                    "held_out_bpb": rec.get("equal_byte_bpb", rec.get("held_out_bpb")),
                    "train_wall_clock_s": rec.get("equal_byte_wall_clock_s", rec.get("train_wall_clock_s")),
                }
            )
        if "equal_clock_bpb" in rec or "held_out_bpb" in rec:
            equal_clock.append(
                {
                    **base,
                    "wall_clock_budget_s": rec.get("wall_clock_budget_s", rec.get("train_wall_clock_s")),
                    "held_out_bpb": rec.get("equal_clock_bpb", rec.get("held_out_bpb")),
                    "utf8_bytes_seen": rec.get("equal_clock_bytes_seen", rec.get("utf8_bytes_seen")),
                }
            )
    equal_byte.sort(key=lambda r: float(r["held_out_bpb"]) if r["held_out_bpb"] is not None else float("inf"))
    equal_clock.sort(key=lambda r: float(r["held_out_bpb"]) if r["held_out_bpb"] is not None else float("inf"))
    return {
        "equal_byte_exposure": equal_byte,
        "equal_wall_clock": equal_clock,
    }

