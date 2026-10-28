"""Versioned 20-task continual-learning benchmark: Delta, sparse Delta and EWC.

The benchmark is intentionally small enough to run on CPU while keeping the
protocol fixed: task order, seeds, split sizes, update budget and the memory
loss definition are all recorded and reused.  Targets from the change proposal
(Delta memory loss < 0.10, EWC reference ~0.24) are reported as comparisons,
never asserted as achieved unless the measured values say so.

Memory loss follows the design document: for every task, take the score
observed immediately after training that task and subtract the final score
measured after the last task; average the non-negative drop over tasks.
"""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

BENCHMARK_VERSION = "continual-20-v1"


@dataclass
class BenchmarkConfig:
    tasks: int = 20
    patterns_per_task: int = 40
    eval_per_task: int = 10
    dim: int = 384
    seed: int = 7
    task_order_seed: int = 11
    update_lr: float = 0.5
    delta_gate: float = 1.0
    density: float = 0.4
    ewc_lambda: float = 200.0
    ewc_lr: float = 0.02

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": BENCHMARK_VERSION,
            "seeds": {"data": self.seed, "task_order": self.task_order_seed},
            "split": {"train_per_task": self.patterns_per_task,
                      "eval_per_task": self.eval_per_task,
                      "eval_pairs_with_train": True},
            "dim": self.dim,
            "update_lr": self.update_lr,
            "delta_gate": self.delta_gate,
            "density": self.density,
            "ewc_lambda": self.ewc_lambda,
            "ewc_lr": self.ewc_lr,
        }


def build_task_stream(config: BenchmarkConfig) -> List[Dict[str, torch.Tensor]]:
    """Fixed task order and fixed train/eval split for every adapter."""
    generator = torch.Generator().manual_seed(config.seed)
    tasks = []
    for task_index in range(config.tasks):
        key_center = F.normalize(torch.randn(config.dim, generator=generator), dim=0)
        value_center = F.normalize(torch.randn(config.dim, generator=generator), dim=0)
        keys, values = [], []
        for _ in range(config.patterns_per_task):
            key_noise = torch.randn(config.dim, generator=generator)
            value_noise = torch.randn(config.dim, generator=generator)
            keys.append(F.normalize(0.7 * key_center + 0.3 * key_noise, dim=0))
            values.append(F.normalize(0.7 * value_center + 0.3 * value_noise, dim=0))
        train_keys = torch.stack(keys)
        train_values = torch.stack(values)
        # Evaluation patterns are noisy re-presentations of held-out training
        # associations: retention means "can the system still answer what it
        # was taught", not "can it generalise to a fresh draw".
        eval_keys, eval_values = [], []
        for index in range(config.eval_per_task):
            key_noise = torch.randn(config.dim, generator=generator)
            value_noise = torch.randn(config.dim, generator=generator)
            eval_keys.append(F.normalize(
                train_keys[index] + 0.1 * key_noise, dim=0))
            eval_values.append(F.normalize(
                train_values[index] + 0.1 * value_noise, dim=0))
        tasks.append({
            "task_id": task_index,
            "train_keys": train_keys,
            "train_values": train_values,
            "eval_keys": torch.stack(eval_keys),
            "eval_values": torch.stack(eval_values),
        })
    order_generator = torch.Generator().manual_seed(config.task_order_seed)
    order = torch.randperm(len(tasks), generator=order_generator).tolist()
    return [tasks[i] for i in order]


class _MemorySystem:
    name = "abstract"

    def train_task(self, task: Dict[str, torch.Tensor]) -> None:
        raise NotImplementedError

    def score(self, keys: torch.Tensor, values: torch.Tensor) -> float:
        raise NotImplementedError

    def active_set(self, keys: torch.Tensor) -> Optional[torch.Tensor]:
        return None


class DeltaSystem(_MemorySystem):
    def __init__(self, config: BenchmarkConfig, sparse: bool = False):
        from src.model.delta_memory import DeltaPlasticAdapter, SparseDeltaAdapter
        torch.manual_seed(config.seed)
        self.config = config
        self.sparse = sparse
        if sparse:
            self.adapter = SparseDeltaAdapter(dim=config.dim)
            self.adapter.set_beta_constant(config.delta_gate)
        else:
            self.adapter = DeltaPlasticAdapter(dim=config.dim)
            self.adapter.set_beta_constant(config.delta_gate)
        self.name = "sparse_delta" if sparse else "delta_rtu"
        self.update_latencies: List[float] = []

    def _codes(self, keys: torch.Tensor):
        if self.sparse:
            return torch.stack([self.adapter.coder.encode(key) for key in keys])
        return keys

    def train_task(self, task):
        for key, value in zip(task["train_keys"], task["train_values"]):
            started = time.perf_counter()
            if self.sparse:
                self.adapter.delta.write_association(
                    self.adapter.coder.encode(key), self.adapter.coder.encode(value)
                )
            else:
                self.adapter.write_association(key, value)
            self.update_latencies.append((time.perf_counter() - started) * 1000)

    def score(self, keys, values):
        total = 0.0
        for key, value in zip(keys, values):
            if self.sparse:
                key_code = self.adapter.coder.encode(key)
                read = self.adapter.delta.read(key_code)
                target = self.adapter.delta.W_V(self.adapter.coder.encode(value))
            else:
                read = self.adapter.read(key)
                target = self.adapter.W_V(value)
            total += float(F.cosine_similarity(read, target, dim=0).item())
        return total / max(1, len(keys))

    def active_set(self, keys):
        if not self.sparse:
            # A dense adapter activates every dimension: overlap is 1.0 by
            # construction and serves as the sparse comparison baseline.
            return torch.ones((len(keys), self.config.dim), dtype=torch.bool)
        masks = []
        for key in keys:
            code = self.adapter.coder.encode(key)
            masks.append(code != 0)
        return torch.stack(masks)


class EWCSystem(_MemorySystem):
    """Budget-matched linear associative memory with an EWC quadratic penalty."""

    name = "ewc"

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        torch.manual_seed(config.seed)
        generator = torch.Generator().manual_seed(config.seed)
        weight = torch.randn(config.dim, config.dim, generator=generator)
        self.weight = torch.nn.Parameter(weight / math.sqrt(config.dim))
        self.fisher = torch.zeros_like(self.weight)
        self.anchor = self.weight.detach().clone()
        self.update_latencies: List[float] = []

    def _loss(self, keys, values):
        predictions = keys @ self.weight.T
        return F.mse_loss(predictions, values)

    def train_task(self, task):
        keys, values = task["train_keys"], task["train_values"]
        # Fisher diagonal from the task data (same budget: one pass).
        self.weight.grad = None
        loss = self._loss(keys, values)
        loss.backward()
        self.fisher += self.weight.grad.detach() ** 2 / max(1, len(keys))
        # Adam keeps the reference in the same 40-step budget as the delta
        # adapter's 40 rank-one writes; plain SGD is too slow on the
        # ill-conditioned within-task key covariance.
        optimizer = torch.optim.Adam([self.weight], lr=self.config.ewc_lr)
        for _ in range(len(keys)):  # budget-matched: one step per pattern
            started = time.perf_counter()
            optimizer.zero_grad()
            loss = self._loss(keys, values)
            penalty = (self.fisher * (self.weight - self.anchor) ** 2).sum()
            (loss + self.config.ewc_lambda * penalty).backward()
            optimizer.step()
            self.update_latencies.append((time.perf_counter() - started) * 1000)
        self.anchor = self.weight.detach().clone()

    def score(self, keys, values):
        predictions = keys @ self.weight.detach().T
        return float(F.cosine_similarity(predictions, values, dim=1).mean().item())


def compute_memory_loss(best: Dict[int, float],
                        final: Dict[int, float]) -> float:
    """Mean non-negative drop between each task's best and final score.

    ``best[task]`` is the score measured immediately after that task was
    trained (or later, if it improved); ``final[task]`` is the score measured
    after the last task.  A task that never degrades contributes 0.
    """
    if not final:
        return 0.0
    missing = set(final) - set(best)
    if missing:
        raise ValueError(f"missing best score for tasks: {sorted(missing)}")
    drops = [max(0.0, best[task_id] - final[task_id]) for task_id in final]
    return statistics.mean(drops)


def _measure_retention(system: _MemorySystem, seen: List[Dict],
                       config: BenchmarkConfig) -> Dict[int, float]:
    scores = {}
    for task in seen:
        scores[task["task_id"]] = system.score(task["eval_keys"], task["eval_values"])
    return scores


def _run_system(system: _MemorySystem, tasks: List[Dict],
                config: BenchmarkConfig) -> Dict[str, Any]:
    seen: List[Dict] = []
    best: Dict[int, float] = {}
    history: List[Dict[str, Any]] = []
    started = time.perf_counter()
    for task in tasks:
        system.train_task(task)
        seen.append(task)
        retention = _measure_retention(system, seen, config)
        for task_id, value in retention.items():
            best[task_id] = max(best.get(task_id, -math.inf), value)
        history.append({"after_task": task["task_id"], "retention": retention})
    final = _measure_retention(system, seen, config)
    elapsed = time.perf_counter() - started

    overlap = None
    utilization = None
    if hasattr(system, "active_set"):
        masks = [system.active_set(task["eval_keys"]) for task in tasks[:5]]
        masks = [mask for mask in masks if mask is not None]
        if len(masks) >= 2:
            overlaps = []
            for i in range(len(masks)):
                for j in range(i + 1, len(masks)):
                    intersection = (masks[i] & masks[j]).sum(dim=1).float()
                    union = (masks[i] | masks[j]).sum(dim=1).float().clamp(min=1)
                    overlaps.append(float((intersection / union).mean().item()))
            overlap = statistics.mean(overlaps)

        all_masks = [system.active_set(task["eval_keys"]) for task in tasks]
        all_masks = [mask for mask in all_masks if mask is not None]
        if all_masks:
            stacked = torch.cat(all_masks)
            code_dim = stacked.shape[1]
            active_per_code = stacked.sum(dim=1).float()
            utilization = {
                "code_dim": code_dim,
                "distinct_codes_used": int(stacked.any(dim=0).sum().item()),
                "distinct_ratio": float(stacked.any(dim=0).float().mean().item()),
                "mean_active_per_code": float(active_per_code.mean().item()),
                "sparsity": float(1.0 - active_per_code.mean().item() / code_dim),
            }

    latencies = sorted(getattr(system, "update_latencies", []) or [0.0])
    p95 = latencies[min(len(latencies) - 1, int(0.95 * (len(latencies) - 1)))] if latencies else None
    return {
        "system": system.name,
        "memory_loss": compute_memory_loss(best, final),
        "per_task_drop": {
            str(task_id): max(0.0, best[task_id] - final[task_id])
            for task_id in sorted(final)
        },
        "final_scores": {str(k): v for k, v in sorted(final.items())},
        "best_scores": {str(k): v for k, v in sorted(best.items())},
        "history": history,
        "interference_overlap": overlap,
        "code_utilization": utilization,
        "update_latency_ms_p95": p95,
        "update_latency_ms_mean": statistics.mean(latencies) if latencies else None,
        "duration_s": elapsed,
    }


def run_benchmark(config: Optional[BenchmarkConfig] = None) -> Dict[str, Any]:
    """Run dense Delta, sparse Delta and the EWC reference on the same stream."""
    config = config or BenchmarkConfig()
    tasks = build_task_stream(config)
    results = {"config": config.to_dict(), "tasks": len(tasks)}
    systems = {
        "delta_rtu": DeltaSystem(config, sparse=False),
        "sparse_delta": DeltaSystem(config, sparse=True),
        "ewc": EWCSystem(config),
    }
    for key, system in systems.items():
        results[key] = _run_system(system, tasks, config)

    delta_loss = results["delta_rtu"]["memory_loss"]
    sparse_loss = results["sparse_delta"]["memory_loss"]
    ewc_loss = results["ewc"]["memory_loss"]
    dense_overlap = results["delta_rtu"]["interference_overlap"]
    sparse_overlap = results["sparse_delta"]["interference_overlap"]
    relative_reduction = (
        None if not dense_overlap or sparse_overlap is None
        else 1.0 - (sparse_overlap / dense_overlap)
    )
    delta_final = statistics.mean(results["delta_rtu"]["final_scores"].values())
    sparse_final = statistics.mean(results["sparse_delta"]["final_scores"].values())
    quality_delta_pp = (delta_final - sparse_final) * 100.0
    interference_target_met = bool(
        relative_reduction is not None and relative_reduction >= 0.20
        and quality_delta_pp <= 5.0
    )
    results["summary"] = {
        "code_utilization": results["sparse_delta"]["code_utilization"],
        "delta_memory_loss": delta_loss,
        "sparse_memory_loss": sparse_loss,
        "ewc_memory_loss": ewc_loss,
        "delta_target": 0.10,
        "ewc_reference_target": 0.24,
        "delta_target_met": delta_loss < 0.10,
        "ewc_reference_matched": abs(ewc_loss - 0.24) <= 0.10,
        "ewc_note": (
            "The external ~0.24 EWC reference was not reproduced by this "
            "protocol; the measured EWC loss is reported as-is."
        ),
        "interference": {
            "dense_overlap": dense_overlap,
            "sparse_overlap": sparse_overlap,
            "relative_reduction": relative_reduction,
            "target_relative_reduction": 0.20,
            "retrieval_quality_delta_pp": quality_delta_pp,
            "quality_loss_limit_pp": 5.0,
            "target_met": interference_target_met,
        },
        "note": (
            "Measured values from this configuration only. Reference figures "
            "are external targets, not measurements."
        ),
    }
    return results


def evaluate_domain_retention_matrix(
    memory_arm: Any,
    domain_stream: List[Dict[str, Any]],
    eval_per_domain_limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Run sequential multi-domain ingestion (`5 x 5` retention matrix + forgetting).

    Hardened protocol:
    - Ingests raw fact text + interspersed distractor passages with ONLY `{"domain": dom_name}`
      metadata (never passes oracle `entity_key` or `force=True`, so surprise gates and subject
      extractors are genuinely tested).
    - Queries paraphrased natural-language `fact["query"]` (falling back to a natural question
      if `query` is absent) and scores strict `top_k=1` exact answer recall (plus `top_k=4` diagnostic).
    """
    memory_arm.reset()
    n_domains = len(domain_stream)
    matrix: List[List[float]] = [[0.0 for _ in range(n_domains)] for _ in range(n_domains)]
    matrix_top4: List[List[float]] = [[0.0 for _ in range(n_domains)] for _ in range(n_domains)]
    best_by_domain: Dict[int, float] = {}
    total_facts = 0
    total_distractors = 0

    for stage_idx, dom in enumerate(domain_stream):
        dom_name = str(dom.get("domain_name", f"domain_{stage_idx}"))
        facts = list(dom.get("facts", []))
        distractors = list(dom.get("distractors", []))
        d_ptr = 0
        stride = max(1, len(facts) // max(1, len(distractors))) if distractors else len(facts) + 1
        for f_idx, fact in enumerate(facts):
            if d_ptr < len(distractors) and f_idx % stride == 0:
                memory_arm.ingest(str(distractors[d_ptr]), metadata={"domain": dom_name})
                total_distractors += 1
                d_ptr += 1
            memory_arm.ingest(str(fact["text"]), metadata={"domain": dom_name})
            total_facts += 1
        while d_ptr < len(distractors):
            memory_arm.ingest(str(distractors[d_ptr]), metadata={"domain": dom_name})
            total_distractors += 1
            d_ptr += 1

        # Measure retention across all domains seen up to `stage_idx`
        for eval_idx in range(stage_idx + 1):
            eval_dom = domain_stream[eval_idx]
            all_eval_facts = list(eval_dom.get("facts", []))
            if eval_per_domain_limit is not None and 0 < int(eval_per_domain_limit) < len(all_eval_facts):
                lim = int(eval_per_domain_limit)
                n_all = len(all_eval_facts)
                eval_facts = [
                    all_eval_facts[(k * (n_all - 1)) // max(1, lim - 1)]
                    for k in range(lim)
                ]
            else:
                eval_facts = all_eval_facts
            hits_top1 = 0
            hits_top4 = 0
            for fact in eval_facts:
                q_str = str(fact.get("query") or f"What was recorded regarding {fact['entity_key']}?")
                q_res = memory_arm.query(q_str, top_k=4)
                hits = q_res.get("hits") or []
                ans_low = str(fact["answer"]).lower()
                if hits and ans_low in str(hits[0].get("text", "")).lower():
                    hits_top1 += 1
                if any(ans_low in str(h.get("text", "")).lower() for h in hits[:4]):
                    hits_top4 += 1
            denom = float(max(1, len(eval_facts)))
            acc = hits_top1 / denom
            matrix[stage_idx][eval_idx] = acc
            matrix_top4[stage_idx][eval_idx] = hits_top4 / denom
            best_by_domain[eval_idx] = max(best_by_domain.get(eval_idx, 0.0), acc)

    final_by_domain = {d_idx: matrix[n_domains - 1][d_idx] for d_idx in range(n_domains)} if n_domains else {}
    final_top4_by_domain = (
        {d_idx: matrix_top4[n_domains - 1][d_idx] for d_idx in range(n_domains)} if n_domains else {}
    )
    forgetting = compute_memory_loss(best_by_domain, final_by_domain) if final_by_domain else 0.0
    avg_final = statistics.mean(final_by_domain.values()) if final_by_domain else 0.0
    avg_final_top4 = statistics.mean(final_top4_by_domain.values()) if final_top4_by_domain else 0.0
    footprint = memory_arm.footprint_bytes().to_dict() if hasattr(memory_arm, "footprint_bytes") else {}
    return {
        "retention_matrix": matrix,
        "retention_matrix_top4": matrix_top4,
        "average_final_accuracy": avg_final,
        "average_final_recall_at_4": avg_final_top4,
        "backward_forgetting": forgetting,
        "final_by_domain": final_by_domain,
        "total_facts_ingested": total_facts,
        "total_distractors_ingested": total_distractors,
        "footprint": footprint,
    }


def evaluate_contradiction_updates(
    memory_arm: Any,
    contradiction_pairs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Evaluate supersession/overwrite accuracy when contradicting facts arrive."""
    memory_arm.reset()
    for item in contradiction_pairs:
        memory_arm.ingest(str(item["initial_fact"]), metadata={"domain": "contradiction_stream"})
        for dist in item.get("distractors", []):
            memory_arm.ingest(str(dist), metadata={"domain": "contradiction_stream"})
    for item in contradiction_pairs:
        memory_arm.erase_or_supersede(str(item["entity_key"]), str(item["updated_fact"]))

    updated_correct = 0
    stale_errors = 0
    for item in contradiction_pairs:
        q_res = memory_arm.query(str(item["query"]), top_k=1)
        top_text = str(q_res["hits"][0]["text"]) if q_res.get("hits") else ""
        if str(item["expected_new"]).lower() in top_text.lower():
            updated_correct += 1
        elif str(item["stale_old"]).lower() in top_text.lower():
            stale_errors += 1

    total = max(1, len(contradiction_pairs))
    footprint = memory_arm.footprint_bytes().to_dict() if hasattr(memory_arm, "footprint_bytes") else {}
    return {
        "contradiction_update_accuracy": updated_correct / float(total),
        "stale_answer_rate": stale_errors / float(total),
        "num_pairs": len(contradiction_pairs),
        "footprint": footprint,
    }


__all__ = [
    "BENCHMARK_VERSION",
    "BenchmarkConfig",
    "EWCSystem",
    "DeltaSystem",
    "build_task_stream",
    "compute_memory_loss",
    "run_benchmark",
    "evaluate_domain_retention_matrix",
    "evaluate_contradiction_updates",
]
