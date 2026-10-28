"""CPU-friendly micro-batched multi-task trainer and O4 (`online_rtrl` vs `chunked_bptt`)."""

from __future__ import annotations

import inspect
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import psutil
import torch
import torch.nn as nn

from src.model.cores.contracts import CandidateLM
from src.model.cores.objectives import PureCEObjective

MAX_CPU_RSS_BYTES = int(2.5 * (1024 ** 3))  # 2.5 GiB ceiling


@dataclass
class TrainerConfig:
    lr: float = 1e-3
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    warmup_fraction: float = 0.05
    total_steps: int = 200
    micro_batch_size: int = 4
    gradient_mode: str = "chunked_bptt"  # "chunked_bptt" | "online_rtrl"
    rtrl_chunk_len: int = 16
    batching_mode: str = "interleaved"   # "interleaved" | "summed"
    task_probs: Dict[str, float] = field(
        default_factory=lambda: {"lm": 0.80, "decision": 0.15, "routing": 0.05}
    )
    summed_weights: Dict[str, float] = field(
        default_factory=lambda: {"lm": 1.0, "decision": 0.25, "routing": 0.15}
    )
    seed: int = 1337
    rss_limit_bytes: int = MAX_CPU_RSS_BYTES

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_adamw_optimizer(
    modules: Sequence[nn.Module],
    lr: float = 1e-3,
    weight_decay: float = 0.1,
    betas: Tuple[float, float] = (0.9, 0.95),
) -> torch.optim.AdamW:
    """Create AdamW excluding 1-D norms/biases and embeddings if tied appropriately."""
    decay_params: List[nn.Parameter] = []
    no_decay_params: List[nn.Parameter] = []
    seen: set[int] = set()
    for mod in modules:
        for name, param in mod.named_parameters():
            if not param.requires_grad or id(param) in seen:
                continue
            seen.add(id(param))
            if param.dim() < 2 or "norm" in name or name.endswith("bias") or "decay" in name or "A_log" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
    groups = [
        {"params": decay_params, "weight_decay": float(weight_decay)},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=float(lr), betas=betas)


class MultiTaskBatchSampler:
    """Deterministic seeded task-step sampler (`0.80` LM / `0.15` decision / `0.05` routing)."""

    def __init__(self, task_probs: Dict[str, float], seed: int = 1337) -> None:
        total = sum(task_probs.values())
        if total <= 0:
            raise ValueError("task_probs must sum to a positive value")
        self.tasks = list(task_probs.keys())
        probs = [float(task_probs[k]) / total for k in self.tasks]
        self.cdf: List[float] = []
        acc = 0.0
        for p in probs:
            acc += p
            self.cdf.append(acc)
        self.cdf[-1] = 1.0
        self.gen = torch.Generator()
        self.gen.manual_seed(int(seed))
        self.counts: Dict[str, int] = {k: 0 for k in self.tasks}

    def sample_task(self) -> str:
        u = float(torch.rand(1, generator=self.gen).item())
        for task, edge in zip(self.tasks, self.cdf):
            if u <= edge:
                self.counts[task] += 1
                return task
        chosen = self.tasks[-1]
        self.counts[chosen] += 1
        return chosen


def _cosine_lr(step: int, total_steps: int, base_lr: float, warmup_fraction: float, min_lr_ratio: float) -> float:
    total = max(1, int(total_steps))
    warmup = max(1, int(round(total * warmup_fraction)))
    if step < warmup:
        return base_lr * float(step + 1) / float(warmup)
    if total <= warmup:
        return base_lr
    progress = min(1.0, max(0.0, float(step - warmup) / float(total - warmup)))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


class TournamentTrainer:
    """Unified CPU-safe trainer supporting micro-batch accumulation, `O4` RTRL, and atomic resume."""

    def __init__(
        self,
        model: CandidateLM,
        objective: Optional[nn.Module] = None,
        extra_modules: Optional[Sequence[nn.Module]] = None,
        config: Optional[TrainerConfig] = None,
        pad_id: int = 0,
        eos_id: int = 1,
    ) -> None:
        self.config = config or TrainerConfig()
        if self.config.gradient_mode not in {"chunked_bptt", "online_rtrl"}:
            raise ValueError(f"Unknown gradient_mode {self.config.gradient_mode!r}")
        if self.config.gradient_mode == "online_rtrl" and not getattr(model.core, "supports_online_rtrl", False):
            raise ValueError(
                f"Core {getattr(model.core, 'core_id', type(model.core).__name__)} does not support "
                "online_rtrl; matrix and attention cores require gradient_mode='chunked_bptt'."
            )

        self.model = model
        self.objective = objective if objective is not None else PureCEObjective()
        try:
            sig = inspect.signature(self.objective.compute_loss)  # type: ignore[attr-defined]
            self._accepts_raw_hidden = (
                "raw_hidden_states" in sig.parameters
                or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            )
        except (TypeError, ValueError):
            self._accepts_raw_hidden = False

        self.extra_modules = list(extra_modules) if extra_modules is not None else []
        self.all_modules: List[nn.Module] = [self.model, self.objective, *self.extra_modules]
        self.optimizer = build_adamw_optimizer(
            self.all_modules,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            betas=self.config.betas,
        )
        self.sampler = MultiTaskBatchSampler(self.config.task_probs, seed=self.config.seed)
        self.pad_id = int(pad_id)
        self.eos_id = int(eos_id)
        self.step_idx = 0
        self.peak_rss_bytes = 0
        self._proc = psutil.Process(os.getpid())

    def _call_objective(
        self,
        input_ids: torch.Tensor,
        details: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "input_ids": input_ids,
            "logits": details["logits"],
            "hidden_states": details["hidden_states"],
            "raw_embeds": details["raw_embeds"],
            "pad_id": self.pad_id,
            "eos_id": self.eos_id,
        }
        if self._accepts_raw_hidden:
            kwargs["raw_hidden_states"] = details.get("raw_hidden_states")
        return self.objective.compute_loss(**kwargs)  # type: ignore[operator]

    def _record_rss(self) -> int:
        rss = int(self._proc.memory_info().rss)
        if rss > self.peak_rss_bytes:
            self.peak_rss_bytes = rss
        if rss > self.config.rss_limit_bytes:
            raise MemoryError(
                f"Process RSS {rss / (1024**3):.2f} GiB exceeded budget {self.config.rss_limit_bytes / (1024**3):.2f} GiB"
            )
        return rss

    def _all_trainable_params(self) -> List[nn.Parameter]:
        out: List[nn.Parameter] = []
        seen: set[int] = set()
        for mod in self.all_modules:
            for p in mod.parameters():
                if p.requires_grad and id(p) not in seen:
                    seen.add(id(p))
                    out.append(p)
        return out

    def accumulate_lm_batch(self, input_ids: torch.Tensor, scale: float = 1.0) -> Dict[str, float]:
        """Run backward over `input_ids` split into micro-batches with exact token-weighted scaling."""
        for mod in self.all_modules:
            mod.train()

        if self.config.gradient_mode == "online_rtrl":
            return self._accumulate_online_rtrl(input_ids, scale=scale)

        batch_size = input_ids.shape[0]
        mb = max(1, int(self.config.micro_batch_size))
        total_valid = int((input_ids[:, 1:] != self.pad_id).sum().item())
        if total_valid == 0:
            return {"ce_loss": 0.0, "aux_loss": 0.0, "total_loss": 0.0}

        ce_sum = 0.0
        aux_sum = 0.0
        tot_sum = 0.0
        for start in range(0, batch_size, mb):
            stop = min(start + mb, batch_size)
            sub_ids = input_ids[start:stop]
            sub_valid = int((sub_ids[:, 1:] != self.pad_id).sum().item())
            if sub_valid == 0:
                continue
            weight = float(scale) * (float(sub_valid) / float(total_valid))
            details = self.model(sub_ids, return_details=True)
            loss_dict = self._call_objective(sub_ids, details)
            (loss_dict["total_loss"] * weight).backward()
            ce_sum += float(loss_dict["ce_loss"].detach().item()) * (sub_valid / total_valid)
            aux_sum += float(loss_dict["aux_loss"].detach().item()) * (sub_valid / total_valid)
            tot_sum += float(loss_dict["total_loss"].detach().item()) * (sub_valid / total_valid)
            self._record_rss()

        return {"ce_loss": ce_sum, "aux_loss": aux_sum, "total_loss": tot_sum}

    def _accumulate_online_rtrl(self, input_ids: torch.Tensor, scale: float = 1.0) -> Dict[str, float]:
        """Online diagonal RTRL across detached temporal windows with carried O(d) eligibility traces."""
        batch, seq_len = input_ids.shape
        win = max(2, int(self.config.rtrl_chunk_len))
        total_chunks = 0
        ce_acc = 0.0
        tot_acc = 0.0

        state = self.model.init_state(batch_size=batch, device=input_ids.device)
        core = self.model.core
        trace = torch.zeros_like(state)

        stride = max(1, win - 1)
        for start in range(0, max(1, seq_len - 1), stride):
            stop = min(seq_len, start + win)
            if stop - start < 2:
                break
            sub_ids = input_ids[:, start:stop]
            state_detached = state.detach()
            coupled_states: List[torch.Tensor] = []
            for idx, layer in enumerate(core.layers):  # type: ignore[attr-defined]
                if hasattr(layer, "decay"):
                    theta = layer.decay.view(1, -1)
                    s_coupled = state_detached[idx] + (theta - theta.detach()) * trace[idx].detach()
                else:
                    theta = layer.w_g.bias.view(1, -1)
                    s_coupled = state_detached[idx] + (theta - theta.detach()) * trace[idx].detach()
                coupled_states.append(s_coupled)
            state_in = torch.stack(coupled_states, dim=0)

            details = self.model(sub_ids, state=state_in, return_details=True)
            loss_dict = self._call_objective(sub_ids, details)
            chunk_count = max(1, math.ceil((seq_len - 1) / stride))
            (loss_dict["total_loss"] * (float(scale) / float(chunk_count))).backward()

            with torch.no_grad():
                next_state = details["state"].detach()
                new_traces: List[torch.Tensor] = []
                steps_n = float(stop - start)
                for idx, layer in enumerate(core.layers):  # type: ignore[attr-defined]
                    if hasattr(layer, "decay"):
                        g = torch.sigmoid(layer.decay).view(1, -1)
                    else:
                        g = torch.sigmoid(layer.w_g.bias).view(1, -1)
                    dg = g * (1.0 - g)
                    new_traces.append((g ** steps_n) * trace[idx] + dg * state_detached[idx])
                trace = torch.stack(new_traces, dim=0)
                state = next_state

            ce_acc += float(loss_dict["ce_loss"].detach().item())
            tot_acc += float(loss_dict["total_loss"].detach().item())
            total_chunks += 1
            self._record_rss()

        denom = float(max(1, total_chunks))
        return {"ce_loss": ce_acc / denom, "aux_loss": (tot_acc - ce_acc) / denom, "total_loss": tot_acc / denom}

    def optimizer_step(self) -> Dict[str, float]:
        lr_now = _cosine_lr(
            self.step_idx,
            self.config.total_steps,
            self.config.lr,
            self.config.warmup_fraction,
            self.config.min_lr_ratio,
        )
        for group in self.optimizer.param_groups:
            group["lr"] = lr_now
        grad_norm = float(torch.nn.utils.clip_grad_norm_(self._all_trainable_params(), self.config.grad_clip).item())
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        if hasattr(self.objective, "on_step_end"):
            self.objective.on_step_end(self.model)  # type: ignore[operator]
        self.step_idx += 1
        self._record_rss()
        return {"lr": lr_now, "grad_norm": grad_norm}

    def train_step(
        self,
        *,
        lm_batch: Optional[torch.Tensor] = None,
        decision_loss_fn: Optional[Callable[[], torch.Tensor]] = None,
        routing_loss_fn: Optional[Callable[[], torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """Execute one multi-task training step under `interleaved` or `summed` schedule."""
        self.optimizer.zero_grad(set_to_none=True)
        metrics: Dict[str, Any] = {}

        if self.config.batching_mode == "interleaved":
            task = self.sampler.sample_task()
            metrics["selected_task"] = task
            if task == "decision" and decision_loss_fn is not None:
                loss = decision_loss_fn()
                loss.backward()
                metrics["decision_loss"] = float(loss.detach().item())
            elif task == "routing" and routing_loss_fn is not None:
                loss = routing_loss_fn()
                loss.backward()
                metrics["routing_loss"] = float(loss.detach().item())
            elif lm_batch is not None:
                metrics.update(self.accumulate_lm_batch(lm_batch, scale=1.0))
            else:
                raise ValueError(f"Task {task!r} sampled, but no corresponding batch/loss_fn provided")
        elif self.config.batching_mode == "summed":
            metrics["selected_task"] = "summed"
            if lm_batch is not None:
                metrics.update(self.accumulate_lm_batch(lm_batch, scale=self.config.summed_weights.get("lm", 1.0)))
            if decision_loss_fn is not None:
                d_loss = decision_loss_fn()
                (d_loss * self.config.summed_weights.get("decision", 0.25)).backward()
                metrics["decision_loss"] = float(d_loss.detach().item())
            if routing_loss_fn is not None:
                r_loss = routing_loss_fn()
                (r_loss * self.config.summed_weights.get("routing", 0.15)).backward()
                metrics["routing_loss"] = float(r_loss.detach().item())
        else:
            raise ValueError(f"Unsupported batching_mode {self.config.batching_mode!r}")

        step_info = self.optimizer_step()
        metrics.update(step_info)
        return metrics

    def save_checkpoint(self, path: str | Path) -> Path:
        """Atomically persist trainer state so interrupted runs resume cleanly."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "step_idx": self.step_idx,
            "config": self.config.to_dict(),
            "model": self.model.state_dict(),
            "objective": self.objective.state_dict(),
            "extra": [m.state_dict() for m in self.extra_modules],
            "optimizer": self.optimizer.state_dict(),
            "sampler_counts": dict(self.sampler.counts),
            "sampler_rng": self.sampler.gen.get_state(),
        }
        fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=".ckpt_", suffix=".pt")
        os.close(fd)
        try:
            torch.save(payload, tmp_name)
            os.replace(tmp_name, target)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        return target

    def load_checkpoint(self, path: str | Path) -> None:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        self.step_idx = int(payload["step_idx"])
        self.model.load_state_dict(payload["model"])
        self.objective.load_state_dict(payload["objective"])
        for mod, state in zip(self.extra_modules, payload.get("extra", [])):
            mod.load_state_dict(state)
        self.optimizer.load_state_dict(payload["optimizer"])
        self.sampler.counts = dict(payload["sampler_counts"])
        self.sampler.gen.set_state(payload["sampler_rng"])
