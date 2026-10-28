"""Protocols, data structures, and state-complexity contracts for candidate cores.

Implements the ``model/core-candidates`` slot interfaces so every sequence
backbone, tokenizer, auxiliary objective, System-1 decision readout, 5-way
action router, and memory/adaptation arm plugs into one invariant harness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple, runtime_checkable

import torch
import torch.nn as nn

STATE_COMPLEXITY_CLASSES = (
    "constant_vector",
    "constant_matrix",
    "sliding_window_kv",
    "linear_kv",
)

ACTION_NAMES = (
    "generate",
    "classify",
    "retrieve",
    "tool",
    "abstain",
)


@dataclass(frozen=True)
class StateSpec:
    """Describes how a sequence core's state scales with sequence length."""

    complexity_class: str
    bytes_per_sequence: int
    shape_per_layer: Tuple[int, ...]
    num_layers: int
    window_size: Optional[int] = None

    def __post_init__(self) -> None:
        if self.complexity_class not in STATE_COMPLEXITY_CLASSES:
            raise ValueError(
                f"Unknown state complexity_class {self.complexity_class!r}; "
                f"expected one of {STATE_COMPLEXITY_CLASSES}"
            )
        if self.bytes_per_sequence < 0:
            raise ValueError("bytes_per_sequence must be non-negative")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "complexity_class": self.complexity_class,
            "bytes_per_sequence": int(self.bytes_per_sequence),
            "shape_per_layer": list(self.shape_per_layer),
            "num_layers": int(self.num_layers),
            "window_size": self.window_size,
        }


@dataclass
class CoreOutput:
    """Return payload of ``SequenceCore.forward`` and ``SequenceCore.step``."""

    hidden_states: torch.Tensor  # (B, T, d_model)
    state: Any
    state_spec: StateSpec
    aux_info: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EncodedSpan:
    """Tokenized span paired with exact underlying UTF-8 byte count."""

    token_ids: List[int]
    utf8_byte_count: int
    control_token_count: int = 0
    vocab_size: int = 256


@dataclass
class DecisionResult:
    """Uniform output of any Track-3 System-1 decision readout (D1..D4)."""

    selected_index: int
    selected_option: str
    probabilities: List[float]
    logits: List[float]
    confidence: float
    is_null_or_abstain: bool
    latency_ms: float
    mode: str = "standard"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_null(self) -> bool:
        return bool(self.is_null_or_abstain)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_index": int(self.selected_index),
            "selected_option": str(self.selected_option),
            "probabilities": [float(p) for p in self.probabilities],
            "logits": [float(z) for z in self.logits],
            "confidence": float(self.confidence),
            "is_null_or_abstain": bool(self.is_null_or_abstain),
            "latency_ms": float(self.latency_ms),
            "mode": str(self.mode),
            "metadata": dict(self.metadata),
        }


@dataclass
class RoutingResult:
    """Uniform output of any Track-4 5-way action router (R1..R4)."""

    action: str
    raw_action: str
    probabilities: Dict[str, float]
    confidence: float
    guard_triggered: bool
    guard_reason: Optional[str]
    latency_ms: float
    tool_call: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.action not in ACTION_NAMES:
            raise ValueError(f"Invalid routed action {self.action!r}; expected {ACTION_NAMES}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "raw_action": self.raw_action,
            "probabilities": {k: float(v) for k, v in self.probabilities.items()},
            "confidence": float(self.confidence),
            "guard_triggered": bool(self.guard_triggered),
            "guard_reason": self.guard_reason,
            "latency_ms": float(self.latency_ms),
            "tool_call": self.tool_call,
        }


@dataclass(frozen=True)
class MemoryFootprint:
    """Static parameter bytes vs dynamic state/store bytes for a MemoryArm."""

    arm_id: str
    static_param_bytes: int
    dynamic_store_bytes: int
    active_items: int

    @property
    def total_bytes(self) -> int:
        return int(self.static_param_bytes + self.dynamic_store_bytes)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "static_param_bytes": int(self.static_param_bytes),
            "dynamic_store_bytes": int(self.dynamic_store_bytes),
            "total_bytes": int(self.total_bytes),
            "active_items": int(self.active_items),
        }


LayerMemoryHook = Callable[[int, torch.Tensor, Optional[torch.Tensor]], torch.Tensor]


@runtime_checkable
class SequenceCore(Protocol):
    """Uniform sequence-mixing backbone protocol implemented by C0..C6."""

    core_id: str
    d_model: int
    n_layers: int
    supports_online_rtrl: bool

    def state_spec(self, batch_size: int = 1, seq_len: int = 1) -> StateSpec:
        ...

    def init_state(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> Any:
        ...

    def reset_state(self) -> None:
        ...

    def forward_sequence(
        self,
        x: torch.Tensor,
        state: Any = None,
        token_ids: Optional[torch.Tensor] = None,
        layer_hook: Optional[LayerMemoryHook] = None,
    ) -> CoreOutput:
        ...

    def step(
        self,
        x_t: torch.Tensor,
        state: Any = None,
        token_id: Optional[torch.Tensor] = None,
        layer_hook: Optional[LayerMemoryHook] = None,
    ) -> CoreOutput:
        ...


@runtime_checkable
class TokenizerAdapter(Protocol):
    """Uniform tokenizer wrapper tracking exact UTF-8 byte counts."""

    name: str
    vocab_size: int
    pad_id: int
    eos_id: int
    opt_id: int
    sep_id: int
    route_token_ids: Mapping[str, int]

    def encode(self, text: str, add_eos: bool = False) -> List[int]:
        ...

    def decode(self, token_ids: Sequence[int]) -> str:
        ...

    def encode_span(self, text: str, add_eos: bool = False) -> EncodedSpan:
        ...


@runtime_checkable
class AuxiliaryObjective(Protocol):
    """Training loss wrapper (O1, O2, O3) decoupled from core architecture."""

    objective_id: str

    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        hidden_states: torch.Tensor,
        target_embeddings: Optional[torch.Tensor] = None,
        stop_probs: Optional[torch.Tensor] = None,
        stop_targets: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        ...


@runtime_checkable
class DecisionReadout(Protocol):
    """Non-autoregressive System-1 option-scoring readout (D1..D4)."""

    readout_id: str

    def score_options(
        self,
        query_hidden: torch.Tensor,
        option_hiddens: Sequence[torch.Tensor],
        option_names: Sequence[str],
        include_null: bool = True,
        temperature: float = 1.0,
    ) -> DecisionResult:
        ...


@runtime_checkable
class ActionRouter(Protocol):
    """Five-way agent action router (R1..R4)."""

    router_id: str

    def route(
        self,
        query_text: str,
        hidden_states: torch.Tensor,
        lm_logits_last: Optional[torch.Tensor] = None,
        retrieval_similarity: Optional[float] = None,
        tool_schema: Optional[Mapping[str, Any]] = None,
        proposed_tool_call: Optional[Mapping[str, Any]] = None,
    ) -> RoutingResult:
        ...


@runtime_checkable
class MemoryArm(Protocol):
    """Pluggable memory/adaptation arm (M0..M5) for continual & contradiction tests."""

    arm_id: str

    def ingest(self, text: str, metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        ...

    def query(self, query_text: str, top_k: int = 4) -> Dict[str, Any]:
        ...

    def erase_or_supersede(self, key_or_subject: str, new_text: Optional[str] = None) -> Dict[str, Any]:
        ...

    def reset(self) -> None:
        ...

    def footprint_bytes(self) -> MemoryFootprint:
        ...


class CandidateLM(nn.Module):
    """Thin invariant shell wrapping Embedding + SequenceCore + Tied LM Head."""

    def __init__(
        self,
        core: nn.Module,
        vocab_size: int,
        tied_embeddings: bool = True,
        include_stop_head: bool = False,
    ) -> None:
        super().__init__()
        self.core = core
        self.vocab_size = int(vocab_size)
        self.d_model = int(core.d_model)
        self.tied_embeddings = bool(tied_embeddings)

        self.embed = nn.Embedding(self.vocab_size, self.d_model)
        nn.init.normal_(self.embed.weight, mean=0.0, std=1.0 / (self.d_model ** 0.5))
        self.final_norm = nn.RMSNorm(self.d_model, eps=1e-6)
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)
        if self.tied_embeddings:
            self.lm_head.weight = self.embed.weight
        else:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=1.0 / (self.d_model ** 0.5))

        self.stop_head = nn.Linear(self.d_model, 1) if include_stop_head else None
        self.memory_layer_hook: Optional[LayerMemoryHook] = None

    def reset_state(self) -> None:
        self.core.reset_state()

    def init_state(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> Any:
        dev = device if device is not None else self.embed.weight.device
        return self.core.init_state(batch_size=batch_size, device=dev, dtype=dtype)

    def forward(
        self,
        token_ids: torch.Tensor,
        state: Any = None,
        return_details: bool = False,
    ) -> Any:
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        x = self.embed(token_ids) * (self.d_model ** 0.5)
        out: CoreOutput = self.core.forward_sequence(
            x,
            state=state,
            token_ids=token_ids,
            layer_hook=self.memory_layer_hook,
        )
        normed = self.final_norm(out.hidden_states)
        logits = self.lm_head(normed)
        if not return_details:
            return logits
        stop_probs = torch.sigmoid(self.stop_head(normed)) if self.stop_head is not None else None
        return {
            "logits": logits,
            "hidden_states": normed,
            "raw_hidden_states": out.hidden_states,
            "raw_embeds": x,
            "state": out.state,
            "state_spec": out.state_spec,
            "stop_probs": stop_probs,
            "aux_info": out.aux_info,
        }

    def step(self, token_id: torch.Tensor, state: Any = None) -> Dict[str, Any]:
        if token_id.dim() == 0:
            token_id = token_id.unsqueeze(0)
        x_t = self.embed(token_id) * (self.d_model ** 0.5)  # (B, d)
        out: CoreOutput = self.core.step(
            x_t,
            state=state,
            token_id=token_id,
            layer_hook=self.memory_layer_hook,
        )
        h_last = self.final_norm(out.hidden_states)  # (B, 1, d)
        logits = self.lm_head(h_last)
        return {
            "logits": logits[:, 0, :],
            "hidden_state": h_last[:, 0, :],
            "state": out.state,
            "state_spec": out.state_spec,
        }
