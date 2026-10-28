"""Track-4 System-1 5-way action routers (`R1`..`R4`) + deterministic risk & tool-schema guards."""

from __future__ import annotations

import math
import re
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.confidence import fuse_confidence, generation_confidence, retrieval_confidence
from src.model.cores.contracts import ACTION_NAMES, RoutingResult

_TYPE_CHECKERS: Dict[str, Tuple[type, ...]] = {
    "string": (str,),
    "str": (str,),
    "integer": (int,),
    "int": (int,),
    "number": (int, float),
    "float": (int, float),
    "boolean": (bool,),
    "bool": (bool,),
    "object": (dict,),
    "array": (list, tuple),
}


def validate_tool_call_guard(
    tool_schema: Optional[Mapping[str, Any]],
    proposed_tool_call: Optional[Mapping[str, Any]],
) -> Tuple[bool, Optional[str]]:
    """Deterministic tool-schema and permission guard executed before emitting `action='tool'`.

    Expected `tool_schema` layout:
      {
        "allowed_tools": ["calculator", "weather_lookup", ...],
        "tools": {
          "calculator": {
            "required": ["expression"],
            "properties": {"expression": {"type": "string"}},
            "permitted": True
          }
        }
      }
    """
    if tool_schema is None:
        return True, None
    if not proposed_tool_call or not isinstance(proposed_tool_call, Mapping):
        return False, "missing_tool_call"

    name = proposed_tool_call.get("name")
    args = proposed_tool_call.get("arguments")
    if not isinstance(name, str) or not name:
        return False, "missing_tool_name"
    if not isinstance(args, Mapping):
        return False, "invalid_arguments_object"

    allowed = tool_schema.get("allowed_tools")
    if allowed is not None and name not in allowed:
        return False, f"permission_denied:{name}"

    tools_spec = tool_schema.get("tools", {})
    spec = tools_spec.get(name)
    if spec is None:
        return False, f"unknown_tool:{name}"
    if spec.get("permitted", True) is False:
        return False, f"permission_denied:{name}"

    for req_field in spec.get("required", ()):
        if req_field not in args or args[req_field] is None:
            return False, f"missing_required_arg:{req_field}"

    props = spec.get("properties", {})
    if spec.get("additionalProperties", True) is False:
        extra = set(args.keys()) - set(props.keys())
        if extra:
            return False, f"unexpected_arg:{sorted(extra)[0]}"

    for key, val in args.items():
        if key not in props:
            continue
        expected_type = str(props[key].get("type", "")).lower()
        if not expected_type:
            continue
        # Note: in Python `isinstance(True, int)` is True, so reject bools for int/number!
        if expected_type in {"integer", "int", "number", "float"} and isinstance(val, bool):
            return False, f"type_mismatch:{key}:expected_{expected_type}"
        py_types = _TYPE_CHECKERS.get(expected_type)
        if py_types is not None and not isinstance(val, py_types):
            return False, f"type_mismatch:{key}:expected_{expected_type}"
    return True, None


def apply_safety_guards(
    probs_vec: torch.Tensor,
    abstain_threshold: float,
    tool_schema: Optional[Mapping[str, Any]] = None,
    proposed_tool_call: Optional[Mapping[str, Any]] = None,
    latency_ms: float = 0.0,
) -> RoutingResult:
    p = probs_vec.detach().float().reshape(len(ACTION_NAMES))
    p = p / p.sum().clamp(min=1e-8)
    raw_idx = int(torch.argmax(p).item())
    raw_action = ACTION_NAMES[raw_idx]
    conf = float(p[raw_idx].item())

    final_action = raw_action
    guard_triggered = False
    guard_reason: Optional[str] = None

    if final_action != "abstain" and conf < float(abstain_threshold):
        final_action = "abstain"
        guard_triggered = True
        guard_reason = f"low_confidence:{conf:.3f}<{abstain_threshold:.3f}"

    if final_action == "tool":
        ok, err_reason = validate_tool_call_guard(tool_schema, proposed_tool_call)
        if not ok:
            final_action = "abstain"
            guard_triggered = True
            guard_reason = err_reason

    prob_dict = {act: float(p[i].item()) for i, act in enumerate(ACTION_NAMES)}
    return RoutingResult(
        action=final_action,
        raw_action=raw_action,
        probabilities=prob_dict,
        confidence=conf,
        guard_triggered=guard_triggered,
        guard_reason=guard_reason,
        latency_ms=latency_ms,
    )


class HeuristicCosineRouter(nn.Module):
    """`R1` (`heuristic_cosine_gate`): lexical cue rules + cosine prototype gating."""

    router_id = "R1"

    def __init__(self, d_model: int, abstain_threshold: float = 0.35) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.abstain_threshold = float(abstain_threshold)
        gen = torch.Generator().manual_seed(101)
        protos = F.normalize(torch.randn(len(ACTION_NAMES), self.d_model, generator=gen), dim=-1)
        self.register_buffer("prototypes", protos)

    @torch.no_grad()
    def fit_prototypes(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        new_p = self.prototypes.clone()
        for idx in range(len(ACTION_NAMES)):
            mask = labels == idx
            if mask.any():
                new_p[idx] = F.normalize(features[mask].float().mean(dim=0), dim=-1, eps=1e-6)
        self.prototypes.copy_(new_p)

    def route(
        self,
        query_text: str,
        hidden_states: torch.Tensor,
        lm_logits_last: Optional[torch.Tensor] = None,
        retrieval_similarity: Optional[float] = None,
        tool_schema: Optional[Mapping[str, Any]] = None,
        proposed_tool_call: Optional[Mapping[str, Any]] = None,
    ) -> RoutingResult:
        del lm_logits_last
        t0 = time.perf_counter()
        h = hidden_states[0] if hidden_states.dim() == 3 else hidden_states
        q_vec = F.normalize(h.float().mean(dim=0), dim=-1, eps=1e-6)
        logits = 4.0 * torch.mv(self.prototypes, q_vec)

        text_l = query_text.lower()
        if re.search(r"\b(calculate|compute|tool|api|run|execute|weather|lookup)\b", text_l):
            logits[ACTION_NAMES.index("tool")] += 1.5
        if re.search(r"\b(option|choose|classify|which of|select one)\b", text_l):
            logits[ACTION_NAMES.index("classify")] += 1.5
        if re.search(r"\b(according to|recall|stored|memory|document|fact)\b", text_l) or (
            retrieval_similarity is not None and retrieval_similarity >= 0.72
        ):
            logits[ACTION_NAMES.index("retrieve")] += 1.5
        if re.search(r"\b(unanswerable|unknown|insufficient|cannot tell|secret password)\b", text_l):
            logits[ACTION_NAMES.index("abstain")] += 1.6

        probs = F.softmax(logits, dim=-1)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return apply_safety_guards(
            probs,
            abstain_threshold=self.abstain_threshold,
            tool_schema=tool_schema,
            proposed_tool_call=proposed_tool_call,
            latency_ms=dt_ms,
        )


class ThresholdedConfidenceRouter(nn.Module):
    """`R2` (`thresholded_confidence`): uses `src/model/confidence.py` harmonic-mean fusion."""

    router_id = "R2"

    def __init__(self, d_model: int, abstain_threshold: float = 0.42) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.abstain_threshold = float(abstain_threshold)
        self.linear = nn.Linear(self.d_model, len(ACTION_NAMES))
        nn.init.zeros_(self.linear.bias)

    def route(
        self,
        query_text: str,
        hidden_states: torch.Tensor,
        lm_logits_last: Optional[torch.Tensor] = None,
        retrieval_similarity: Optional[float] = None,
        tool_schema: Optional[Mapping[str, Any]] = None,
        proposed_tool_call: Optional[Mapping[str, Any]] = None,
    ) -> RoutingResult:
        t0 = time.perf_counter()
        h = hidden_states[0] if hidden_states.dim() == 3 else hidden_states
        pooled = h.float().mean(dim=0)
        base_logits = self.linear(pooled)

        # Compute generation & retrieval confidence via existing src/model/confidence.py
        if lm_logits_last is not None:
            lp = F.log_softmax(lm_logits_last.detach().float().reshape(-1), dim=-1)
            p_tok = lp.exp().clamp(min=1e-12)
            ent_norm = float(-(p_tok * lp).sum().item() / math.log(max(2, p_tok.numel())))
            top_lp = float(lp.max().item())
            c_gen = generation_confidence(token_logprobs=[top_lp], entropy=ent_norm)
        else:
            c_gen = 0.60

        sim = float(retrieval_similarity) if retrieval_similarity is not None else 0.35
        c_ret = retrieval_confidence(sim)
        c_fused = fuse_confidence(c_ret, c_gen)

        boost = torch.zeros_like(base_logits)
        if sim >= 0.70:
            boost[ACTION_NAMES.index("retrieve")] += 1.4
        if proposed_tool_call is not None or "tool" in query_text.lower() or "calculate" in query_text.lower():
            boost[ACTION_NAMES.index("tool")] += 1.1
        if c_fused < 0.35:
            boost[ACTION_NAMES.index("abstain")] += 1.5
        else:
            boost[ACTION_NAMES.index("generate")] += 0.5 * c_gen

        probs = F.softmax(base_logits + boost, dim=-1)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return apply_safety_guards(
            probs,
            abstain_threshold=self.abstain_threshold,
            tool_schema=tool_schema,
            proposed_tool_call=proposed_tool_call,
            latency_ms=dt_ms,
        )


class PromptedRouteTokenRouter(nn.Module):
    """`R3` (`prompted_route_tokens`): scores reserved `<ROUTE_*>` control tokens from LM head."""

    router_id = "R3"

    def __init__(
        self,
        route_token_ids: Mapping[str, int],
        abstain_threshold: float = 0.35,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.route_token_ids = {act: int(route_token_ids[act]) for act in ACTION_NAMES}
        self.abstain_threshold = float(abstain_threshold)
        self.temperature = float(temperature)

    def forward_route_logits(self, lm_logits_last: torch.Tensor) -> torch.Tensor:
        if lm_logits_last.dim() == 1:
            lm_logits_last = lm_logits_last.unsqueeze(0)
        idx = torch.tensor([self.route_token_ids[a] for a in ACTION_NAMES], device=lm_logits_last.device)
        return lm_logits_last.index_select(dim=-1, index=idx)

    def route(
        self,
        query_text: str,
        hidden_states: torch.Tensor,
        lm_logits_last: Optional[torch.Tensor] = None,
        retrieval_similarity: Optional[float] = None,
        tool_schema: Optional[Mapping[str, Any]] = None,
        proposed_tool_call: Optional[Mapping[str, Any]] = None,
    ) -> RoutingResult:
        del query_text, hidden_states, retrieval_similarity
        if lm_logits_last is None:
            raise ValueError("PromptedRouteTokenRouter (R3) requires `lm_logits_last` from CandidateLM")
        t0 = time.perf_counter()
        r_logits = self.forward_route_logits(lm_logits_last)[0] / max(1e-4, self.temperature)
        probs = F.softmax(r_logits, dim=-1)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return apply_safety_guards(
            probs,
            abstain_threshold=self.abstain_threshold,
            tool_schema=tool_schema,
            proposed_tool_call=proposed_tool_call,
            latency_ms=dt_ms,
        )


class PooledMLPRouter(nn.Module):
    """`R4` (`pooled_mlp_guarded`): Candidate A MLP over `[u; mean(last_16_states)]` + safety guards."""

    router_id = "R4"

    def __init__(
        self,
        d_model: int,
        abstain_threshold: float = 0.45,
        temperature: float = 1.0,
        tail_window: int = 16,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.abstain_threshold = float(abstain_threshold)
        self.temperature = float(temperature)
        self.tail_window = int(tail_window)

        self.attn_pool = nn.Linear(self.d_model, 1, bias=False)
        hidden_dim = max(32, self.d_model // 2)
        self.mlp = nn.Sequential(
            nn.RMSNorm(2 * self.d_model, eps=1e-6),
            nn.Linear(2 * self.d_model, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, len(ACTION_NAMES)),
        )

    def extract_features(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)
        weights = F.softmax(self.attn_pool(hidden_states).squeeze(-1) / math.sqrt(self.d_model), dim=-1)
        u = torch.sum(hidden_states * weights.unsqueeze(-1), dim=1)  # (B, d)
        tail = hidden_states[:, -self.tail_window:, :].mean(dim=1)  # (B, d)
        return torch.cat([u, tail], dim=-1)  # (B, 2d)

    def forward_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.extract_features(hidden_states))

    def set_calibration(self, temperature: float, abstain_threshold: Optional[float] = None) -> None:
        self.temperature = max(1e-4, float(temperature))
        if abstain_threshold is not None:
            self.abstain_threshold = float(abstain_threshold)

    def route(
        self,
        query_text: str,
        hidden_states: torch.Tensor,
        lm_logits_last: Optional[torch.Tensor] = None,
        retrieval_similarity: Optional[float] = None,
        tool_schema: Optional[Mapping[str, Any]] = None,
        proposed_tool_call: Optional[Mapping[str, Any]] = None,
    ) -> RoutingResult:
        del query_text, lm_logits_last, retrieval_similarity
        t0 = time.perf_counter()
        logits = self.forward_logits(hidden_states)[0] / max(1e-4, self.temperature)
        probs = F.softmax(logits, dim=-1)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return apply_safety_guards(
            probs,
            abstain_threshold=self.abstain_threshold,
            tool_schema=tool_schema,
            proposed_tool_call=proposed_tool_call,
            latency_ms=dt_ms,
        )


def compute_routing_utility(
    predicted_actions: Sequence[str],
    gold_actions: Sequence[str],
    *,
    correct_reward: float = 1.0,
    false_tool_penalty: float = 4.0,
    false_action_penalty: float = 1.5,
    safe_abstain_penalty: float = 0.25,
) -> Dict[str, float]:
    """Cost-weighted routing utility heavily penalizing ungrounded `tool` calls vs safe `abstain`."""
    if len(predicted_actions) != len(gold_actions) or not gold_actions:
        raise ValueError("predicted_actions and gold_actions must have equal non-zero length")
    n = len(gold_actions)
    total_u = 0.0
    correct = 0
    false_tools = 0
    for pred, gold in zip(predicted_actions, gold_actions):
        if pred == gold:
            total_u += correct_reward
            correct += 1
        elif pred == "tool" and gold != "tool":
            total_u -= false_tool_penalty
            false_tools += 1
        elif pred == "abstain" and gold != "abstain":
            total_u -= safe_abstain_penalty
        else:
            total_u -= false_action_penalty
    # Normalize against oracle utility `n * correct_reward`
    return {
        "normalized_utility": total_u / float(n * correct_reward),
        "accuracy": correct / float(n),
        "false_tool_rate": false_tools / float(n),
    }


def build_router(
    router_id: str,
    d_model: int,
    route_token_ids: Optional[Mapping[str, int]] = None,
    **kwargs: Any,
) -> nn.Module:
    key = router_id.strip().upper()
    if key in {"R1", "HEURISTIC_COSINE_GATE"}:
        return HeuristicCosineRouter(d_model=d_model, **kwargs)
    if key in {"R2", "THRESHOLDED_CONFIDENCE"}:
        return ThresholdedConfidenceRouter(d_model=d_model, **kwargs)
    if key in {"R3", "PROMPTED_ROUTE_TOKENS"}:
        if route_token_ids is None:
            raise ValueError("R3 requires `route_token_ids` mapping from TokenizerAdapter")
        return PromptedRouteTokenRouter(route_token_ids=route_token_ids, **kwargs)
    if key in {"R4", "POOLED_MLP_GUARDED"}:
        return PooledMLPRouter(d_model=d_model, **kwargs)
    raise ValueError(f"Unknown router_id {router_id!r}")
