"""Registry for all seven Track-1 candidate sequence-mixing cores (`C0`..`C6`)."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

import torch.nn as nn

from src.model.cores.attention_cores import NeedleSANCore, SoftmaxTransformerCore
from src.model.cores.contracts import CandidateLM
from src.model.cores.diagonal_cores import StackedGatedRTUCore, UnstackedRTUCore
from src.model.cores.matrix_cores import GatedDeltaMatrixCore, RetNetDecayMatrixCore
from src.model.cores.param_matcher import (
    ALL_CORE_IDS,
    MatchedCoreConfig,
    canonical_core_id,
    solve_core_config,
)
from src.model.cores.ssm_core import SelectiveSSMCore

CORE_METADATA: Dict[str, Dict[str, Any]] = {
    "C0": {
        "slug": "unstacked_rtu",
        "title": "Repo Unstacked Diagonal RTU",
        "complexity_class": "constant_vector",
        "supports_online_rtrl": True,
    },
    "C1": {
        "slug": "stacked_gated_rtu",
        "title": "Candidate B Stacked Input-Gated RTU",
        "complexity_class": "constant_vector",
        "supports_online_rtrl": True,
    },
    "C2": {
        "slug": "selective_ssm",
        "title": "Mamba-Style Selective Diagonal SSM",
        "complexity_class": "constant_vector",
        "supports_online_rtrl": False,
    },
    "C3": {
        "slug": "retnet_decay_matrix",
        "title": "RetNet-Style Pure-Decay Matrix Recurrence",
        "complexity_class": "constant_matrix",
        "supports_online_rtrl": False,
    },
    "C4": {
        "slug": "gated_delta_matrix",
        "title": "Candidate A Gated-Delta Matrix Core",
        "complexity_class": "constant_matrix",
        "supports_online_rtrl": False,
    },
    "C5": {
        "slug": "needle_san",
        "title": "Needle-Style Sliding-Window QK-Norm SAN",
        "complexity_class": "sliding_window_kv",
        "supports_online_rtrl": False,
    },
    "C6": {
        "slug": "softmax_transformer",
        "title": "Parameter-Matched Softmax Transformer",
        "complexity_class": "linear_kv",
        "supports_online_rtrl": False,
    },
}


def list_cores() -> List[str]:
    return list(ALL_CORE_IDS)


def build_core(
    config_or_id: MatchedCoreConfig | Mapping[str, Any] | str,
    *,
    chunk_size: int = 64,
    **overrides: Any,
) -> nn.Module:
    """Instantiate a `SequenceCore` (`C0`..`C6`) from a `MatchedCoreConfig`, dict, or ID."""
    if isinstance(config_or_id, str):
        tier = overrides.pop("tier", "small")
        vocab_size = int(overrides.pop("vocab_size", 4096))
        use_conv = bool(overrides.pop("use_conv", True))
        cfg_dict = solve_core_config(config_or_id, tier_or_budget=tier, vocab_size=vocab_size, use_conv=use_conv).to_dict()
    elif isinstance(config_or_id, MatchedCoreConfig):
        cfg_dict = config_or_id.to_dict()
    else:
        cfg_dict = dict(config_or_id)
    cfg_dict.update(overrides)

    cid = canonical_core_id(str(cfg_dict["core_id"]))
    d_model = int(cfg_dict["d_model"])
    n_layers = int(cfg_dict["n_layers"])
    n_heads = int(cfg_dict.get("n_heads", 4))
    d_ff = int(cfg_dict.get("d_ff", 0))
    use_conv = bool(cfg_dict.get("use_conv", True))
    conv_kernel = int(cfg_dict.get("conv_kernel", 4))
    window_size = int(cfg_dict.get("window_size", 256))
    state_dim = int(cfg_dict.get("state_dim", 16))

    if cid == "C0":
        return UnstackedRTUCore(d_model=d_model, n_layers=n_layers, d_ff=d_ff, chunk_size=chunk_size)
    if cid == "C1":
        return StackedGatedRTUCore(d_model=d_model, n_layers=n_layers, d_ff=d_ff, chunk_size=chunk_size)
    if cid == "C2":
        return SelectiveSSMCore(
            d_model=d_model,
            n_layers=n_layers,
            d_ff=d_ff,
            state_dim=state_dim,
            conv_kernel=conv_kernel,
            chunk_size=chunk_size,
        )
    if cid == "C3":
        return RetNetDecayMatrixCore(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            chunk_size=chunk_size,
        )
    if cid == "C4":
        return GatedDeltaMatrixCore(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            use_conv=use_conv,
            conv_kernel=conv_kernel,
            chunk_size=chunk_size,
        )
    if cid == "C5":
        return NeedleSANCore(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            window_size=window_size,
        )
    if cid == "C6":
        return SoftmaxTransformerCore(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
        )
    raise ValueError(f"Unsupported core_id {cid!r}")


def build_candidate_lm(
    config_or_id: MatchedCoreConfig | Mapping[str, Any] | str,
    *,
    tied_embeddings: bool = True,
    include_stop_head: bool = False,
    chunk_size: int = 64,
    **overrides: Any,
) -> CandidateLM:
    if isinstance(config_or_id, str):
        vocab_size = int(overrides.get("vocab_size", 4096))
    elif isinstance(config_or_id, MatchedCoreConfig):
        vocab_size = config_or_id.vocab_size
    else:
        vocab_size = int(config_or_id["vocab_size"])
    core = build_core(config_or_id, chunk_size=chunk_size, **overrides)
    return CandidateLM(
        core=core,
        vocab_size=vocab_size,
        tied_embeddings=tied_embeddings,
        include_stop_head=include_stop_head,
    )
