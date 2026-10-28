"""Modular candidate cores, tokenizers, objectives, readouts, routers, and memory arms."""

from src.model.cores.contracts import (
    ACTION_NAMES,
    STATE_COMPLEXITY_CLASSES,
    ActionRouter,
    AuxiliaryObjective,
    CandidateLM,
    CoreOutput,
    DecisionReadout,
    DecisionResult,
    EncodedSpan,
    MemoryArm,
    MemoryFootprint,
    RoutingResult,
    SequenceCore,
    StateSpec,
    TokenizerAdapter,
)
from src.model.cores.param_matcher import (
    ALL_CORE_IDS,
    MatchedCoreConfig,
    canonical_core_id,
    estimate_candidate_lm_params,
    solve_all_cores_for_tier,
    solve_core_config,
)
from src.model.cores.tokenizer import (
    CompactBPETokenizer,
    UTF8ByteTokenizer,
    build_tokenizer,
)
from src.model.cores.registry import (
    CORE_METADATA,
    build_candidate_lm,
    build_core,
    list_cores,
)

__all__ = [
    "ACTION_NAMES",
    "ALL_CORE_IDS",
    "STATE_COMPLEXITY_CLASSES",
    "ActionRouter",
    "AuxiliaryObjective",
    "CandidateLM",
    "CompactBPETokenizer",
    "CoreOutput",
    "DecisionReadout",
    "DecisionResult",
    "EncodedSpan",
    "MatchedCoreConfig",
    "MemoryArm",
    "MemoryFootprint",
    "RoutingResult",
    "SequenceCore",
    "StateSpec",
    "TokenizerAdapter",
    "UTF8ByteTokenizer",
    "build_tokenizer",
    "canonical_core_id",
    "estimate_candidate_lm_params",
    "solve_all_cores_for_tier",
    "solve_core_config",
]
