
try:
    from src.model.plastic_adapter import PlasticAdapter, register_adapter, get_adapter, list_adapters
except:
    try:
        from .plastic_adapter import PlasticAdapter, register_adapter, get_adapter, list_adapters
    except:
        PlasticAdapter = None
        register_adapter = None
        get_adapter = None
        list_adapters = lambda: []

from src.model.rtu import (
    Model,
    Layer,
    Encoder,
    Decoder,
    CustomAdamW,
    loss_variance,
    loss_pred_mse,
    loss_crossentropy,
    loss_stop_mse,
    compute_losses,
    RTUMemoryBlock,
    PlasticAssociativeRTU,
)

try:
    from src.model.droid import DroidEngine
except:
    try:
        from .droid import DroidEngine
    except:
        DroidEngine = None

try:
    from src.model.droid_manager import DroidManager
except:
    try:
        from .droid_manager import DroidManager
    except:
        DroidManager = None

try:
    from src.model.knowledge_store import KnowledgeStore
except:
    try:
        from .knowledge_store import KnowledgeStore
    except:
        KnowledgeStore = None

try:
    from src.model.decision_head import DecisionHead
except:
    try:
        from .decision_head import DecisionHead
    except:
        DecisionHead = None

try:
    from src.model.droid_package import DroidPackage
except:
    try:
        from .droid_package import DroidPackage
    except:
        DroidPackage = None

try:
    from src.model.gguf_backend import GgufBackend
except:
    try:
        from .gguf_backend import GgufBackend
    except:
        GgufBackend = None

try:
    from src.model.memory_lifecycle import (
        SnapshotManager, MergeEngine, SemanticStore, ContradictionResolver,
        RetrievalPolicy, ContinuousStream
    )
except:
    try:
        from .memory_lifecycle import (
            SnapshotManager, MergeEngine, SemanticStore, ContradictionResolver,
            RetrievalPolicy, ContinuousStream
        )
    except:
        SnapshotManager = None
        MergeEngine = None
        SemanticStore = None
        ContradictionResolver = None
        RetrievalPolicy = None
        ContinuousStream = None

try:
    from src.model.memory_injection import inject_prompt, inject_hidden, inject_kv_cache, select_strategy
except:
    try:
        from .memory_injection import inject_prompt, inject_hidden, inject_kv_cache, select_strategy
    except:
        inject_prompt = None
        inject_hidden = None
        inject_kv_cache = None
        select_strategy = None

try:
    from src.model.confidence import (
        retrieval_confidence, generation_confidence, fuse_confidence,
        confidence_tier, estimate_confidence
    )
except:
    try:
        from .confidence import (
            retrieval_confidence, generation_confidence, fuse_confidence,
            confidence_tier, estimate_confidence
        )
    except:
        retrieval_confidence = None
        generation_confidence = None
        fuse_confidence = None
        confidence_tier = None
        estimate_confidence = None

__all__ = [
    "Model",
    "Layer",
    "Encoder",
    "Decoder",
    "CustomAdamW",
    "loss_variance",
    "loss_pred_mse",
    "loss_crossentropy",
    "loss_stop_mse",
    "compute_losses",
    "RTUMemoryBlock",
    "PlasticAssociativeRTU",
    "PlasticAdapter",
    "DroidEngine",
    "DroidManager",
    "KnowledgeStore",
    "DecisionHead",
    "DroidPackage",
    "GgufBackend",
    "SnapshotManager",
    "MergeEngine",
    "SemanticStore",
    "ContradictionResolver",
    "RetrievalPolicy",
    "ContinuousStream",
]
