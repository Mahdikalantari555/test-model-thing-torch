# Proposal

## Why
The current `PlasticAssociativeRTU` (from `plastic-memory-lifelong`) is tightly coupled to the ONNX MiniLM anchor and a specific associative memory architecture. To enable future replacement with LoRA adapters, Mamba/SSM layers, or other plastic memory mechanisms — without rewriting `DroidEngine` — we need a clean adapter interface that decouples the plastic memory implementation from the Droid orchestration layer.

## What Changes
- Define `PlasticAdapter` abstract base class with standard interface: `update(x)`, `retrieve(q)`, `predict_next()`, `compute_surprise(x)`, `state_dict()`, `load_state_dict()`
- Refactor `PlasticAssociativeRTU` to implement `PlasticAdapter`
- Add adapter registry for dynamic loading: `register_adapter(name, cls)`, `get_adapter(name)`
- Add per-profile adapter selection in config: `plastic_adapter: "associative_rtu" | "lora" | "mamba" | "custom"`
- Add adapter-agnostic checkpointing in `save_profile()`/`load_profile()`

## Capabilities

### New Capabilities
- `adaptation/plastic-adapter-interface`: Abstract interface and registry for plastic memory adapters
- `adaptation/adapter-registry`: Dynamic adapter discovery and instantiation

### Modified Capabilities
- `adaptation/plastic-adapter`: Existing spec updated to require `PlasticAdapter` interface compliance
- `adaptation/plastic-memory-titans`: Updated to implement `PlasticAdapter`
- `droid/profile-management`: Extended with `plastic_adapter` config field

## Impact
- `src/model/plastic_adapter.py`: New module with ABC, registry, base classes
- `src/model/rtu.py`: `PlasticAssociativeRTU` inherits from `PlasticAdapter`
- `src/model/droid.py`: Uses `PlasticAdapter` interface methods instead of concrete class
- `src/model/droid_manager.py`: Adapter selection per profile
- No new external dependencies