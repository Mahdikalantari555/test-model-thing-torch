# Proposal

## Why
The current `RTUMemoryBlock` uses a simple 1D vector EMA ($h_t = \lambda h_{t-1} + e_t$) which suffers from eigen-saturation, cannot overwrite stale associations, and is completely bypassed during factual retrieval to avoid query centroid pollution. This leaves the plastic recurrent state dead at inference time and provides no mechanism for forgetting, consolidation, or selective erasure — essential for bounded lifelong learning on edge devices.

## What Changes
- Replace `RTUMemoryBlock` with a multi-head associative memory (Titans + RWKV-7 hybrid) supporting predictive surprise gating, momentum-augmented updates, and Householder-like selective erasure
- Add surprise-based episodic admission gate so only novel propositions enter the discrete knowledge store
- Implement anti-pollution retrieval: semantic similarity as hard gate, contextual alignment as bounded multiplicative boost
- Add Ebbinghaus power-law forgetting with sleep consolidation (background pruning + associative replay)
- Add state snapshot/versioning for memory checkpoints and rollback
- Add memory health metrics (saturation, forgetting rate, interference, quality)

## Capabilities

### New Capabilities
- `adaptation/plastic-memory-titans`: Multi-head associative RTU with predictive surprise gating and selective erasure
- `adaptation/memory-forgetting`: Ebbinghaus power-law decay, sleep consolidation, and pruning
- `adaptation/memory-health`: Saturation, interference, forgetting-rate, and quality metrics
- `adaptation/memory-snapshots`: State checkpointing, versioning, and rollback

### Modified Capabilities
- `adaptation/plastic-adapter`: Upgrades RTU recurrence math; changes `RTUMemoryBlock` interface to `PlasticAssociativeRTU`
- `droid/conversational-teaching`: Updates `teach()` to use surprise admission gate and `recall()` to use anti-pollution contextual scoring

## Impact
- `src/model/rtu.py`: New `PlasticAssociativeRTU` class replacing `RTUMemoryBlock`
- `src/model/droid.py`: `teach()` uses surprise gate; `recall()` uses multiplicative contextual scoring; new `consolidate_memory()` method
- `tests/test_rtu.py`: New unit tests for associative memory, surprise, erasure
- `tests/test_droid_engine.py`: Integration tests for surprise gate, consolidation, snapshots
- No new external dependencies (stdlib + torch only)