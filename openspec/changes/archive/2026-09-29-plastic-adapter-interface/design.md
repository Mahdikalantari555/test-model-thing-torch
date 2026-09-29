# Design

## Context

See `proposal.md`. Current: `PlasticAssociativeRTU` in `rtu.py` used directly by `DroidEngine`. This change extracts an interface and registry.

## Goals / Non-Goals

**Goals:**
- Abstract `PlasticAdapter` ABC with minimal required methods
- Registry for dynamic adapter loading
- `PlasticAssociativeRTU` implements interface
- Per-profile adapter selection
- Third-party entry point support

**Non-Goals:**
- No implementation of LoRA/Mamba adapters (future changes)
- No change to ONNX anchor or knowledge store
- No change to `save_profile()` format beyond adapter name field

## Decisions

### Decision 1: Minimal interface — 6 methods
- **Why**: Keeps implementations simple. `update`, `retrieve`, `predict_next`, `compute_surprise`, `state_dict`, `load_state_dict` cover all `DroidEngine` needs.
- **Alternatives**: Larger interface with `consolidate()`, `health_metrics()`, `snapshot()` — pushes complexity to adapters. Defer to future.
- **Implementation**: `abc.ABC` with `@abstractmethod` decorators in `src/model/plastic_adapter.py`.

### Decision 2: Registry as module-level functions
- **Why**: Simple, no singleton class needed. `register_adapter`, `get_adapter`, `list_adapters` in `plastic_adapter.py`.
- **Implementation**: `_ADAPTER_REGISTRY: Dict[str, Type[PlasticAdapter]] = {}`

### Decision 3: Entry points via `importlib.metadata`
- **Why**: Standard Python plugin mechanism. Third parties add `[project.entry-points."tmt.plastic_adapters"]` in pyproject.toml.
- **Implementation**: On module import, scan `entry_points(group="tmt.plastic_adapters")` and auto-register.

### Decision 4: Profile config field `plastic_adapter`
- **Why**: Explicit, versioned, auditable. Default `"associative_rtu"` for backward compat.
- **Implementation**: `DroidEngine.__init__` reads `config.get("plastic_adapter", "associative_rtu")`, calls `get_adapter(name, **config.get("adapter_kwargs", {}))`.

### Decision 5: Adapter kwargs from profile
- **Why**: Different adapters need different params (e.g., LoRA rank, Mamba d_state). Pass via `adapter_kwargs` dict in config.
- **Implementation**: `config.json` adds `plastic_adapter: "associative_rtu"`, `adapter_kwargs: {"dim": 384, "heads": 4}`.

## Risks / Trade-offs

- **[Risk]** Interface too minimal; future adapters need more methods. → **Mitigation**: Add optional `@abstractmethod` with default implementations (e.g., `consolidate()`, `get_health()`) that base class implements as no-op.
- **[Risk]** State dict compatibility across adapter versions. → **Mitigation**: Include `_adapter_version` in state_dict; `load_state_dict` handles migration.
- **[Risk]** Entry point discovery slows startup. → **Mitigation**: Lazy load on first `get_adapter()` call; cache registry.

## Migration Plan

1. Create `src/model/plastic_adapter.py` with `PlasticAdapter` ABC and registry functions
2. Update `PlasticAssociativeRTU` in `rtu.py` to inherit from `PlasticAdapter`
3. Verify all abstract methods implemented (add stubs for any missing)
4. Update `DroidEngine.__init__` to use `get_adapter(profile_config["plastic_adapter"], **profile_config.get("adapter_kwargs", {}))`
5. Update `DroidManager.create_droid()` to accept `plastic_adapter` and `adapter_kwargs`
6. Update `save_profile()` to write `plastic_adapter` and `adapter_kwargs` to config
7. Update `load_profile()` to read and instantiate adapter
8. Add `register_adapter("associative_rtu", PlasticAssociativeRTU)` in `rtu.py` module init
9. Tests: `tests/test_plastic_adapter.py` for ABC compliance, registry, round-trip
10. Run full test suite

## Open Questions

- Should `PlasticAdapter` be a `nn.Module`? → Yes, for parameter registration. Make ABC inherit from `nn.Module`.
- State dict key naming: include adapter type prefix? → Yes, `adapter_type` key in state_dict for migration safety.
- Default `adapter_kwargs` for built-in? → `{"dim": 384, "heads": 4}` matching current defaults.