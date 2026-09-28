# Tasks

## 1. PlasticAdapter Interface & Registry

- [ ] 1.1 Create `src/model/plastic_adapter.py` with `PlasticAdapter` ABC (inherits `nn.Module`) and 6 abstract methods: `update`, `retrieve`, `predict_next`, `compute_surprise`, `state_dict`, `load_state_dict`
- [ ] 1.2 Add optional methods with default implementations: `consolidate()`, `get_health()`, `create_snapshot()`, `restore_snapshot()`
- [ ] 1.3 Implement registry: `_ADAPTER_REGISTRY`, `register_adapter(name, cls)`, `get_adapter(name, **kwargs)`, `list_adapters()`
- [ ] 1.4 Implement entry point auto-discovery: scan `importlib.metadata.entry_points(group="tmt.plastic_adapters")` on module import
- [ ] 1.5 Register built-in: `register_adapter("associative_rtu", PlasticAssociativeRTU)` in `rtu.py`
- [ ] 1.6 Verify: `tests/test_plastic_adapter.py` tests ABC compliance, registry, entry points

## 2. PlasticAssociativeRTU Implements Interface

- [ ] 2.1 Update `PlasticAssociativeRTU` class definition: `class PlasticAssociativeRTU(PlasticAdapter)`
- [ ] 2.2 Verify all 6 abstract methods implemented (map existing: `update_associative_memory`→`update`, `retrieve`→`retrieve`, `predict_next`, `compute_surprise`, `state_dict`, `load_state_dict`)
- [ ] 2.3 Add `state_dict()` returning `{**super().state_dict(), "_adapter_version": 1, "_adapter_type": "associative_rtu"}`
- [ ] 2.4 Add `load_state_dict()` handling version migration
- [ ] 2.5 Verify: `tests/test_rtu.py` passes with interface checks

## 3. DroidEngine Integration

- [ ] 3.1 Update `DroidEngine.__init__` to accept `plastic_adapter` instance or create via `get_adapter(config["plastic_adapter"], **config.get("adapter_kwargs", {}))`
- [ ] 3.2 Replace direct `self.memory.method()` calls with `self.memory.method()` (interface same, but type is now `PlasticAdapter`)
- [ ] 3.3 Update type hints: `self.memory: PlasticAdapter`
- [ ] 3.4 Verify: `tests/test_droid_engine.py` works with interface

## 4. Profile & Manager Integration

- [ ] 4.1 Update `DroidManager.create_droid()` to accept `plastic_adapter="associative_rtu"`, `adapter_kwargs={}`
- [ ] 4.2 Update `config.json` schema: add `plastic_adapter`, `adapter_kwargs` fields
- [ ] 4.3 Update `save_profile()` to write `plastic_adapter` and `adapter_kwargs`
- [ ] 4.4 Update `load_profile()` to read and instantiate adapter via registry
- [ ] 4.5 Verify: manager test creates droid with explicit adapter config

## 5. Tests & Validation

- [ ] 5.1 Create `tests/test_plastic_adapter.py`: ABC compliance, registry CRUD, entry point mock, state_dict round-trip
- [ ] 5.2 Extend `tests/test_rtu.py`: verify `PlasticAssociativeRTU` passes ABC checks
- [ ] 5.3 Extend `tests/test_droid_engine.py`: test with mock adapter implementing interface
- [ ] 5.4 Run full test suite: `PYTHONPATH=. /home/asus/miniforge3/envs/ai/bin/pytest tests -v`
- [ ] 5.5 Benchmark: adapter instantiation < 10ms, no performance regression vs direct class