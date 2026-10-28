# Spec Delta

## Purpose

Defines the machine-readable manifest, human-readable synthesis report, and runtime core factory produced by the tournament so the empirically winning architecture becomes the canonical, drop-in offline core of the `recamem` project.

## ADDED Requirements

### Requirement: Provenance-stamped tournament ledger and human synthesis report
Upon completing the 4-stage tournament, the system SHALL atomically write:
1. `reports/evidence/recamem-core-tournament.json`: a schema-validated JSON ledger containing commit hash, hardware fingerprint, dependency versions, `dataset_source` (`real` or `fixture`), every candidate's per-stage metrics (Equal-Byte and Equal-Wall-Clock), bootstrap 95% CIs, Stage-1/2/3/4 gate decisions, eliminated/pruned components with exact reasons, and the final winner's complete hyperparameter specification;
2. `reports/recamem-core-selection.md`: a human-readable engineering report summarizing the track leaderboards, the Shared-vs-Split interference verdict, the Leave-One-Out ablation waterfall, and exact migration bindings for `recamem`.

#### Scenario: Unmeasured or skipped stages cannot crown an unverified winner
- **WHEN** the tournament export runs with any required gate measurement missing, skipped due to load/variance guards, or failed
- **THEN** `recamem-core-tournament.json` SHALL record `outcome="blocked"` or `outcome="incomplete"` naming every missing gate
- **THEN** it SHALL NOT stamp a candidate as `status="verified_winner"` until all required stage gates pass on valid measurements

### Requirement: Canonical RecaMem core manifest and factory contract
The system SHALL write the winning architecture's declarative configuration to `configs/recamem_selected_core.json` and expose a single construction entry point (`build_recamem_core`) that instantiates the winning topology (whether Unified Shared Core or Split Dual-Pathway) behind stable generation, System-1 decision, 5-way action routing, and plastic-memory adapter interfaces consumable by both `Experiment` (`src/model/`) and `recamem` (`recamem/backends.py`, `recamem/engine.py`).

#### Scenario: Selected core instantiates directly from exported manifest
- **WHEN** `build_recamem_core("configs/recamem_selected_core.json")` is invoked in an offline environment
- **THEN** it SHALL construct the exact winning core, tokenizer, decision head, router with deterministic guards, and plastic/episodic memory wiring declared in the manifest
- **THEN** its parameter count, capability flags (`BackendCapabilities`), and state layout SHALL match the values recorded in `reports/evidence/recamem-core-tournament.json`

#### Scenario: Compatibility with recamem engine lifecycle
- **WHEN** the exported RecaMem core is attached to the engine for `chat`, `decide`, `teach`, `recall`, `reset`, and profile `save`/`load`
- **THEN** inference calls SHALL keep base backbone weights frozen (`torch.no_grad()`), route actions deterministically through the calibrated guard layer, and round-trip state through safetensors + JSON metadata without silent fallback
