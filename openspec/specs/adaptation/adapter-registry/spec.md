# adapter-registry Specification

## Purpose

Adapter registry for dynamic discovery, registration, and instantiation of `PlasticAdapter` implementations by name, enabling per-profile adapter selection and third-party extensions.

## Requirements

### Requirement: Adapter registration
The system SHALL provide `register_adapter(name: str, cls: Type[PlasticAdapter])` to register a concrete adapter class under a unique name.

#### Scenario: Register built-in adapter
- **WHEN** `register_adapter("associative_rtu", PlasticAssociativeRTU)` is called
- **THEN** `"associative_rtu"` SHALL be available in `list_adapters()`

### Requirement: Adapter listing
The system SHALL provide `list_adapters() -> List[str]` returning all registered adapter names.

#### Scenario: List shows registered
- **WHEN** `list_adapters()` is called after registration
- **THEN** it SHALL include `"associative_rtu"`

### Requirement: Adapter instantiation by name
The system SHALL provide `get_adapter(name: str, **kwargs) -> PlasticAdapter` to instantiate a registered adapter with config kwargs.

#### Scenario: Create adapter from config
- **WHEN** `get_adapter("associative_rtu", dim=384, heads=4)` is called
- **THEN** it SHALL return a `PlasticAssociativeRTU` instance with those params

### Requirement: Per-profile adapter config
The system SHALL read `plastic_adapter` field from Droid profile config and instantiate the corresponding adapter.

#### Scenario: Profile uses LoRA adapter
- **WHEN** profile config has `plastic_adapter: "lora"` and `lora_rank: 16`
- **THEN** `DroidEngine` SHALL instantiate `LoRAAdapter(rank=16)`

### Requirement: Entry point discovery
The system SHALL support `importlib.metadata` entry points for third-party adapter packages to auto-register on import.

#### Scenario: Third-party adapter auto-registers
- **WHEN** package with entry point `tmt.plastic_adapters: my_adapter = mypkg:MyAdapter` is installed
- **THEN** `MyAdapter` SHALL be available via `get_adapter("my_adapter")` without explicit registration