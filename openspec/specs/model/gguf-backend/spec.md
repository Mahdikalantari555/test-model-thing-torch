# gguf-backend Specification

## Purpose

GGUF model backend using llama.cpp: load quantized LLMs (2B–8B params, 2–6 GB), run inference on CPU/GPU, manage model lifecycle per Droid profile.

## Requirements

### Requirement: GGUF model loading
The system SHALL load GGUF models from local path or Hugging Face Hub using `llama-cpp-python`, supporting quantization formats Q4_K_M, Q5_K_M, Q8_0.

#### Scenario: Load local GGUF
- **WHEN** `GgufBackend(model_path="models/llama-3.2-3b-q4_k_m.gguf")` is called
- **THEN** the model SHALL load and be ready for inference
- **THEN** context window SHALL be configurable (default 4096)

#### Scenario: Download from Hugging Face
- **WHEN** `GgufBackend(repo_id="bartowski/Llama-3.2-3B-Instruct-GGUF", filename="*q4_k_m*")` is called
- **THEN** the model SHALL download and cache via `huggingface_hub`
- **THEN** subsequent loads SHALL use cached file

### Requirement: Token generation with streaming
The system SHALL generate tokens with optional streaming callback for real-time output.

#### Scenario: Streaming generation
- **WHEN** `generate(prompt, stream=True, callback=fn)` is called
- **THEN** `callback(token)` SHALL be invoked for each generated token
- **THEN** final output SHALL be returned on completion

### Requirement: Model parameter configuration
The system SHALL expose temperature, top_p, top_k, repeat_penalty, max_tokens as configurable parameters.

#### Scenario: Parameters affect output
- **WHEN** `generate(..., temperature=0.7, top_p=0.9)` is called
- **THEN** sampling SHALL use specified parameters

### Requirement: Per-profile model config
The system SHALL store GGUF model path/repo_id and parameters in Droid profile config.

#### Scenario: Profile uses specific model
- **WHEN** Droid profile config has `gguf_model: "models/phi-3-mini-q4.gguf"`
- **THEN** `DroidEngine` SHALL initialize `GgufBackend` with that model