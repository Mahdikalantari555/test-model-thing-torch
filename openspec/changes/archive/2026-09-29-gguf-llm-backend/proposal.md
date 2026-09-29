# Proposal

## Why
The current system uses a frozen ONNX MiniLM anchor for semantic encoding and a plastic RTU for memory, but lacks a generative LLM backend for synthesis, reasoning, and open-ended generation. Adding a GGUF/llama.cpp backend enables on-device generation with memory injection (prompt stuffing, hidden state injection, KV cache priming), confidence estimation, and a unified fast-slow architecture where System-1 (RTU) handles retrieval/routing and System-2 (GGUF LLM) handles synthesis.

## What Changes
- Add GGUF model loader using llama.cpp Python bindings (llama-cpp-python)
- Implement memory injection: RTU facts → prompt context, hidden state injection, KV cache priming
- Add confidence estimation: retrieval confidence + LLM self-consistency scoring
- Implement dual-mode generation: fast RTU-only for factual recall, slow RTU+LLM for synthesis
- Add model management: download, quantize, switch GGUF models per profile

## Capabilities

### New Capabilities
- `model/gguf-backend`: GGUF model loading, inference, and management via llama.cpp
- `model/memory-injection`: RTU memory → prompt/hidden/KV injection strategies
- `model/confidence-estimation`: Combined retrieval + generation confidence scoring
- `model/dual-mode-generation`: Fast (RTU-only) vs slow (RTU+LLM) paths

### Modified Capabilities
- `droid/conversational-teaching`: Optional LLM-assisted proposition extraction
- `droid/profile-management`: Per-profile GGUF model configuration

## Impact
- `src/model/gguf_backend.py`: New module for llama.cpp integration
- `src/model/droid.py`: New `generate()`, `synthesize()`, `estimate_confidence()` methods
- `src/model/droid_manager.py`: GGUF model config per profile
- `requirements.txt`: Add `llama-cpp-python`
- Optional: `huggingface_hub` for model downloads
- GPU acceleration optional (CUDA/Metal); CPU fallback default