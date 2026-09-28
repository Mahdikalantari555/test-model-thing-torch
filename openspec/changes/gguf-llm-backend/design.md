# Design

## Context

See `proposal.md`. Current: `DroidEngine` with `PlasticAssociativeRTU` + `KnowledgeStore`. `system1-mcp-surface` change adds `DecisionHead` for routing. This change adds GGUF LLM backend for generation.

## Goals / Non-Goals

**Goals:**
- GGUF model loading via llama-cpp-python (CPU default, GPU optional)
- Three memory injection strategies (prompt, hidden, KV cache)
- Dual-mode generation with automatic routing via System-1 decision head
- Fused confidence estimation (retrieval + generation)
- Per-profile GGUF model config

**Non-Goals:**
- No change to ONNX MiniLM anchor or RTU core
- No training/fine-tuning of GGUF models (inference only)
- No change to Streamlit UI (CLI/MCP only)
- No quantization/conversion tools (use pre-quantized GGUF)

## Decisions

### Decision 1: llama-cpp-python as GGUF backend
- **Why**: Mature, actively maintained, supports CPU/GPU (CUDA/Metal), Pythonic API, streaming, KV cache access.
- **Alternatives**: `ctransformers` (less maintained), `llama.cpp` CLI via subprocess (no KV access), `exllamaV2` (GPU only).
- **Implementation**: `GgufBackend` class wrapping `Llama` from `llama_cpp`. Context window 4096 default. `n_gpu_layers=-1` for auto GPU offload.

### Decision 2: Injection strategies as separate methods
- **Why**: Different trade-offs: prompt stuffing simplest but uses context window; hidden state injection requires model surgery but saves tokens; KV cache priming most efficient for many facts but complex.
- **Implementation**:
  - `prompt`: format facts as `Context:\n- fact1\n- fact2\n\nQuery:`
  - `hidden`: register forward hook on target layer, add projected $h_t$ to residual
  - `kv_cache`: encode facts separately, copy K/V into cache before query encoding

### Decision 3: Dual-mode routing via System-1 decision head
- **Why**: `system1-mcp-surface` already adds `DecisionHead` with prototype routing. Extend prototypes to include `fast_recall`, `slow_synthesis`.
- **Implementation**: `DecisionHead` prototypes: `["chat", "recall", "teach", "tool", "fast_recall", "slow_synthesis"]`. `DroidEngine.chat()` calls `decide_action()` first.

### Decision 4: Confidence fusion via harmonic mean
- **Why**: Harmonic mean penalizes imbalance (low retrieval OR low generation drags down score). Arithmetic mean would overestimate.
- **Implementation**: `estimate_confidence(retrieval_conf, generation_conf)` returns `2 * Cr * Cg / (Cr + Cg)` with weights.

### Decision 5: Per-profile GGUF config in DroidManager
- **Why**: Different droids may need different models (small for speed, large for quality).
- **Implementation**: `config.json` adds `gguf_model_path`, `gguf_repo_id`, `gguf_filename`, `gguf_params` (temp, top_p, etc.). `DroidEngine.__init__` reads and initializes `GgufBackend`.

## Risks / Trade-offs

- **[Risk]** llama-cpp-python adds ~50 MB wheel + runtime memory (2–6 GB for model). → **Mitigation**: Optional dependency; only load when profile has GGUF config; document RAM requirements.
- **[Risk]** Hidden state injection requires matching LLM hidden dim (e.g., 4096 for Llama-3B) vs RTU dim (384). → **Mitigation**: Learn projection matrix $W_{proj} \in \mathbb{R}^{384 \times D_{llm}}$ during `teach()` or fixed random projection.
- **[Risk]** KV cache priming needs access to internal cache tensors. → **Mitigation**: `llama_cpp` exposes `llama_get_kv_cache` / `llama_set_kv_cache` in newer versions; fallback to prompt stuffing if unavailable.
- **[Risk]** Dual-mode routing misclassification. → **Mitigation**: Confidence threshold; fallback to slow path on ambiguity; explicit `/fast` `/slow` overrides.

## Migration Plan

1. Add `llama-cpp-python` to `requirements.txt` (optional extra: `pip install -e .[gguf]`)
2. Create `src/model/gguf_backend.py` with `GgufBackend` class
3. Create `src/model/memory_injection.py` with three strategy functions
4. Create `src/model/confidence.py` with fusion logic
5. Update `DroidEngine.__init__`: optional `gguf_backend` parameter
6. Add `generate_with_memory()`, `estimate_confidence()`, `decide_path()` to `DroidEngine`
7. Update `chat()` to route via `decide_action()` → fast/slow path
8. Update `DroidManager`: GGUF config in profile, model download helper
9. Add CLI commands: `tmt-droid download-model`, `tmt-droid chat --mode fast|slow`
10. Tests: `tests/test_gguf_backend.py`, `tests/test_memory_injection.py`, extend `test_droid_engine.py`
11. Run full test suite

## Open Questions

- Should hidden projection $W_{proj}$ be learned or fixed random? → Start fixed random (JL lemma); learn if quality insufficient.
- KV cache priming: encode facts with same tokenizer? → Yes, use GGUF tokenizer for consistency.
- Model download: auto-download on first use or explicit CLI? → Explicit CLI `download-model`; auto-download as opt-in.
- Quantization recommendations: Q4_K_M default? → Yes, best quality/size trade-off per llama.cpp benchmarks.