# Design: Improved Test-Model-Thing (TMT) Droid Engine

## Context

See `proposal.md` for background. The core innovation of `test-model-thing` is internal state + recurrent trace units (RTU) undergoing test-time training. To make it viable for practical domain specialization and in-chat teaching, we anchor the RTU to a quantized ONNX MiniLM feature space and automate all training mechanics.

## Goals / Non-Goals

**Goals:**
- Enable continuous conversational teaching directly in chat without training collapse or model silence.
- Execute sub-15ms semantic feature extraction using ONNX MiniLM cached in `~/.cache/huggingface`.
- Automatically calibrate training parameters (learning rate, decay rates, step counts).
- Support named Droids (`droids/<name>/`) with isolated state persistence and transparent audit logs.
- Provide optional teacher hooks for OpenAI-compatible APIs (Ollama/Groq/etc.) to generate structured training pairs.

**Non-Goals:**
- Training 100M+ autoregressive models from scratch.
- Forcing external cloud dependencies (runs 100% offline with local ONNX anchor).

## Decisions

### 1. ONNX MiniLM (INT8) as the Semantic Anchor
- **Decision**: Use `onnxruntime` to run `all-MiniLM-L6-v2` (~22 MB). It produces stable 384-dimensional dense vectors.
- **Why**: Eliminates raw character/byte representation collapse. The base language manifold remains frozen while RTU states learn.

### 2. Conversational In-Chat Absorption Loop
- **Decision**: In chat, user messages containing definitions or domain knowledge are split into factual chunks, passed through the ONNX anchor, and accumulated into the RTU decay memory in a single fast pass (< 50ms).
- **Why**: Zero manual steps. Users teach the Droid by simply talking to it.

### 3. Automatic Hyperparameter Heuristic
- **Decision**: Effective learning rate $\eta = \eta_0 / \sqrt{1 + \text{tokens}/200}$, decay rate dynamically tuned based on input entropy.
- **Why**: Prevents gradients from exploding or zeroing out stop probabilities.

### 4. Droid Profile Structure
- Stored under `droids/<droid-name>/`:
  - `config.json`: metadata, step counter, active base model ID.
  - `memory.safetensors`: RTU `states`, `decaytrace`, `embedtrace`.
  - `train_log.json`: audit history of absorbed texts and metrics.

## Risks / Trade-offs

- **[Risk] Cold start without HF cache**: First run needs MiniLM ONNX files in `~/.cache/huggingface`.
  - *Mitigation*: Auto-download and cache to `~/.cache/huggingface` once, with graceful fallback.
- **[Risk] High memory saturation**: Long sessions might accumulate state norms indefinitely.
  - *Mitigation*: LayerNorm and EMA decay naturally bound state magnitude: $||h_t|| \le \frac{1}{1 - \text{decay}} ||z_t||$.
