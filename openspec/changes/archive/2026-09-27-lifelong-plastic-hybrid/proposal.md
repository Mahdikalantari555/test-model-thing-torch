# Proposal: Improved Test-Model-Thing (TMT) Droid Engine

## Why

The original `test-model-thing` (TMT) architecture introduced a novel concept: byte-level continuous test-time training with recurrent trace units (RTUs) and internal state memory. However, in practice, training raw byte weights from scratch collapses after 3-4 steps, loses basic syntax, operates at a slow sequential crawl (~30-50 tokens/sec), and is overly sensitive to manual hyperparameters.

This change upgrades TMT into an improved, rock-solid **Droid Engine**:
1. Anchors the RTU recurrent trace memory to an ultra-lightweight ONNX semantic backbone (`all-MiniLM-L6-v2`, ~22 MB INT8, cached in `~/.cache/huggingface`) so it never collapses or loses language structure.
2. Enables **Conversational Teaching** ("teaches through talking"): the user simply pastes domain text or talks to the Droid, and it absorbs facts and specialized knowledge online during the conversation.
3. Automatically sets learning rates, decay schedules, and step counts without fragile manual tuning.
4. Supports selectable, installable named **Droids** with isolated persistent memory caches and live transparent logs.

## What Changes

- **Conversational In-Chat Absorption**: The Droid can be trained directly in chat through natural interaction or command cues (`/teach`, `/absorb`, or pasting domain paragraphs like Remote Sensing).
- **ONNX Semantic Anchor (Zero Collapse)**: Replaces fragile unanchored random byte projections with an ONNX-accelerated MiniLM embedding manifold (< 25 MB RAM). Base language anchors remain stable while RTU states adapt.
- **Automatic Parameter Tuning**: Learning rates, EMA half-lives, and consolidation steps scale automatically based on input length and entropy.
- **Selectable Droids**: Multi-agent profile management allowing users to create, switch, and install named Droids (e.g. `droid-base`, `droid-remote-sensing`) with their own cached `.safetensors` memory traces.
- **Teacher Integration (Optional)**: Support for OpenAI-compatible local/remote APIs (Ollama, Groq, OpenRouter, or local LFM2.5) to synthesize rich question-choice pairs from raw domain text on demand.
- **Live Transparent Logs**: WebUI stream exposing token throughput, memory state norm ($||h_t||$), prediction confidence, and training loss curves.

## Capabilities

### New Capabilities
- `droid/conversational-teaching`: In-chat continuous online learning and instant domain factual absorption.
- `droid/onnx-anchor`: Sub-25MB ONNX runtime semantic feature extractor preventing representation collapse.
- `droid/profile-management`: Named Droid profile caching, serialization, and memory isolation.
- `training/parallel-scan`: Chunked vectorized sequence scan for 50x faster training throughput.

### Modified Capabilities
- `adaptation/plastic-adapter`: Extends RTU memory to couple with ONNX semantic vectors instead of raw character embeddings.

## Impact

- Upgrades `src/model/droid.py` and `src/model/rtu.py` to support ONNX MiniLM latent streams.
- Refactors `app.py` WebUI into a Droid control station with live logs, named profile switcher, and conversational teaching.
- Stores model caches in `/home/asus/.cache/huggingface` and Droid states in `droids/<droid-name>/`.
