# Proposal

## Why

`KnowledgeStore` keeps normalized embeddings in a cached dense matrix and performs linear dense candidate scoring alongside SQLite FTS5/RRF. This remains O(N) as the store grows. The root `knowledge-store` specification already sets a <5 ms target at 100,000 facts; this change preserves that external target while adding an approximate candidate index and a reproducible recall/latency benchmark. The retrieval path has no cross-encoder reranker or 64-dimensional fast path today.

The project can load GGUF models through `GgufBackend`, but it lacks a maintained, hardware-aware on-device SLM catalog and a generation-side diverse verifier search. Test-time training on retrieval is not the same as inference-time search over candidate answers. An edge profile needs recommendations based on RAM, throughput, and task, plus a bounded way to spend extra inference compute when a small model is uncertain.

## What Changes

- Add `HNSWIndex` for dense candidate search, targeting <5 ms at 100,000 facts against a ~15 ms brute-force reference, while retaining FTS5/RRF and an exact fallback.
- Add `CrossEncoderReranker` targeting +20% relative Precision@5 on a fixed labeled retrieval set.
- Add `MatryoshkaAnchor` with a 64-D fast path and 384-D slow path, targeting 6x faster first-stage retrieval with no more than 5% relative quality loss.
- Add a versioned `EdgeSLMRegistry` with `recommend(ram_gb, min_tok_s, task)` and at least 13 entries: Llama 3.2 1B/3B; Qwen2.5 0.5B/1.5B/3B; SmolLM2 135M/360M/1.7B/3B; Gemma 3 270M/1B/4B; and Phi-4-mini.
- Add `TestTimeCompute` for bounded, diverse candidate-tree search scored by a verifier, with an evaluation target where a 1B model plus search/verifier exceeds a selected 8B baseline on a fixed task benchmark.

## Impact

- **New capabilities:** `memory/hybrid-retrieval`, `model/edge-optimization`, and `model/test-time-compute`.
- **Code areas:** `src/model/knowledge_store.py`, a new hybrid-retrieval module, `src/model/onnx_anchor.py`, `src/model/gguf_backend.py`, a registry module, `src/model/droid.py`, profile configuration, and retrieval/model benchmarks.
- **Compatibility:** Keep `KnowledgeStore.recall()` result shape and RRF lexical contribution. HNSW is an optional dense-candidate implementation with exact fallback; existing model profiles remain loadable.
- **Target outcome:** sub-5 ms retrieval at 100k facts, 6x faster 64-D fast path with <=5% relative quality loss, improved P@5, and edge model selection/compute bounded by device budgets.

## Risks

- HNSW speed and recall depend on parameters, thread count, CPU, and index warmup. Publish the exact hardware, corpus, `efSearch`/`M`, recall@5, and percentile latency; keep an exact mode for sensitive queries.
- The current knowledge-store spec already promises <5 ms at 100k using dense MVM/RRF. The index change must meet that same external SLA and preserve lexical exact-match behavior; benchmark before changing the default backend.
- A 64-D path may not work by naive truncation of an existing 384-D encoder. Require a trained/validated Matryoshka representation or explicit projection and test embedding migration.
- Cross-encoder gains add latency and a model dependency. Rerank only a bounded candidate set and make it optional.
- Published RAM/token rates and benchmark scores are device-, quantization-, and runtime-specific. Store source and measurement provenance; treat catalog values as estimates rather than guarantees.
- TTC can exceed latency/RAM budgets or reward verifier loopholes. Enforce candidate/token/depth limits and evaluate verifier calibration against a fixed 8B reference.

## Research References

- GGUF models for an i5/16 GB system: https://ggufloader.github.io/2025-07-07-top-10-gguf-models-i5-16gb.html
- On-device SLM survey and model details: https://v-chandra.github.io/on-device-llms/
