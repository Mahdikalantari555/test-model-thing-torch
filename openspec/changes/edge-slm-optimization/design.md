# Design

## Architecture

Keep SQLite facts/metadata and FTS5 lexical search authoritative. Add an index abstraction for dense candidate retrieval: `HNSWIndex` supplies approximate dense neighbors, while current exact matrix-vector scoring remains available as a fallback and benchmark baseline. Candidate IDs and dense scores are fused with lexical candidates using the existing RRF contract. Index metadata is versioned and can be rebuilt from stored normalized embeddings after restore; deletion/supersession must not return inactive facts.

Use a two-stage retrieval path. `MatryoshkaAnchor` encodes/query-searches in 64 dimensions for the fast path and uses 384 dimensions to rerank/confirm a bounded candidate set. `CrossEncoderReranker` may rerank the final candidate pool and is configurable so low-memory profiles can skip it. Do not truncate an arbitrary 384-D embedding without validation: the 64-D representation must come from an encoder trained for Matryoshka prefixes or a validated projection, and migration should rebuild/index both compatible vector forms.

`EdgeSLMRegistry` is a versioned metadata catalog, separate from model downloads. Entries record model ID/family, parameter count, supported quantization/format, estimated RAM, measured/source-reported tokens per second, task tags, quality references, source URL, and last-verified date. `recommend()` first filters by RAM and requested minimum throughput, then ranks by task-specific metadata and returns estimates plus provenance. Unknown device-specific throughput remains unknown rather than invented.

`TestTimeCompute` wraps the existing generation backend with a strict search budget. It samples diverse candidates, expands promising partial candidates as a bounded tree, deduplicates branches, and asks a verifier to score correctness/evidence. It returns the selected answer and an optional trace containing candidate IDs, verifier scores, token counts, and budget use. Search is opt-in/configurable and never writes unverified candidates into persistent knowledge automatically.

## Components

- **`HNSWIndex`** — optional dense vector index with add/update/delete/rebuild, metadata filtering, configurable search parameters, and exact fallback.
- **`CrossEncoderReranker`** — optional bounded reranker over top candidates; records pre/post ranks and supports standard Precision@5 evaluation.
- **`MatryoshkaAnchor`** — validated 64-D fast representation and 384-D slow representation, shared text-to-vector API, and explicit embedding/index versioning.
- **`EdgeSLMRegistry`** — catalog of at least 13 requested models plus structured RAM/speed/task provenance and a deterministic `recommend()` API.
- **`TestTimeCompute`** — diversity strategy, tree expansion/pruning, verifier interface, strict token/time/candidate budget, output selection, and trace.
- **`KnowledgeStore`/Droid integration** — preserve FTS5, RRF, active-fact filters, and existing return schema while selecting HNSW or exact search from profile configuration.
- **Benchmarks** — 100k-fact latency/recall, labeled P@5, 64-D vs 384-D quality/latency, registry recommendation constraints, and paired 1B+TTC vs 8B inference results.

The catalog shall include these requested entries: Llama 3.2 1B and 3B; Qwen2.5 0.5B, 1.5B, and 3B; SmolLM2 135M, 360M, 1.7B, and 3B; Gemma 3 270M, 1B, and 4B; Phi-4-mini. Source-reported reference fields include Llama 3.2 1B Q4_K_M at about 1.5 GB RAM and 30–50 tokens/s, Qwen2.5 1.5B as a reasoning-oriented option, SmolLM2 1.7B's reported 11T-token training corpus, and Gemma 3 1B at about 0.8 GB with a reported 62.8% GSM8K result. Each is tagged as source-reported, not a universal local benchmark.

## Alternatives

- **Keep exact O(N) MVM only:** remains the correctness baseline but lacks a latency path for larger stores.
- **Replace SQLite/FTS5 with a vector database:** rejected for this change because it would broaden persistence/migration scope and could weaken exact lexical behavior.
- **Make HNSW mandatory:** rejected so minimal installations retain an exact path and do not acquire an unneeded native dependency.
- **Rerank every fact:** rejected due edge latency; rerank only a bounded candidate set.
- **Use 64-D embeddings everywhere:** rejected because it can lose fine semantic distinctions; use a fast first stage with a 384-D slow path.
- **Always load the largest available model:** rejected because RAM and throughput differ by device; recommendations and TTC must honor explicit budgets.
- **Use a single greedy decode instead of TTC:** lower latency, but it gives no mechanism for verifier-guided correction when the small model is uncertain.

## Risks / Trade-offs

- Approximate search can miss rare lexical/semantic matches. FTS5 candidates remain in RRF, HNSW recall is measured against exact search, and profiles can choose exact mode.
- Maintaining two embedding dimensions costs disk and write time. Version each representation and rebuild atomically so partial migrations do not mix dimensions.
- Cross-encoder and verifier runtimes may dominate the edge budget. Provide timeout, model-size, and candidate-count controls and report per-stage timing.
- Catalog model aliases, quantization sizes, licensing, and benchmarks change. Version entries and preserve provenance; recommendations should never imply a guaranteed speed/quality outcome.
- TTC may overfit a weak verifier. Calibrate it on held-out task data and retain baseline generation as a valid fallback.
