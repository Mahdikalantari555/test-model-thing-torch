# Design

## Context

The engine is trapped inside a Streamlit WebUI and requires an external LLM just to classify intents and route tool calls. Small models (< 100 MB) catastrophically collapse when trained autoregressively, making sub-3ms, deterministic decision routing impossible with existing architecture. See `proposal.md` for the motivation.

## Goals / Non-Goals

**Goals:**
- Implement a non-autoregressive System-1 decision head with sub-3ms CPU latency.
- Expose Droid memory and decisions to IDE agents via stdio MCP.
- Build a unified headless CLI for terminal scripting.
- Create a portable `.droid` archive format for cross-device state transfer.

**Non-Goals:**
- No change to the ONNX anchor or RTU recurrence math.
- No change to the Streamlit UI layout.
- No new heavyweight runtime dependencies; MCP server uses only stdlib modules.
- No multimodal remote sensing support (deferred to v2.0).

## Decisions

### Decision 1: Hyperspherical Cosine Argmax for Intent Routing
- **Why**: Autoregressive token generation for tool selection takes 800–2,200ms and requires network connectivity. A pure matrix multiplication over 384-dim embeddings executes in 1.8–3.3ms on CPU with zero KV-cache and zero catastrophic collapse.
- **Alternatives**: External LLM function calling (slow, network-dependent); fine-tuned transformer classifier (requires backprop, risks collapse); HNSW-based routing (overkill for 5–10 prototypes).
- **Implementation**: `src/model/decision_head.py` with:
  - `DecisionHead` class holding prototype embeddings $P \in \mathbb{R}^{K \times 384}$.
  - `decide(query_emb)` computes scaled dot-product softmax over prototypes, then computes normalized Shannon entropy confidence.
  - `train_probe(X, Y)` computes closed-form Ridge solution $W = (X^\top X + \lambda I)^{-1} X^\top Y$ using `numpy.linalg.solve` in < 5ms.

### Decision 2: Zero-Dependency Stdio MCP Server
- **Why**: Claude Code, Cursor, and Windsurf all support MCP stdio servers. Using the official `mcp` SDK adds a heavy dependency; a stdlib JSON-RPC 2.0 implementation over stdin/stdout is sufficient and keeps the engine self-contained.
- **Alternatives**: FastAPI + SSE (adds 80MB+ of dependencies); official `mcp` Python SDK (adds dependency and complexity); REST microdaemon (not supported by all IDE agents).
- **Implementation**: `src/mcp_server.py` using `sys.stdin`/`sys.stdout` for JSON-RPC 2.0 framing. All logging goes to `sys.stderr` to prevent JSON-RPC framing corruption. Tools: `recall_memory`, `teach_fact`, `decide_action`, `list_droids`, `inspect_state`.

### Decision 3: Unified Headless CLI
- **Why**: Terminal users and shell scripts need a clean, Unix-friendly interface without launching a web browser.
- **Implementation**: `src/cli.py` using `argparse` with subcommands `chat`, `teach`, `decide`, `eval`, `serve`, `package`. `serve --mcp` launches the MCP server; `serve --rest` launches a `ThreadingHTTPServer` on localhost.

### Decision 4: Portable `.droid` Archive
- **Why**: Users need to share, backup, and restore expert Droids across machines. A self-contained ZIP with integrity verification is the simplest portable format.
- **Implementation**: `src/model/droid_package.py` using stdlib `zipfile` and `hashlib.sha256`. Manifest stores member checksums. Extraction validates checksums and rejects zip-slip paths.

## Risks / Trade-offs

- **[Risk]** MCP stdio server may break if any library writes to stdout. → **Mitigation**: Redirect all logging to `sys.stderr` at the top of the module; use `logging.basicConfig(stream=sys.stderr)`.
- **[Risk]** Ridge probe may overfit on few examples. → **Mitigation**: Use default $\lambda = 1.0$ regularization; reject training if N < 5.
- **[Risk]** ZIP archives may be large for Droids with 100k+ facts. → **Mitigation**: Compress knowledge.db with zlib; document expected size in README.

## Migration Plan

1. Create `src/model/decision_head.py` with `DecisionHead` class.
2. Create `src/mcp_server.py` with stdio JSON-RPC 2.0 server.
3. Create `src/cli.py` with unified CLI entrypoint.
4. Create `src/model/droid_package.py` with ZIP archive packaging.
5. Update `app.py` to optionally launch the REST microdaemon.
6. Run existing tests; add `tests/test_decision_head.py`, `tests/test_mcp_server.py`, `tests/test_cli.py`.

## Open Questions

- Should the REST daemon support CORS for browser clients? → Default: no CORS; localhost-only for security.