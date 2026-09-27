# Proposal

## Why
The engine is trapped inside a Streamlit WebUI and requires an external LLM just to classify intents and route tool calls. Small models (< 100 MB) catastrophically collapse when trained autoregressively, making sub-3ms, deterministic decision routing impossible with existing architecture. Without a CLI or programmable interface, the Droid cannot be embedded into agent toolchains or terminal workflows.

## What Changes
- Implement a **non-autoregressive System-1 decision head** that routes queries (Chat, Recall, Teach, Tool) via hyperspherical cosine argmax in under 3 ms on CPU, with calibrated Shannon entropy gating to reject low-confidence out-of-domain requests.
- Expose a **zero-dependency stdio Model Context Protocol (MCP) server** (`src/mcp_server.py`) implementing JSON-RPC 2.0 tools (`recall_memory`, `teach_fact`, `decide_action`, `inspect_state`) for Claude Code, Cursor, and Windsurf.
- Build a **unified headless CLI** (`src/cli.py` / `tmt-droid`) exposing subcommands: `chat`, `teach`, `decide`, `eval`, `serve --mcp`.
- Create a **portable `.droid` brain archive format** (`src/model/droid_package.py`) bundling `manifest.json`, `config.json`, `memory.safetensors`, and `knowledge.db` with SHA-256 member checksum verification.

## Capabilities

### New Capabilities
- `system1-decision`: Zero-shot prototype routing and calibrated entropy gating replacing external LLM intent classification.
- `mcp-server`: Standardized MCP JSON-RPC 2.0 stdio server exposing Droid memory recall, teaching, and decision primitives.
- `cli`: Unified headless command-line interface for terminal chat, markdown ingestion, System-1 decisions, and daemon serving.
- `portable-brain`: ZIP-bundled `.droid` archive format with integrity verification for cross-device droid state transfer.

### Modified Capabilities
*(none — no existing spec requirements change)*

## Impact
- New files: `src/model/decision_head.py`, `src/mcp_server.py`, `src/cli.py`, `src/model/droid_package.py`.
- `app.py` gains a lightweight embedded REST microdaemon option alongside Streamlit.
- No new heavyweight runtime dependencies; MCP server uses only stdlib `json`, `socketserver`, and `http.server`.
- `requirements.txt` additions: optional `h3-py` for geospatial indexing (deferred to v2.0); otherwise zero new deps.
