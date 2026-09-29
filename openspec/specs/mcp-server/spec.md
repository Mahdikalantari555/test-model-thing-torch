# mcp-server Specification

## Purpose

Zero-dependency stdio Model Context Protocol (MCP) server exposing Droid memory recall, teaching, and decision primitives to external AI agents (Claude Code, Cursor, Windsurf) via JSON-RPC 2.0 over standard input/output.

## Requirements

### Requirement: Stdio JSON-RPC 2.0 transport
The system SHALL implement an MCP server that communicates exclusively over stdin/stdout using JSON-RPC 2.0 framing, with all diagnostic logging redirected to stderr.

#### Scenario: JSON-RPC framing integrity
- **WHEN** the server receives a JSON-RPC request on stdin
- **THEN** the server SHALL parse the request and dispatch to the appropriate tool handler
- **THEN** the response SHALL be written to stdout as a single JSON-RPC response object
- **THEN** no diagnostic output SHALL appear on stdout

#### Scenario: Stdio server startup
- **WHEN** the MCP server is launched via `python -m src.cli serve --mcp`
- **THEN** the server SHALL initialize without errors
- **THEN** the server SHALL advertise its available tools via the `tools/list` method

### Requirement: Exposed MCP tools
The system SHALL expose the following tools via the MCP protocol:
- `recall_memory(query, top_k)`: Sub-5ms factual lookup from plastic memory.
- `teach_fact(text, domain)`: Incremental on-device learning with automatic contradiction detection.
- `decide_action(query, options)`: Non-autoregressive System-1 option scoring.
- `list_droids()`: List available Droid profiles.
- `inspect_state(droid_name)`: Audit trace buffers, plastic weight norms, and fact counts.

#### Scenario: Tool invocation
- **WHEN** an agent sends a `tools/call` request for `recall_memory`
- **THEN** the server SHALL execute the recall against the specified Droid
- **THEN** the response SHALL include the matched facts with similarity scores
- **THEN** the response SHALL be a valid JSON-RPC result object

### Requirement: Zero-dependency implementation
The system SHALL implement the MCP server using only Python standard library modules (`json`, `sys`, `socketserver`), without any external web framework or MCP SDK dependencies.

#### Scenario: Stdlib-only implementation
- **WHEN** the MCP server is imported
- **THEN** no third-party packages SHALL be required beyond the existing project dependencies
- **THEN** the server SHALL run in the standard `ai` conda environment without additional installs