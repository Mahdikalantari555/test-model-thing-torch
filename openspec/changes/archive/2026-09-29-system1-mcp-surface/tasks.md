# Tasks

## Phase 1: System-1 Decision Head
- [ ] T1.1 Create `src/model/decision_head.py` with `DecisionHead` class
- [ ] T1.2 Implement zero-shot prototype routing via cosine similarity
- [ ] T1.3 Implement calibrated Shannon entropy confidence gating
- [ ] T1.4 Implement closed-form Ridge linear probe training
- [ ] T1.5 Implement structured decision output serialization

## Phase 2: MCP Server
- [ ] T2.1 Create `src/mcp_server.py` with stdio JSON-RPC 2.0 server
- [ ] T2.2 Implement `tools/list` method advertising available tools
- [ ] T2.3 Implement `tools/call` dispatch for `recall_memory`, `teach_fact`, `decide_action`, `list_droids`, `inspect_state`
- [ ] T2.4 Redirect all logging to `sys.stderr` to prevent JSON-RPC framing corruption
- [ ] T2.5 Create `tests/test_mcp_server.py` with unit tests for tool dispatch

## Phase 3: Headless CLI
- [ ] T3.1 Create `src/cli.py` with unified CLI entrypoint
- [ ] T3.2 Implement `chat` subcommand with interactive terminal loop
- [ ] T3.3 Implement `teach` subcommand with stdin markdown ingestion
- [ ] T3.4 Implement `decide` subcommand with System-1 option scoring
- [ ] T3.5 Implement `serve --mcp` and `serve --rest` subcommands
- [ ] T3.6 Implement `package export` and `package import` subcommands
- [ ] T3.7 Create `tests/test_cli.py` with unit tests for subcommand routing

## Phase 4: Portable Brain Packages
- [ ] T4.1 Create `src/model/droid_package.py` with ZIP archive packaging
- [ ] T4.2 Implement SHA-256 member checksum verification in `manifest.json`
- [ ] T4.3 Implement zip-slip path traversal protection during extraction
- [ ] T4.4 Create `tests/test_droid_package.py` with unit tests for export/import

## Phase 5: Validation
- [ ] T5.1 Benchmark decision head latency at K=5, K=10, K=20 prototypes
- [ ] T5.2 Verify out-of-domain rejection behavior
- [ ] T5.3 Verify MCP server JSON-RPC framing integrity
- [ ] T5.4 Run full test suite: `PYTHONPATH=. /home/asus/miniforge3/envs/ai/bin/pytest tests`