# Spec Delta

## Purpose

Unified headless command-line interface (`tmt-droid`) for terminal chat, markdown ingestion, System-1 decisions, and daemon serving without the Streamlit web UI.

## ADDED Requirements

### Requirement: Subcommand routing
The system SHALL support a unified CLI entrypoint with subcommands: `chat`, `teach`, `decide`, `eval`, `serve`, and `package`.

#### Scenario: Chat subcommand
- **WHEN** user runs `python -m src.cli chat --name droid-alpha`
- **THEN** the CLI SHALL start an interactive terminal chat loop with the specified Droid
- **THEN** user input SHALL be sent to the Droid's `chat()` method and the response printed to stdout

#### Scenario: Teach subcommand
- **WHEN** user runs `cat report.md | python -m src.cli teach --name droid-remote-sensing`
- **THEN** the CLI SHALL read markdown from stdin
- **THEN** the CLI SHALL ingest the markdown into the specified Droid's memory
- **THEN** the CLI SHALL print the number of new facts absorbed and the memory state norm

#### Scenario: Decide subcommand
- **WHEN** user runs `python -m src.cli decide "Check soil moisture" --options "irrigate,inspect,wait"`
- **THEN** the CLI SHALL score the options using the System-1 decision head
- **THEN** the CLI SHALL print the selected action, probability distribution, and confidence score

#### Scenario: Serve subcommand
- **WHEN** user runs `python -m src.cli serve --mcp`
- **THEN** the CLI SHALL start the MCP stdio server
- **WHEN** user runs `python -m src.cli serve --rest`
- **THEN** the CLI SHALL start a lightweight REST microdaemon on localhost

#### Scenario: Package subcommand
- **WHEN** user runs `python -m src.cli package export droid-remote-sensing --out ./rs-v1.droid`
- **THEN** the CLI SHALL create a portable `.droid` archive containing the Droid's full state
- **WHEN** user runs `python -m src.cli package import ./rs-v1.droid --name droid-restored`
- **THEN** the CLI SHALL restore the Droid state from the archive

### Requirement: Unix-friendly output
The system SHALL print clean, parseable text to stdout suitable for shell scripting and piping.

#### Scenario: Scriptable output
- **WHEN** the CLI is run in non-interactive mode
- **THEN** all response text SHALL be printed to stdout without ANSI escape codes or progress bars
- **THEN** error messages SHALL be printed to stderr with a clear error prefix