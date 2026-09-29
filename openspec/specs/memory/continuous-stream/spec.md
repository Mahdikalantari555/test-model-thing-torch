# continuous-stream Specification

## Purpose

Continuous learning stream: background absorption of interaction logs, chat history, and external data sources into the Droid's memory without explicit `teach()` calls, with configurable rate limits and novelty filtering.

## Requirements

### Requirement: Stream ingestion from interaction log
The system SHALL consume a stream of text interactions (user messages, assistant responses, tool outputs) and absorb propositions automatically.

#### Scenario: Stream processes chat history
- **WHEN** `start_continuous_stream(log_source)` is called with chat log path
- **THEN** the stream SHALL extract propositions from each interaction
- **THEN** propositions SHALL pass through surprise gate before insertion

### Requirement: Configurable rate limiting
The system SHALL limit stream ingestion rate (facts per minute) and total buffer size to prevent memory saturation.

#### Scenario: Rate limit enforced
- **WHEN** stream produces facts faster than `max_facts_per_minute`
- **THEN** excess facts SHALL be queued or dropped per policy
- **THEN** `get_stream_stats()` SHALL report dropped count

### Requirement: Novelty filtering in stream
The system SHALL apply a higher surprise threshold for stream ingestion than interactive `teach()` to avoid absorbing redundant conversational filler.

#### Scenario: Stream uses higher surprise threshold
- **WHEN** `continuous_learn()` processes streamed text
- **THEN** surprise threshold SHALL be $\theta_{\text{stream}} = 0.6$ (vs 0.4 for teach)
- **THEN** only genuinely novel stream facts SHALL enter episodic store

### Requirement: Stream pause/resume and status
The system SHALL support pausing, resuming, and querying the stream status.

#### Scenario: Stream controllable
- **WHEN** `pause_stream()` / `resume_stream()` called
- **THEN** ingestion SHALL stop/start immediately
- **WHEN** `get_stream_status()` called
- **THEN** it SHALL return `running`, `facts_processed`, `facts_absorbed`, `queue_depth`