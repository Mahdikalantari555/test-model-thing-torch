# droid/conversational-teaching Specification

## Purpose
Enables the model to absorb specialized domain knowledge (e.g. Remote Sensing concepts) directly through natural conversational interaction without manual training configuration or model collapse.

## Requirements

### Requirement: In-Chat Knowledge Ingestion
The system SHALL accept unstructured domain text directly in the chat stream and update persistent RTU memory states during the conversation.

#### Scenario: User provides domain definition paragraph
- **WHEN** user sends or attaches a domain text (e.g., Remote Sensing definition)
- **THEN** the system SHALL extract semantic features, update recurrent RTU trace vectors, and log the absorption confirmation with token count and memory state norm

#### Scenario: Zero parameter collapse after multiple teaching rounds
- **WHEN** conversational teaching is repeated over 5 or more successive turns
- **THEN** output logits and stop probabilities SHALL remain bounded and English syntax stability SHALL not degrade to empty or repeated characters

### Requirement: Automatic Hyperparameter Selection
The system SHALL automatically determine the learning rate, decay half-life, and consolidation iterations based on the ingested text length without user prompt.

#### Scenario: Auto-scaled update
- **WHEN** a text of arbitrary length $N$ is provided for conversational teaching
- **THEN** the system SHALL apply an effective learning rate scaled by $1 / \sqrt{N}$ and log the chosen values
