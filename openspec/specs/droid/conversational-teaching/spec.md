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

### Requirement: Rolling Discourse Anaphora Resolution
The system SHALL resolve third-person pronouns (it, they, this, these) to their active discourse antecedents during paragraph ingestion so that split propositions remain self-contained.

#### Scenario: Pronoun anchoring in markdown
- **WHEN** user teaches a multi-sentence markdown paragraph where sentence 2 begins with "It"
- **THEN** the system SHALL replace "It" in the split proposition with the active subject from sentence 1
- **THEN** the resulting proposition SHALL be self-contained and semantically complete

#### Scenario: Multi-paragraph context carryover
- **WHEN** user teaches paragraph A ending with subject "Remote sensing"
- **WHEN** user teaches paragraph B beginning with "It uses"
- **THEN** the system SHALL carry over the last active subject across paragraph boundaries
- **THEN** paragraph B's proposition SHALL contain the resolved antecedent

### Requirement: Truth Maintenance Metadata on Facts
The system SHALL attach truth maintenance metadata (`superseded`, `valid_until`, `superseded_by`) to every stored fact during ingestion.

#### Scenario: Metadata attached on teach
- **WHEN** a fact is taught
- **THEN** the stored fact record SHALL include `superseded: false`, `valid_until: null`, and `superseded_by: null` fields
- **WHEN** a fact is superseded by a later contradiction
- **THEN** the stored fact record SHALL update `superseded: true`, `valid_until: <timestamp>`, and `superseded_by: <fact_id>`
