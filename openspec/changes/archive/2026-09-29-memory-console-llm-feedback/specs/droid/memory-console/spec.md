# Spec Delta

## Purpose

Provides a persistent Memory Console for human-in-the-loop verification of LLM-synthesized responses and full CRUD management of the Droid's knowledge store facts.

## ADDED Requirements

### Requirement: Inline LLM Response Feedback
The system SHALL display an expandable feedback section after each LLM-synthesized assistant response containing auto-extracted proposition checkboxes for selective memory absorption.

#### Scenario: LLM response shows feedback section
- **WHEN** the assistant generates a response with source label indicating LLM synthesis (llm_grounded_memory or llm_general_knowledge)
- **THEN** an expandable "Memory Console" section SHALL appear below the response
- **THEN** the section SHALL contain checkboxes for each extracted proposition

#### Scenario: Proposition extraction from LLM response
- **WHEN** the feedback section is rendered
- **THEN** the system SHALL split the LLM response into sentences
- **THEN** sentences shorter than 30 characters SHALL be excluded
- **THEN** sentences matching conversational markers ("Based on...", "Here is...", "According to...", "In summary...", "To summarize...") SHALL be excluded
- **THEN** remaining sentences SHALL be presented as selectable propositions

#### Scenario: User accepts selected propositions
- **WHEN** user checks one or more proposition checkboxes and clicks "Accept Selected"
- **THEN** the system SHALL call `droid.teach()` with the concatenated selected propositions
- **THEN** the system SHALL display a confirmation with the number of facts absorbed
- **THEN** the feedback section SHALL collapse or show completion state

### Requirement: Memory Console Tab - Fact Browser
The system SHALL provide a dedicated tab for browsing, searching, and managing all facts in the knowledge store.

#### Scenario: Browse all facts
- **WHEN** user opens the Memory Console tab
- **THEN** the system SHALL display a searchable, paginated list of all active facts
- **THEN** each fact row SHALL show: fact text, source, timestamp, novelty score, and superseded status

#### Scenario: Search facts
- **WHEN** user enters a search query
- **THEN** the system SHALL filter facts by text match (case-insensitive substring)
- **THEN** results SHALL update in real-time without page reload

#### Scenario: View fact details
- **WHEN** user clicks a fact row
- **THEN** a detail view SHALL show: full text, source, timestamp, step, novelty, superseded status, valid_until, superseded_by

### Requirement: Memory Console Tab - Fact CRUD
The system SHALL support Create, Read, Update, Delete operations on knowledge store facts.

#### Scenario: Edit fact text
- **WHEN** user clicks "Edit" on a fact and modifies the text
- **THEN** the system SHALL update the fact text in the knowledge store
- **THEN** the fact's timestamp SHALL update to current time
- **THEN** the fact SHALL be re-embedded with the ONNX anchor for semantic search

#### Scenario: Supersede fact
- **WHEN** user clicks "Supersede" on a fact
- **THEN** the system SHALL mark the fact as superseded (superseded=true, valid_until=now)
- **THEN** the fact SHALL be excluded from future recall queries

#### Scenario: Delete fact
- **WHEN** user clicks "Delete" on a fact and confirms
- **THEN** the system SHALL remove the fact from the knowledge store entirely

### Requirement: Memory Console Tab - LLM Teach Workflow
The system SHALL provide a workflow to paste raw text, distill it via LLM, review propositions, and teach selected ones to the Droid.

#### Scenario: LLM distillation of pasted text
- **WHEN** user pastes text into the LLM Teach input area and clicks "Distill"
- **WHEN** LLM Provider is enabled
- **THEN** the system SHALL call `teacher.distill_propositions()` on the text
- **THEN** the resulting propositions SHALL be displayed with checkboxes

#### Scenario: Teach selected distilled propositions
- **WHEN** user selects propositions from distillation results and clicks "Teach Selected"
- **THEN** the system SHALL call `droid.teach()` with the selected propositions
- **THEN** the system SHALL display absorption confirmation with fact count and memory norm

#### Scenario: Fallback without LLM Provider
- **WHEN** LLM Provider is not enabled
- **THEN** the system SHALL use local sentence splitting as fallback for proposition extraction
- **THEN** user can still review and teach the locally-split propositions