# system1-decision Specification

## Purpose

Non-autoregressive System-1 decision head that routes user queries to discrete actions (Chat, Recall, Teach, Tool) via hyperspherical cosine argmax in under 3 ms on CPU, with calibrated Shannon entropy gating to reject low-confidence out-of-domain requests.

## Requirements

### Requirement: Zero-shot prototype routing
The system SHALL classify a user query into one of K predefined intent categories using cosine similarity between the query embedding and prototype anchor embeddings, executing in under 3 ms on CPU.

#### Scenario: Intent routing latency
- **WHEN** a query is embedded via the ONNX MiniLM anchor
- **WHEN** cosine similarity is computed against K=5 prototype anchors
- **THEN** the classification SHALL complete in under 3 ms on CPU
- **THEN** the returned intent SHALL be the argmax of the cosine similarity scores

#### Scenario: Zero-shot prototype accuracy
- **WHEN** a query clearly matches one of the K intent categories
- **THEN** the top-1 cosine similarity score SHALL exceed 0.65
- **THEN** the returned intent SHALL match the expected category

### Requirement: Calibrated confidence gating
The system SHALL compute a calibrated confidence score using normalized Shannon entropy and reject ambiguous or out-of-domain requests when confidence falls below a threshold.

#### Scenario: Normalized entropy confidence
- **WHEN** a query produces a softmax distribution over K intent categories
- **THEN** the confidence score SHALL be computed as $1 - \mathcal{H}_{\text{norm}}$ where $\mathcal{H}_{\text{norm}} = \frac{-\sum p_k \ln p_k}{\ln K}$
- **THEN** the confidence score SHALL be bounded between 0.0 and 1.0

#### Scenario: Out-of-domain rejection
- **WHEN** a query has no clear match to any intent category
- **WHEN** the maximum cosine similarity is below 0.35
- **THEN** the system SHALL reject the decision and return an "ambiguous" or "out-of-domain" status
- **THEN** the system SHALL NOT execute any tool or action based on the rejected decision

### Requirement: Closed-form Ridge linear probe for custom tools
The system SHALL train a linear probe mapping query embeddings to custom tool action probabilities using closed-form Ridge regression, completing in under 5 ms on CPU without backpropagation.

#### Scenario: Custom tool training
- **WHEN** user provides N labeled examples of (query, tool_action) pairs
- **THEN** the system SHALL compute the closed-form Ridge solution $W = (X^\top X + \lambda I)^{-1} X^\top Y$
- **THEN** training SHALL complete in under 5 ms on CPU
- **THEN** the resulting probe SHALL be serializable to disk as a safetensors tensor

### Requirement: Structured decision output
The system SHALL return a structured decision object containing the selected action, probability distribution, confidence score, and execution metadata.

#### Scenario: Structured decision response
- **WHEN** a decision is made
- **THEN** the response SHALL include `action` (string), `probabilities` (list of floats), `confidence` (float), and `latency_ms` (float)
- **THEN** the response SHALL be JSON-serializable