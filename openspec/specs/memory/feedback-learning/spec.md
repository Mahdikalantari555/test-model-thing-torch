# feedback-learning Specification

## Purpose

User feedback API for retrieved facts: `approve()`, `reject()`, `correct()` — each updates retrieval policy, triggers re-distillation, and adjusts fact confidence/accessibility.

## Requirements

### Requirement: Approve feedback
The system SHALL record user approval of a retrieved fact, increasing its access count, boosting semantic confidence, and reinforcing the retrieval strategy.

#### Scenario: Approve boosts confidence
- **WHEN** `approve(fact_id)` is called
- **THEN** the fact's `access_count` SHALL increment
- **THEN** if semantic, its `confidence` SHALL increase
- **THEN** the active retrieval strategy SHALL get positive reward

### Requirement: Reject feedback
The system SHALL record user rejection, decreasing strategy probability, flagging the fact for review, and triggering alternative retrieval on next similar query.

#### Scenario: Reject flags fact and penalizes strategy
- **WHEN** `reject(fact_id)` is called
- **THEN** the fact SHALL get `needs_review=true`
- **THEN** the active strategy SHALL get negative reward
- **THEN** next similar query SHALL try alternative strategy

### Requirement: Correct feedback with replacement
The system SHALL accept a corrected proposition from the user, superseding the rejected fact and inserting the correction with high novelty.

#### Scenario: Correct inserts replacement
- **WHEN** `correct(old_fact_id, "new corrected proposition")` is called
- **THEN** the old fact SHALL be marked `superseded_by=new_id`
- **THEN** the new proposition SHALL be inserted with `novelty=1.0`
- **THEN** the correction SHALL trigger immediate distillation update

### Requirement: Feedback audit trail
The system SHALL log all feedback events with timestamp, fact ID, action, user correction (if any), and resulting state changes.

#### Scenario: Full feedback traceability
- **WHEN** feedback is given
- **THEN** `get_feedback_log()` SHALL return complete history
- **THEN** log SHALL be exportable for analysis