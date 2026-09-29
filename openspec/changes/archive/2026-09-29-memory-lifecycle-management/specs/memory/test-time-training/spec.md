# Spec Delta

## Purpose

Test-time training: online updates to plastic associative weights during inference (recall/chat) based on retrieval outcomes and user feedback, without requiring a separate training phase.

## ADDED Requirements

### Requirement: Retrieval-driven associative update
The system SHALL update the associative memory $S$ during `recall()` when a fact is retrieved, strengthening the query-fact association.

#### Scenario: Successful retrieval strengthens association
- **WHEN** a fact is retrieved and returned to user
- **THEN** `PlasticAssociativeRTU.update_associative_memory()` SHALL be called with the fact embedding
- **THEN** surprise factor SHALL be derived from retrieval confidence

### Requirement: Feedback-driven plastic update
The system SHALL update plastic weights immediately on `approve()`/`reject()`/`correct()` feedback, using the corrected or approved proposition as target.

#### Scenario: Approve reinforces retrieved association
- **WHEN** `approve(fact_id)` is called
- **THEN** the fact embedding SHALL be used to update $S$ with positive surprise
- **THEN** the trace $h$ SHALL incorporate the approved fact

#### Scenario: Correct updates with replacement
- **WHEN** `correct(old_id, new_text)` is called
- **THEN** the new proposition embedding SHALL update $S$ with high surprise
- **THEN** the old fact's association SHALL be weakened via erasure

### Requirement: Test-time training budget
The system SHALL limit test-time training compute: max 1 gradient step per recall, max 3 per feedback, with LR schedule decaying over session.

#### Scenario: Budget prevents runaway adaptation
- **WHEN** many retrievals occur in a session
- **THEN** cumulative test-time steps SHALL not exceed `max_ttt_steps_per_session`
- **THEN** learning rate SHALL decay as $lr_t = lr_0 / \sqrt{1 + t}$

### Requirement: Test-time training opt-out
The system SHALL allow disabling test-time training per profile for stable deployment.

#### Scenario: Opt-out freezes plastic weights
- **WHEN** profile config has `test_time_training: false`
- **THEN** no associative updates SHALL occur during recall/feedback
- **THEN** plastic state SHALL only update via explicit `teach()`