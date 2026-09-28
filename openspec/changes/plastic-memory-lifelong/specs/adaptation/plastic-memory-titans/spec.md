# Spec Delta

## Purpose

Multi-head associative plastic memory (Titans + RWKV-7 hybrid) replacing the 1D EMA RTU with predictive surprise gating, momentum-augmented surprise updates, and Householder-like selective erasure for bounded lifelong learning without eigen-saturation.

## ADDED Requirements

### Requirement: Multi-head associative state
The system SHALL maintain a multi-head matrix-valued associative state $S \in \mathbb{R}^{H \times D_h \times D_h}$ where $H$ is the number of heads and $D_h = D/H$ is the head dimension, enabling parallel feature subspace tracking.

#### Scenario: Multi-head state initialization
- **WHEN** a new `PlasticAssociativeRTU` is instantiated with `dim=384, heads=4`
- **THEN** the associative state $S$ SHALL be initialized as zeros with shape `(4, 96, 96)`
- **THEN** the momentum buffer SHALL be initialized as zeros with matching shape

### Requirement: Predictive expectation generation
The system SHALL generate a top-down expectation $\hat{e}_t = \text{LayerNorm}(W_{\text{pred}} h_{t-1})$ from the current continuous trace $h_{t-1}$ before observing the next input.

#### Scenario: Expectation before input
- **WHEN** `predict_next()` is called
- **THEN** it SHALL return a normalized prediction vector of dimension `dim` derived from `h_trace`

### Requirement: Surprise computation
The system SHALL compute prediction surprise as $s_t = 1 - \cos(e_t, \hat{e}_t)$ where $e_t$ is the observed input embedding and $\hat{e}_t$ is the predicted expectation, clamped to $[0, 2]$.

#### Scenario: Surprise for novel input
- **WHEN** `compute_surprise(e_t)` is called with an input orthogonal to the expectation
- **THEN** the returned surprise SHALL be $\approx 1.0$

#### Scenario: Surprise for expected input
- **WHEN** `compute_surprise(e_t)` is called with an input aligned with the expectation
- **THEN** the returned surprise SHALL be $\approx 0.0$

### Requirement: Momentum-augmented associative update
The system SHALL update the associative state using momentum-augmented surprise: $M_t = \eta M_{t-1} + \nabla \ell \cdot s_t$, then $S_t = S_{t-1} G_t + M_t$ where $G_t$ is the Householder-like erasure operator.

#### Scenario: Surprise drives stronger update
- **WHEN** `update_associative_memory(e_t, surprise=1.5)` is called
- **THEN** the momentum buffer SHALL increase proportionally to surprise

#### Scenario: Momentum decay between steps
- **WHEN** multiple updates occur with low surprise
- **THEN** momentum SHALL decay by factor 0.85 per step

### Requirement: Householder-like selective erasure
The system SHALL apply a per-head erasure operator $G_h = (I - \alpha_h \hat{k}_h \hat{k}_h^\top) \text{diag}(w_h)$ where $\hat{k}_h$ is the normalized key, $\alpha_h$ is the erasure rate, and $w_h$ is the retention weight, enabling targeted removal of specific feature directions.

#### Scenario: Erasure frees capacity for new associations
- **WHEN** an input shares key direction with an existing association
- **THEN** the erasure operator SHALL reduce the projection along that direction before adding the new value

#### Scenario: Retention weight controls decay speed
- **WHEN** `w_decay` is near 1.0
- **THEN** the state SHALL retain previous associations strongly
- **WHEN** `w_decay` is near 0.0
- **THEN** the state SHALL rapidly forget old associations

### Requirement: 1D trace summary with surprise modulation
The system SHALL maintain a 1D vector trace $h_t$ updated as $h_t = 0.9 h_{t-1} + 0.1 (1 + 0.5 s_t) e_t$, providing a compact summary for retrieval gating.

#### Scenario: Trace update weighted by surprise
- **WHEN** `update_associative_memory` is called with high surprise
- **THEN** the trace update SHALL have larger coefficient

## MODIFIED Requirements

### Requirement: Latent Coupling Interface (from `adaptation/plastic-adapter`)
The system SHALL ingest latent representations from a lightweight pretrained base model and route them through the RTU recurrent memory stream.

#### Scenario: Base model feature extraction
- **WHEN** raw tokens or bytes are fed into the hybrid pipeline
- **THEN** the base model encoder SHALL produce latent vectors without mutating base weights

#### Scenario: Online memory adaptation
- **WHEN** new user text is streamed into the hybrid model
- **THEN** RTU state buffers and plastic weights SHALL adapt online while keeping base language anchors stable