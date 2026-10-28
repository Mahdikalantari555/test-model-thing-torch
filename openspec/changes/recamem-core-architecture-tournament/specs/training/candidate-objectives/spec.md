# Spec Delta

## Purpose

Isolates tokenization, auxiliary latent-prediction losses, online RTRL eligibility traces versus micro-batched chunked BPTT, and multi-task batch interleaving so each training mechanism can be ablated independently on CPU hardware.

## ADDED Requirements

### Requirement: Interchangeable byte and compact BPE tokenization
The system SHALL support both raw UTF-8 byte tokenization (`V=256`) and deterministic compact BPE tokenization (`V=4096`) equipped with dedicated control tokens (`EOS`, `OPT`, `SEP`, and 5 `ROUTE_*` action tokens), while tracking the exact underlying UTF-8 byte length of every encoded span for cross-tokenizer evaluation.

#### Scenario: Lossless UTF-8 byte accounting
- **WHEN** an evaluation document is encoded under either the byte tokenizer (`V=256`) or the compact BPE tokenizer (`V=4096`)
- **THEN** the tokenizer wrapper SHALL record the exact UTF-8 byte count of the evaluated text span
- **THEN** special control tokens (`EOS`, `OPT`, `SEP`, `ROUTE_*`) SHALL occupy distinct, non-overlapping token IDs

### Requirement: Pluggable language-modeling and latent auxiliary objectives
The system SHALL support three selectable training objective configurations on top of any candidate core:
1. `O1` (`ce_eos`): pure autoregressive token cross-entropy with explicit `EOS` termination and no auxiliary stop or latent-MSE head;
2. `O2` (`repo_four_loss`): token cross-entropy plus variance hinge, stop-gradient next-token embedding MSE, and stop-head MSE;
3. `O3` (`ema_latent_vicreg`): token cross-entropy plus cosine-distance prediction against a stop-gradient exponential-moving-average teacher encoding of the next chunk, combined with variance and covariance anti-collapse regularization.

#### Scenario: Auxiliary predictor has zero inference overhead
- **WHEN** a model trained under `O3` (`ema_latent_vicreg`) is switched to evaluation or deployment mode
- **THEN** the auxiliary latent predictor and EMA teacher SHALL NOT execute during forward inference or generation
- **THEN** inference parameter count and latency SHALL match an identically sized `O1` model

#### Scenario: Pure cross-entropy disables stop-head and latent penalties
- **WHEN** objective `O1` (`ce_eos`) is selected
- **THEN** backward gradients SHALL depend solely on next-token cross-entropy (including `EOS`)
- **THEN** no variance hinge, embedding MSE, or stop-head MSE term SHALL contribute to parameter updates

### Requirement: Memory-bounded micro-batched chunked BPTT and online RTRL modes
The system SHALL support training diagonal-state candidates (`C0`, `C1`) under either per-step online eligibility-trace RTRL updates or batched chunked backpropagation through time, and SHALL support micro-batched gradient accumulation for all candidates so large effective token batches (such as `32 × 512`) execute identically within `< 3.8 GiB` CPU RAM without out-of-memory failures.

#### Scenario: Gradient accumulation matches full-batch gradients
- **WHEN** a candidate model processes an effective batch of $B$ sequences either in one pass or across $M$ micro-batches of size $B/M$ before the optimizer step
- **THEN** accumulated parameter gradients before clipping SHALL match within `1e-5` relative tolerance in fp32
- **THEN** peak process resident memory during micro-batched training on Tier-Small and Tier-Target models SHALL stay below `2.5 GiB` on CPU

#### Scenario: Online RTRL vs chunked BPTT comparison mode
- **WHEN** gradient mode `online_rtrl` is selected on a compatible diagonal core (`C0` or `C1`)
- **THEN** eligibility traces SHALL update per token without unrolling a multi-step computation graph
- **WHEN** `online_rtrl` is requested on a matrix or attention core (`C3`–`C6`)
- **THEN** the trainer SHALL refuse with an explicit incompatibility error rather than approximating silently

### Requirement: Non-colliding interleaved multi-task batching
When jointly training language modeling, option-conditioned decision scoring, and 5-way action routing on a shared core, the trainer SHALL sample single-task batches according to a configurable task mixture (default `0.80` LM, `0.15` decision, `0.05` routing) rather than summing heterogeneous losses inside the same forward pass, while supporting summed-batch training as an explicit ablation flag.

#### Scenario: Task sampling honours configured proportions reproducibly
- **WHEN** multi-task training runs for 1,000 steps with a fixed seed and mixture `(0.80, 0.15, 0.05)`
- **THEN** the realized batch counts per task SHALL match the target proportions within binomial sampling tolerance and repeat identically on re-run with the same seed
- **THEN** per-task training losses SHALL be logged separately so interference is observable during training
