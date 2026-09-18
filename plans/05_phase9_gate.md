# 05 Phase 9 Gate (Continual Learning)

## Status

Deferred. Explicitly gated on Phase 8 passing. NOT started until then.

## Gate definition

Phase 8 "passes" when ALL of:
- [ ] TinyStories experiment runs to completion on CPU
- [ ] Train loss decreases monotonically over the run
- [ ] Validation loss tracks train loss (no divergence)
- [ ] tokens/sec and memory usage are recorded and sane
- [ ] Reproduction report written (reports/reproduction_report.md)

Only after the gate is met may Phase 9 begin.

## Phase 9 protocol (planned, not started)

Train on Dataset A -> Train on Dataset B -> Evaluate on A and B.
Metrics: forgetting, adaptation, memory retention.
Output: reports/continual_learning.md

## Why this gate exists

Continual learning measurement is only meaningful if the model has
actually learned. Measuring forgetting on a model that never converged
produces garbage that looks scientific. The gate prevents that.