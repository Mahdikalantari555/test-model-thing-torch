# Tasks

## 1. Autonomous retrieval and sleep API

- [ ] 1.1 Implement `SleepConsolidation` with `autonomous_retrieval()` driven only by the Droid's internal memory state and a decaying self-inhibition/refractory trace.
- [ ] 1.2 Add bounded replay-step, time, confidence, cancellation, and duplicate-retrieval controls.
- [ ] 1.3 Add `DroidEngine.sleep()` returning a structured report with status, duration, coverage, replay counts, pruning counts, and errors.
- [ ] 1.4 Integrate sleep with existing `consolidate_memory()` without conflating fact pruning and parameter pruning.
- [ ] 1.5 Add fixed correlated-pattern tests proving at least 95 of 100 stored patterns are recovered without passing an expected-pattern list to the replay method.

## 2. Memory Aware Synapses

- [ ] 2.1 Implement MAS importance using the squared-L2 output objective and one backward pass per sampled batch.
- [ ] 2.2 Restrict the eligible-parameter set to plastic-adapter weights; exclude the frozen anchor and buffers.
- [ ] 2.3 Prune the lowest-importance 10% deterministically and save importance scores plus reversible masks in adapter state.
- [ ] 2.4 Add tests for importance ranking, one-backward-pass behavior, exact prune count, checkpoint round trip, and no mutation outside the allowlist.
- [ ] 2.5 Measure retrieval/retention quality before and after pruning; report rather than hide regressions.

## 3. Decreasing Backprop schedule

- [ ] 3.1 Implement `DecreasingBackprop` with validated `base_lr`, `decay_factor`, and zero-based index using `lr_i = base_lr * decay_factor^i`.
- [ ] 3.2 Persist and restore the schedule index and effective learning rate in Droid profile/checkpoint metadata.
- [ ] 3.3 Add monotonicity, parameter-boundary, restart/resume, and optimizer-integration tests.
- [ ] 3.4 Keep any ReLUDown activation transform isolated behind an explicit experimental option until its exact reference behavior is verified.

## 4. Command and worker lifecycle

- [ ] 4.1 Add `/sleep` command dispatch and a CLI sleep action that invoke the same `DroidEngine.sleep()` API.
- [ ] 4.2 Add an opt-in single auto-sleep worker per Droid with event-based stop/join and configurable idle/interval settings.
- [ ] 4.3 Serialize sleep with teach/recall through a shared engine lock and test cancellation/profile shutdown.
- [ ] 4.4 Ensure UI reruns and repeated start calls cannot create duplicate workers.

## 5. Validation

- [ ] 5.1 Add lifecycle tests for an empty store, successful cycle, cancellation, concurrent teaching, and worker restart.
- [ ] 5.2 Run the 100-correlated-pattern recovery benchmark and verify >=95% coverage without an external list.
- [ ] 5.3 Run MAS pruning and DecreasingBackprop stability tests; document configurations and quality deltas.
- [ ] 5.4 Run the full project test suite and checkpoint/profile compatibility tests.
