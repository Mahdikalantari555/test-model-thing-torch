# Proposal

## Why

`DroidEngine.consolidate_memory()` currently provides a manual consolidation path that ranks and prunes low-utility facts and performs a small memory regularization step. It does not autonomously retrieve a replay set, compute parameter importance with Memory Aware Synapses (MAS), or use a persisted decreasing learning-rate schedule. Lifelong consolidation should be able to recover associated patterns without an external replay list, while protecting important weights and controlling plastic updates.

The design is motivated by the 2025 Frontiers report on autonomous retrieval through inhibitory plasticity, the MAS method's output-sensitivity importance, and the RDBP stability/plasticity baseline. The cited results motivate measurable targets; they are not assumed to hold for this repository without evaluation.

## What Changes

- Add `SleepConsolidation.autonomous_retrieval()` with self-inhibition so replay can recover at least 95 of 100 correlated patterns without receiving an external pattern list.
- Add `MemoryAwareSynapses` to estimate parameter importance from the squared-L2 output objective with one backward pass per sampled batch, and prune 10% of the least-important eligible plastic-adapter weights.
- Add `DecreasingBackprop` with `lr_i = base_lr * decay_factor^i`, a persisted update/task index, and stability metrics.
- Add `DroidEngine.sleep()` returning a structured report; add an opt-in, cancellable auto-sleep worker and a `/sleep` command that invokes the API and returns the report.
- Serialize sleep configuration, MAS importance/masks, and schedule state so restart does not reset consolidation policy.

## Impact

- **New capabilities:** `memory/sleep-consolidation`, `memory/memory-aware-synapses`, and `memory/decreasing-backprop`.
- **Code areas:** `src/model/memory_lifecycle.py`, `src/model/droid.py`, adapter parameter/state handling, CLI/chat dispatch, profile persistence, and new lifecycle tests.
- **Compatibility:** Automatic sleep is disabled unless configured; existing manual consolidation remains available and can delegate to the new report-producing API.
- **Target outcome:** autonomous recovery of >=95% of a fixed set of 100 correlated patterns, auditable pruning of the 10% lowest-importance eligible weights, and a stable persisted learning-rate schedule.

## Risks

- Autonomous replay can reinforce spurious associations or repeat the same attractor. Use self-inhibition/refractory state, cap replay steps, and report coverage rather than silently claiming recovery.
- Background consolidation can race with `teach()` or retrieval, especially with SQLite and Streamlit reruns. Use a single per-Droid worker, cancellation, and a shared engine lock; never start a duplicate worker on rerun.
- MAS importance estimated from unlabeled activations can under-protect rare facts. Preserve importance metadata, make pruning reversible through masks/checkpoints, and verify post-prune retrieval quality.
- Exponential learning-rate decay can become too small or reset after restart. Validate `decay_factor`, persist the index, and expose the effective rate.
- The 95% figure depends on the pattern set and stopping budget. Version the correlated-pattern benchmark and publish its retrieval protocol.

## Research References

- Autonomous retrieval via inhibitory plasticity: https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2025.1655701/full
- Memory Aware Synapses: https://link.springer.com/chapter/10.1007/978-3-030-01219-9_9
- ReLUDown + Decreasing Backprop (RDBP): https://arxiv.org/html/2507.10637v1
