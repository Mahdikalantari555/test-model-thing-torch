# Design

## Architecture

Implement the sleep cycle as an orchestrator around existing Droid memory and lifecycle APIs rather than as an independent model. `DroidEngine.sleep()` acquires a per-engine reentrant memory lock, runs bounded autonomous retrieval, replays selected internal patterns, applies MAS importance/pruning to eligible plastic-adapter parameters, and returns one `SleepReport`. It may call existing `consolidate_memory()` for fact-utility pruning, but reports fact pruning separately from parameter pruning.

`SleepConsolidation.autonomous_retrieval()` must not accept a list of expected patterns as input. It queries internal memory state and uses an inhibitory/refractory trace to down-weight patterns just retrieved, allowing other correlated attractors to surface. The fixed 100-pattern test set is used only by the evaluator to score coverage. A cycle has strict iteration/time limits and is safe to cancel.

`MemoryAwareSynapses` computes output sensitivity on representative, unlabeled inputs. For each batch it evaluates the squared-L2 output objective and makes one backward pass; importance is the running mean of absolute parameter gradients. Pruning selects the lowest-importance 10% among explicitly eligible plastic-adapter parameters, stores a mask/checkpoint, and excludes the frozen semantic anchor and non-parameter memory buffers. An evaluation gate reports retrieval quality before and after pruning.

`DecreasingBackprop` owns the zero-based update/task index and computes `lr_i = base_lr * decay_factor^i`, with `0 < decay_factor <= 1`. The index and effective rate are persisted. The cited RDBP work also describes ReLUDown; this proposal scopes the named class to the requested schedule and keeps any ReLUDown activation transform as a separately tested, explicitly configured ablation rather than silently changing activations.

## Components

- **`SleepConsolidation`** — bounded replay loop, self-inhibition state, cycle cancellation, replay counters, and autonomous-coverage metrics.
- **`MemoryAwareSynapses`** — parameter allowlist, squared-L2 sensitivity estimate, normalized importance scores, deterministic low-score selection, mask/restore support, and before/after quality report.
- **`DecreasingBackprop`** — schedule validation, `lr_at(i)`, persisted index, and optimizer integration. `i=0` uses `base_lr`; after each committed task/update the index advances once.
- **`DroidEngine.sleep()`** — serializes interaction with teaching/recall, runs configured phases, returns status, duration, patterns replayed/recovered, facts pruned, parameters pruned, learning rate, and warnings.
- **Auto-sleep worker** — opt-in per profile; one daemon worker per engine with an event-based stop/join lifecycle, configurable idle/interval trigger, and no duplicate starts on UI reruns.
- **`/sleep` command** — dispatches before ordinary chat retrieval so the command performs one cycle and returns a concise summary rather than being stored as a user fact.

## Alternatives

- **External replay list:** rejected for the autonomous-retrieval acceptance test because it would supply the answer set and bypass self-retrieval.
- **Manual-only sleep:** keeps the current limitation; retained as an API-compatible mode but not the sole entry point.
- **EWC for importance:** retained as a benchmark comparator, but MAS is selected for the implementation because output sensitivity can be computed without task labels.
- **Unbounded always-running background thread:** rejected due CPU contention, lifecycle leaks, and duplicate workers in a rerun-based UI; use one opt-in cancellable worker.
- **Prune facts instead of weights:** existing consolidation already handles low-utility facts. MAS pruning is specifically limited to eligible plastic parameters; fact pruning remains separately measured.

## Risks / Trade-offs

- Self-inhibition gains coverage at the cost of potentially retrieving weak associations; report scored coverage, duplicate rate, and confidence thresholds.
- Parameter masks add checkpoint metadata and can reduce model capacity; pruning must be reversible and gated on a quality check.
- A shared lock can increase latency during sleep. Keep cycles bounded, make auto-sleep opt-in, and expose status/progress.
- The exact interpretation of RDBP's ReLUDown activation operation is not conflated with the explicit learning-rate formula; verify that operation from the cited method before adding it to the default schedule.
