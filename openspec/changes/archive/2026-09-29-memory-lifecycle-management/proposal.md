# Proposal

## Why
The current system stores all facts in a single episodic store with no lifecycle management: no versioning, no merging of memories across profiles, no distillation of experiences into stable semantic knowledge, no contradiction-aware retrieval policies, and no feedback-driven learning loop. As memory grows, retrieval quality degrades and there is no mechanism to compress, version, or transfer knowledge.

## What Changes
- Add memory versioning with full state snapshots and rollback capability
- Implement memory merge engine for combining multiple Droid profiles
- Add semantic memory store: distill episodic experiences into stable, de-duplicated propositions
- Implement contradiction detection during merge/distillation with resolution policies
- Add adaptive retrieval policy that shifts strategy based on feedback (approve/reject)
- Add feedback learning API: `correct()`, `approve()`, `reject()` on retrieved facts
- Add continuous learning stream: background absorption from interaction logs
- Add test-time training: plastic parameter updates during inference

## Capabilities

### New Capabilities
- `memory/versioning`: Snapshot versioning, branching, and rollback
- `memory/merge-engine`: Multi-profile memory fusion with conflict resolution
- `memory/semantic-store`: Distilled proposition store separate from episodic
- `memory/contradiction-resolution`: Detection and resolution of conflicting facts
- `memory/adaptive-retrieval`: Feedback-driven retrieval strategy selection
- `memory/feedback-learning`: User feedback API (correct/approve/reject)
- `memory/continuous-stream`: Background learning from interaction stream
- `memory/test-time-training`: Online plastic updates during inference

### Modified Capabilities
- `droid/profile-management`: Extends profile save/load with version metadata
- `droid/conversational-teaching`: Updates `teach()` to support continuous stream and feedback

## Impact
- `src/model/memory_lifecycle.py`: New module for versioning, merge, semantic store
- `src/model/droid.py`: New methods `correct()`, `approve()`, `reject()`, `merge_memory()`, `distill()`, `continuous_learn()`
- `src/model/knowledge_store.py`: Schema extensions for version, feedback, semantic flags
- `src/model/droid_manager.py`: `merge_droids()`, `export_version()`, `import_version()`
- No new external dependencies