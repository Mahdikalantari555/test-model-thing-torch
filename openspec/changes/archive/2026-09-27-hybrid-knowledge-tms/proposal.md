# Proposal

## Why
The current `DroidEngine` uses naive linear-scan cosine similarity over an in-memory JSON knowledge bank, which scales $O(N)$, allocates 73 MB of dynamic heap at 50k facts, and cannot resolve contradictions or dangling pronouns. Memory latency spikes and factual recall degrades below 3 ms query targets as the store grows.

## What Changes
- Replace in-memory JSON fact list + linear-scan retrieval with a **SQLite WAL-backed hybrid knowledge store** combining FTS5 lexical BM25 search and pre-normalized dense vector matrix-vector multiplication (MVM), fused via Reciprocal Rank Fusion (RRF).
- Introduce a **Truth Maintenance System (TMS)** using non-monotonic AGM belief revision to supersede contradictory propositions instead of letting conflicting facts coexist in the knowledge bank.
- Add **rolling discourse anaphora resolution** during markdown/text ingestion so pronouns are anchored to their antecedents without external NLP packages.
- Pre-normalize all knowledge vectors on insert so queries allocate 0 KB of dynamic heap and run sub-5 ms on CPU up to 100k facts.

## Capabilities

### New Capabilities
- `knowledge-store`: SQLite-backed persistent knowledge store with FTS5 full-text index, pre-normalized contiguous dense vectors, and RRF hybrid fusion replacing linear scan.
- `belief-revision`: Non-monotonic truth maintenance that detects functional-predicate slot clashes and polarity inversions; marks outdated facts as `superseded_by` while preserving audit trails.

### Modified Capabilities
- `droid/conversational-teaching`: Requirement to ingest markdown through a rolling anaphora resolver before proposition extraction; requirement for fact metadata (`superseded`, `valid_until`) in the stored belief record.

## Impact
- `src/model/droid.py`: Refactor `teach()`, `recall()`, and `chat()` methods to use new `KnowledgeStore`.
- New module `src/model/knowledge_store.py` wrapping SQLite + FTS5 + dense MVM.
- Update `save_profile()` / `load_profile()` serialization to persist the SQLite DB and knowledge vectors.
- No external C++ dependencies; only `sqlite3` stdlib + `torch` already present.
