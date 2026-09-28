# Design

## Context

The current `DroidEngine` stores facts in a Python list (`knowledge_bank`) and performs linear-scan cosine similarity over a growing PyTorch tensor (`knowledge_vectors`). At 50k facts, query-time re-normalization allocates 73 MB of dynamic heap, and retrieval latency scales $O(N)$. See `proposal.md` for the motivation.

## Goals / Non-Goals

**Goals:**
- Replace linear JSON scan with a persistent SQLite-backed hybrid store.
- Achieve sub-3 ms retrieval at 50k facts with 0 KB query-time allocation.
- Detect and supersede contradictory facts instead of letting them coexist.
- Resolve pronouns to antecedents during markdown ingestion.

**Non-Goals:**
- No new external C++ dependencies; only stdlib `sqlite3` and existing `torch`.
- No change to the ONNX anchor or RTU recurrence math.
- No change to the Streamlit UI layout or chat container behavior.

## Decisions

### Decision 1: SQLite WAL + FTS5 + Pre-Normalized MVM
- **Why**: SQLite is zero-dependency, embedded, and supports WAL mode for concurrent reads. FTS5 provides BM25 lexical ranking. Pre-normalized vectors allow pure matrix-vector multiplication (`torch.mv`) with zero per-query allocation.
- **Alternatives**: HNSW/FAISS (200+ MB RAM, C++ build bloat); ChromaDB (heavy Python deps); in-memory PyTorch flat scan (current baseline, O(N) allocation).
- **Implementation**: `src/model/knowledge_store.py` wraps a `sqlite3.Connection` with:
  - `facts` table: `id INTEGER PRIMARY KEY, text TEXT, source TEXT, timestamp REAL, step INTEGER, novelty REAL, superseded INTEGER DEFAULT 0, valid_until REAL, superseded_by INTEGER`.
  - `facts_vecs` table: `id INTEGER PRIMARY KEY, vec BLOB` storing float32 normalized 384-dim vectors.
  - `facts_fts` virtual table: external content FTS5 index on `text`.
  - Reciprocal Rank Fusion via single SQL CTE with `FULL OUTER JOIN`.

### Decision 2: Pre-Normalization on Insert
- **Why**: Cosine similarity between unit-norm vectors reduces to dot product (`torch.mv`). Re-normalizing on every query triggers dynamic heap allocation; normalizing once at insert eliminates it.
- **Implementation**: In `teach()`, after embedding propositions, normalize each vector with `torch.nn.functional.normalize(embs, p=2, dim=-1)` before storing.

### Decision 3: Two-Tier Contradiction Gating
- **Why**: Direct contradictions share entity tokens and yield high cosine similarity (~0.84–0.89 in MiniLM). Pure vector distance cannot detect factual conflict. A two-tier filter combining high semantic similarity ($\ge 0.65$) with deterministic negation, functional-predicate slot clash, and antonym pairs resolves conflicts in < 1 ms on CPU.
- **Implementation**: In `teach()`, for each new proposition:
  1. Embed and compute cosine similarity against existing non-superseded facts.
  2. If $\ge 0.65$, run deterministic SVO slot clash check (same subject, same functional predicate, different object) and polarity inversion check.
  3. If clash detected, mark old fact `superseded = 1`, `valid_until = now`, `superseded_by = new_id`.

### Decision 4: Rolling Discourse Anaphora Resolution
- **Why**: Naive sentence splitting chops pronouns from antecedents, producing incomplete propositions. Zero-dependency propagation restores entity anchors without spaCy.
- **Implementation**: In `_split_into_propositions()`, maintain a `last_subject` variable. When a sentence begins with a third-person pronoun (`it`, `they`, `this`, `these`, `those`), replace it with `last_subject`. Update `last_subject` from the grammatical subject of each sentence.

## Risks / Trade-offs

- **[Risk]** SQLite WAL file grows unbounded on continuous writes. → **Mitigation**: Run `PRAGMA wal_checkpoint(TRUNCATE)` during profile save.
- **[Risk]** FTS5 index build latency on large imports. → **Mitigation**: Batch inserts in a single transaction; FTS5 external content triggers update incrementally.
- **[Risk]** Pronoun resolution may incorrectly replace "it" when the true antecedent is ambiguous. → **Mitigation**: Only replace when a single clear subject exists in the immediately preceding sentence; otherwise leave unchanged.

## Migration Plan

1. Create `src/model/knowledge_store.py` with `KnowledgeStore` class.
2. Update `DroidEngine.__init__` to instantiate `KnowledgeStore` instead of `knowledge_bank` list.
3. Update `teach()` to insert into `KnowledgeStore` and run anaphora resolution + contradiction checks.
4. Update `recall()` to query `KnowledgeStore` with RRF fusion.
5. Update `save_profile()` / `load_profile()` to persist/restore the SQLite DB.
6. Run existing tests; add `tests/test_knowledge_store.py`.

## Open Questions

- Should superseded facts be hard-deleted after N steps or retained indefinitely? → Default: retain indefinitely for auditability; configurable via profile setting.