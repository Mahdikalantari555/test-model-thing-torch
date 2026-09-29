# Design

## Context

See proposal.md for motivation. The current Droid WebUI (app.py) has:
- 4 tabs: Chat, Teach, Logs, Weights
- DroidEngine with KnowledgeStore (SQLite + FTS5 + dense vectors)
- OpenAITeacher for LLM distillation and synthesis
- Inline auto-teach in chat (non-question >50 chars)
- Explicit teach via `learn:` prefix
- Bulk teach in Teach tab with LLM distillation option

## Goals / Non-Goals

**Goals:**
- Add inline feedback UI after LLM-synthesized responses with proposition checkboxes
- Add new "Memory Console" tab with fact browser, CRUD, and LLM teach workflow
- Reuse existing `droid.teach()`, `teacher.distill_propositions()`, `KnowledgeStore` APIs
- No schema changes to KnowledgeStore (existing columns support all operations)

**Non-Goals:**
- Changes to RTU memory internals or ONNX anchor
- New database migrations (existing columns: superseded, valid_until, superseded_by already exist)
- Real-time collaborative editing
- Vector similarity search UI (text search only)

## Decisions

### 1. Streamlit Layout for Inline Feedback
**Decision**: Use `st.expander()` below each LLM-synthesized message in the chat container.
**Rationale**: Streamlit has no native right sidebar; expanders are the standard pattern for progressive disclosure.
**Alternative**: New column layout with `st.columns([3, 1])` - rejected because chat history rendering is sequential and columns would require restructuring the entire chat container.

### 2. Proposition Extraction Algorithm
**Decision**: Local sentence splitting with regex + filtering (no extra LLM call).
**Rationale**: Fast, zero-cost, works offline. Filtering rules:
- Split on `(?<=[.!?])\s+` (preserve abbreviations via placeholder like existing `_split_into_propositions`)
- Drop sentences < 30 chars
- Drop sentences starting with conversational markers: `based on`, `here is`, `here are`, `according to`, `in summary`, `to summarize`, `this response`, `the answer`, `as an ai`
**Alternative**: Call `teacher.distill_propositions()` on each LLM response - rejected due to latency and cost (extra LLM round-trip per response).

### 3. Memory Console Tab Structure
**Decision**: Single tab with three sub-sections rendered via `st.tabs()` inside the main tab:
- **Browse & Manage**: Searchable fact table with inline edit/supersede/delete
- **LLM Teach**: Text area → Distill → Review → Teach
- **Stats**: Knowledge store metrics (active/superseded count, sources, date range)

**Rationale**: Keeps all memory operations in one place; sub-tabs avoid vertical scrolling overload.

### 4. Fact CRUD Implementation
**Decision**: Extend `KnowledgeStore` with `update_fact(fact_id, new_text)` and `delete_fact(fact_id)` methods.
**Rationale**: Current API has `insert_fact`, `supersede`, `get_active_facts`. Update/delete require direct SQL but can be added as thin wrappers.
- `update_fact`: UPDATE facts.text, UPDATE facts_fts, re-embed vector, UPDATE facts_vecs
- `delete_fact`: DELETE FROM facts, facts_vecs, facts_fts (rebuild)

### 5. Session State for Feedback UI
**Decision**: Store feedback state in `st.session_state.feedback_<message_index>` dict with keys: `propositions` (list), `selected` (set of indices), `expanded` (bool).
**Rationale**: Streamlit reruns on every interaction; message index is stable within a render cycle. Using message content hash as key would be more robust but adds complexity.

### 6. LLM Teach Workflow Reuse
**Decision**: Reuse `OpenAITeacher.distill_propositions()` exactly as in Teach tab.
**Rationale**: Same prompt, same behavior. User gets consistent distillation quality.

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| Streamlit reruns lose feedback expander state | Use `st.session_state` keyed by message index; re-render expanded state from session |
| Proposition extraction misses key facts | Add "Add custom proposition" text input in feedback expander as escape hatch |
| Fact edit breaks vector cache | `KnowledgeStore.update_fact()` invalidates `_vec_cache` and `_id_cache` (like insert) |
| Large knowledge store slows browse table | Paginate at 50 rows; add `LIMIT 50 OFFSET` to SQL query |
| Concurrent teach from chat + console | `DroidEngine.teach()` is thread-safe (uses `torch.no_grad`); Streamlit is single-threaded per session |

## Migration Plan

1. Add `update_fact()` and `delete_fact()` to `KnowledgeStore`
2. Add "Memory Console" tab to `app.py` with three sub-tabs
3. Add inline feedback expander rendering in chat tab for LLM messages
4. Add session state management for feedback UI
5. Test with existing droid profiles (no migration needed - schema compatible)

## Open Questions

- Should fact edit create a new fact (supersede old) vs in-place update? Current design: in-place update with re-embedding. If versioning needed, can change to supersede+insert.
- Should LLM Teach workflow support batch teaching from multiple text inputs? Current design: single text area, can be extended.