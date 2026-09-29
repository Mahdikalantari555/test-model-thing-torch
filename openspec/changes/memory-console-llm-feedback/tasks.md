# Tasks

## 1. KnowledgeStore Extensions (CRUD)

- [ ] 1.1 Add `update_fact(fact_id: int, new_text: str)` method to `KnowledgeStore` in `src/model/knowledge_store.py`
  - Update `facts.text` column
  - Update `facts_fts` virtual table (rebuild via `INSERT INTO facts_fts(facts_fts) VALUES ('rebuild')`)
  - Re-embed using ONNX anchor (pass anchor reference or embed externally and pass vector)
  - Update `facts_vecs.vec` blob
  - Invalidate `_vec_cache` and `_id_cache`
  - Verify: Run `pytest tests/test_knowledge_store.py -v` (new test for update_fact)

- [ ] 1.2 Add `delete_fact(fact_id: int)` method to `KnowledgeStore` in `src/model/knowledge_store.py`
  - DELETE FROM facts, facts_vecs, facts_fts (rebuild)
  - Invalidate `_vec_cache` and `_id_cache`
  - Verify: Run `pytest tests/test_knowledge_store.py -v` (new test for delete_fact)

- [ ] 1.3 Add `get_fact(fact_id: int)` method for single fact retrieval
  - Returns full fact dict with all metadata
  - Verify: Run `pytest tests/test_knowledge_store.py -v` (new test)

## 2. Proposition Extraction Utility

- [ ] 2.1 Create `src/model/propositions.py` module with `extract_propositions(text: str) -> List[str]`
  - Split on sentence boundaries `(?<=[.!?])\s+`
  - Protect abbreviations (e.g., "e.g.", "i.e.") using placeholder technique from `droid._split_into_propositions`
  - Filter: length >= 30 chars
  - Filter: exclude sentences starting with conversational markers (case-insensitive):
    `based on`, `here is`, `here are`, `according to`, `in summary`, `to summarize`, `this response`, `the answer`, `as an ai`, `i cannot`, `i don't`, `i am`
  - Verify: Unit test `tests/test_propositions.py` with various LLM response samples

## 3. Inline Feedback in Chat Tab

- [ ] 3.1 Add session state initialization for feedback UI in `app.py`
  - `st.session_state.feedback_state = {}` (dict keyed by message index)
  - Each entry: `{"propositions": [], "selected": set(), "expanded": False}`

- [ ] 3.2 Modify chat rendering loop in `app.py` (tab_chat) to detect LLM-synthesized responses
  - Check `msg.get("source_label", "")` for "LLM" substring
  - If LLM response: extract propositions using `extract_propositions(msg["content"])`
  - Store in `feedback_state[msg_index]`

- [ ] 3.3 Render expandable feedback expander after each LLM message
  - Use `st.expander("🧠 Memory Console: Select propositions to learn", expanded=state["expanded"])`
  - Inside: checkbox for each proposition with `st.checkbox(prop, key=f"fb_{msg_index}_{i}")`
  - "Accept Selected" button: concatenate selected → `active_droid.teach(selected_text, source="chat_feedback")`
  - Show result toast, update feedback state, collapse expander
  - Verify: Manual test in Streamlit - ask LLM question, see feedback, accept, verify fact in logs

## 4. Memory Console Tab

- [ ] 4.1 Add "🧠 Memory Console" tab to main tabs list in `app.py`
  - Position after "Teach" tab (index 2)

- [ ] 4.2 Implement Memory Console tab with three sub-tabs:
  - **Browse & Manage**: `st.tabs(["Browse & Manage", "LLM Teach", "Stats"])`

- [ ] 4.3 Browse & Manage sub-tab
  - Search input: `st.text_input("Search facts", key="mem_search")`
  - Paginated table: `st.dataframe` with columns: Text (truncated), Source, Timestamp, Novelty, Superseded
  - Row actions: Edit button, Supersede button, Delete button (with confirmation)
  - Edit: `st.text_area` pre-filled → on save call `knowledge.update_fact(id, new_text)` + re-embed
  - Supersede: call `knowledge.supersede(id, new_id=None)` (mark only, no replacement)
  - Delete: call `knowledge.delete_fact(id)` with `st.confirm` dialog
  - Verify: Manual test - browse facts, edit one, verify recall returns new text

- [ ] 4.4 LLM Teach sub-tab
  - Text area: `st.text_area("Paste text to distill and teach", height=200)`
  - "Distill with LLM" button: if `llm_enabled` → `teacher.distill_propositions(text)` else local `extract_propositions(text)`
  - Display results as checkboxes: `st.checkbox(prop, key=f"teach_{i}")`
  - "Teach Selected" button: `active_droid.teach(selected_text, source="memory_console_llm")`
  - Show absorption result
  - Verify: Manual test - paste paragraph, distill, teach, verify in browse tab

- [ ] 4.5 Stats sub-tab
  - Active fact count, superseded count, total
  - Sources breakdown (group by source column)
  - Date range (min/max timestamp)
  - Memory norm, step count from active_droid
  - Verify: Visual check matches knowledge store state

## 5. Integration & Polish

- [ ] 5.1 Ensure feedback expander state persists across Streamlit reruns
  - Store `expanded` state in session, restore on render
  - Clear feedback state when chat history clears (droid switch, reset)

- [ ] 5.2 Add "Memory Console" to sidebar droid stats (optional)
  - Show active fact count in sidebar (already there: "Facts in Memory")

- [ ] 5.3 Run full verification
  - `PYTHONPATH=. /home/asus/miniforge3/envs/ai/bin/pytest tests/ -v`
  - `PYTHONPATH=. /home/asus/miniforge3/envs/ai/bin/python scripts/verify_remote_sensing_e2e.py`
  - Launch Streamlit: `/home/asus/miniforge3/envs/ai/bin/streamlit run app.py` and manually test all flows

## 6. Documentation

- [ ] 6.1 Update README.md with Memory Console usage
  - Inline feedback: "After LLM responses, expand Memory Console to select facts"
  - Memory Console tab: "Browse, edit, delete facts; LLM Teach workflow"

- [ ] 6.2 Add docstrings to new KnowledgeStore methods and propositions module