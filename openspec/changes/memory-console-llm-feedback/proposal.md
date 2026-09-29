# Proposal

## Why

The Droid system already supports explicit teaching (`learn:` prefix), auto-teaching from chat, and bulk teaching via the WebUI. However, when the LLM Provider synthesizes answers grounded in plastic memory (or general knowledge), there is no mechanism for the human to **verify, curate, and selectively promote** those LLM-generated responses into the Droid's persistent plastic memory. This creates a gap: the LLM acts as a "teacher" but the human has no feedback loop to accept/reject/modify what gets learned.

## What Changes

- **Inline LLM Response Feedback**: After each LLM-synthesized assistant message, an expandable "Memory Console" section appears with auto-extracted proposition checkboxes. User selects sentences → clicks "Accept Selected" → chosen facts are taught to the Droid via `droid.teach()`.

- **Memory Console Tab** (new tab): Dedicated full-width tab with:
  - **Fact Browser**: Search, view, edit, delete/supersede any fact in the knowledge store
  - **LLM Teach Workflow**: Paste raw text → LLM distills into propositions (via `teacher.distill_propositions`) → review checkboxes → teach selected to Droid
  - **Full CRUD** on knowledge store facts (edit text, mark superseded, delete, view metadata)

- **Proposition Extraction**: Auto-split LLM responses by sentences with filtering (length >30 chars, skip conversational markers like "Based on...", "Here is...", "According to...")

## Capabilities

### New Capabilities
- `droid/memory-console`: Human-in-the-loop feedback UI for LLM responses and full knowledge store management

### Modified Capabilities
- `droid/conversational-teaching`: Extended with inline feedback on LLM-synthesized responses (new scenario for human verification before absorption)

## Impact

- **app.py**: New tab "Memory Console", inline feedback rendering in chat tab, new session state for feedback UI
- **src/model/droid.py**: No core logic changes needed (existing `teach()` handles ingestion)
- **src/model/knowledge_store.py**: No schema changes needed (existing `supersede()`, `update_fact()` if added, or delete via SQL)
- **src/model/teacher.py**: Reuse existing `distill_propositions()` for LLM Teach workflow