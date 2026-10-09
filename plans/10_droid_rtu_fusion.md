# 10 Plan — Droid × Raw Byte RTU Fusion (`ByteRtuBackend`)

Status: **plan (not implemented)**.

Incremental slice of `09_recamem_plan.md` — Decision 1 / Phase 8 `EmbeddedRTUBackend`
plus Phase 6 channel-A recall — delivered against the **current** `src/` layout:
no rename, no packaging, no god-class refactor. After the RecaMem rename lands,
A1 becomes `recamem/embedded_rtu.py` and A3 becomes channel A of
`recamem/retrieval.py`. Nothing here blocks or reorders the 09 phases.

## Why this slice now

- The two engine modes today are either/or (`app.py:73`): **content without
  fluency** (droid fast path returns verbatim fact texts) or **fluency without
  content** (raw byte RTU has no domain knowledge). The fusion gives both, with
  no new training: the RTU supplies surface realization, the droid supplies
  grounded facts.
- Cheapest possible integration. `DroidEngine.generate_with_memory()` already
  speaks exactly one backend protocol — `generate(prompt, max_tokens,
  temperature, top_p)` (`src/model/droid.py:1155`). A byte-RTU backend is a
  drop-in peer of `GgufBackend` with **zero `DroidEngine` surgery**.
- Zero destructive learning *by construction*, which is what the
  "zero catastrophic collapse" claim needs but currently lacks evidence for
  (09 audit: forgetting is never measured; the README overstates).

## Decisions (recorded)

| # | Decision |
|---|---|
| 1 | **RTU weights stay frozen at all times.** Learning happens only in plastic memory (`update_associative_memory`) and the KnowledgeStore. If anyone later wants the RTU to *learn* (pattern B), it runs on a **copy** guarded by snapshot + `rollback_to()` (`droid.py:879`), never on the inference decoder. |
| 2 | Fusion model is `Model(dim=384, layers=8, spread=32)`. `dim` must match MiniLM / `h_trace` (384) so the A3 associative bonus and any state↔trace bridging are dimensionally well-defined. The standalone "Raw Byte RTU" mode keeps its own dim/layers widgets (`app.py:209-215`) untouched. |
| 3 | Fusion injection strategy is **prompt stuffing only**. `select_strategy()` may return `hidden` or `kv_cache`, but those implementations are stubs (`memory_injection.py:35-84` fixed random projection; `:86-123` mock tokens). Pin `injection_strategy: "prompt"` in the profile config so the fast path is deterministic. |
| 4 | **Query goes last in the primed prompt.** The RTU state is an EMA with per-dim half-lives spanning 1…`spread` (`rtu.py:117-119`), so the *tail* of the byte stream dominates state. `inject_prompt()`'s `Context: … \n\nQuery: x` layout is already correct and must not be reordered. |
| 5 | If `recall()` returns no hits, the fused path **must not** let the RTU free-generate. Return the existing fallback string from `droid.py:676`. Grounded silence beats fluent hallucination. |
| 6 | The A3 associative bonus may only **re-rank** candidates that already pass the dense gate (`droid.py:564`). It must never resurrect an out-of-domain query. |
| 7 | Scope of this plan: A1 backend, A2 persistence, A3 recall bonus, A4 WebUI mode. Out of scope: byte-level `teach_bytes()` into plastic memory (pattern D), training the RTU on exported droid facts (pattern B), ANN retrieval index, recall-boost rewrite (09 Phase 6). |

## Verified seams (checked against source)

| Seam | Location | Note |
|---|---|---|
| slow path calls `backend.generate(query, max_tokens=…, temperature=…, top_p=…)` | `droid.py:1153-1157` | same call shape as `GgufBackend` |
| prompt stuffing | `memory_injection.py:7-33` | facts → text; `select_strategy` at `:125-139` |
| associative `retrieve(q)` exists, never called by the engine | `rtu.py:611-641` | dead code today; flagged in 09 audit |
| `Model.load()` is a no-op when the file is missing | `rtu.py:341` | backend works with random init before any checkpoint exists |
| frozen priming pattern already used in production UI | `app.py:366-377` | `raw_model(c, nextb=n, end=…, frozen=True)` then sample loop |
| profile dir serialization | `droid.py:1217-1307` (save), `:1309-1417` (load) | one extra file (`rtu.safetensors`) is enough |

## Honesty notes (carried over from the 09 audit, relevant here)

- `effective_lr` is computed at `droid.py:427` and reported at `:497` but **never
  applied**; every plastic write is `torch.no_grad()` (`rtu.py:571`, `:604`).
  Nothing in this plan depends on a learning rate and the plan makes no LR
  claim for the fused path.
- `Model.sample()` is stochastic (`rtu.py:216-221`), so generation is
  non-deterministic. Determinism tests must compare **state tensors**, not text.
- The suite is 15 files / 57 test functions — not the "43 tests" the README
  claims. Run gate: `PYTHONPATH=. /home/asus/miniforge3/envs/ai/bin/pytest tests`.

## Target architecture

```
                     +-----------------------------------+
   query ----------> |  DroidEngine.recall()             |---> facts (top_k)
                     |   hybrid: BM25 + dense + RRF      |     (verbatim, audited)
                     |   + A3 associative bonus          |
                     +---------------+-------------------+
                                     |
                     inject_prompt(query, facts)      ; strategy pinned to "prompt"
                                     |
                                     v
                     "Context:\n- <fact>\n…\n\nQuery: …"   (query LAST — decision 4)
                                     |
                                     v  utf-8 encode, cap max_prompt_bytes
   ByteRtuBackend.generate()  <------+
     model.reset()                          ; zero states/decaytrace/embedtrace
     for c in prompt_bytes:  model(c, frozen=True)      ; priming, state only
     b = prompt_bytes[-1]
     loop: b, stop = model(b, frozen=True)              ; no optimizer step
           out += b; break if stop > 0.35 or max_tokens
                                     |
                     decode utf-8, errors="ignore"
                                     v
                                answer text

   fast path (recall-only) and standalone raw mode are unchanged.
```

## A1 — `src/model/byte_rtu_backend.py` (new)

```python
from pathlib import Path

from src.model.rtu import Model


class ByteRtuBackend:
    """Generative backend: conditions a FROZEN raw-byte RTU on retrieved facts.

    Mirrors the GgufBackend.generate() signature so it can be assigned to
    DroidEngine.gguf_backend (droid.py:366) without touching DroidEngine.
    """

    def __init__(self, checkpoint: str = "", dim: int = 384, layers: int = 8,
                 spread: int = 32, max_prompt_bytes: int = 1024,
                 max_output_bytes: int = 256, stop_threshold: float = 0.35):
        self.model = Model(dim=dim, layers=layers, spread=spread)
        self.checkpoint = checkpoint
        self.max_prompt_bytes = max_prompt_bytes
        self.max_output_bytes = max_output_bytes
        self.stop_threshold = stop_threshold
        if checkpoint and Path(checkpoint).exists():
            self.model.load(checkpoint)          # no-op if missing (rtu.py:341)

    def tokenize(self, text: str) -> list:
        """Byte-level tokenizer; kept so inject_kv_cache (memory_injection.py:101)
        degrades to prompt stuffing instead of raising."""
        return list(text.encode("utf-8"))

    def reset(self) -> None:
        self.model.reset()                       # rtu.py:304

    def generate(self, prompt: str, max_tokens: int = 256,
                 temperature: float = 0.7, top_p: float = 0.9) -> str:
        # temperature / top_p are accepted for protocol parity and IGNORED:
        # the RTU owns its sampling temperature (rtu.py:219).
        self.model.reset()
        data = prompt.encode("utf-8")[-self.max_prompt_bytes:]
        if not data:
            return ""
        for c in data[:-1]:                       # priming: state only, frozen
            self.model(c, frozen=True)
        b, out = data[-1], bytearray()
        for _ in range(min(max_tokens, self.max_output_bytes)):
            b, stop = self.model(b, frozen=True)  # frozen: no autograd, no step
            out.append(b)
            if stop > self.stop_threshold:
                break
        return bytes(out).decode("utf-8", errors="ignore")
```

Constraints (must hold in review):

- `frozen=True` routes through `rtu.py:248-252`, which detaches and mirrors
  `state` into `layer.states`; **no autograd, no optimizer step**.
- The non-frozen path (`rtu.py:254-298`) runs `self.optimizer.step()` at
  `rtu.py:295` and mutates weights. It must never be reachable from inference.
- `end` is consumed only by training losses (`compute_losses`, `rtu.py:168`) —
  ignore it while priming.
- Priming is a Python loop over bytes × layers. Cap prompt bytes (1024) and
  output bytes (256); the E3 benchmark gate below decides if these caps move.
- `tokenize()` must stay trivial bytes; it is never fed to a real LLM tokenizer.

## A2 — persistence inside the droid profile

Files in `droids/<name>/` after this change:

| File | Owner | Change |
|---|---|---|
| `config.json` | droid | `+ "rtu_checkpoint": "rtu.safetensors"`, `+ "injection_strategy": "prompt"`, `+ "rtu_dim": 384`, `+ "rtu_layers": 8` |
| `rtu.safetensors` | RTU | new — `Model.save()` (`rtu.py:312`): `m.*` params, `state.i`, `decaytrace.i`, `embedtrace.i`, `o.i.*` optimizer state |
| `memory.safetensors` | droid | unchanged |
| `knowledge.db`, `train_log.json`, `snapshots.json`, `retrieval_policy.json` | droid | unchanged |

Rules:

- Save order: knowledge → RTU → `config.json` last (the config is the manifest
  that records what exists).
- `save_profile()` calls `backend.model.save(<dir>/rtu.safetensors)` only when
  `isinstance(self.gguf_backend, ByteRtuBackend)`; the GGUF path is untouched.
- `load_profile()` re-attaches the backend after the existing memory load, and
  raises a clear error (not a silent skip) when `config["rtu_checkpoint"]` is
  set but the file is absent — because a silently random-init RTU is exactly the
  kind of silent degradation 09 Phase 2 exists to remove.
- `DroidManager.create_droid(..., byte_rtu=True)` convenience arg that wires
  `byte_rtu_backend=` on engine construction.

## A3 — associative-recall bonus in `recall()`

In `DroidEngine.recall()` (`droid.py:504-611`), after `q_emb` is normalized
(`:521`), add:

```python
q_assoc = None
try:
    retrieve = getattr(self.memory, "retrieve", None)
    if retrieve is not None:
        q_assoc = retrieve(q_emb)                 # S @ W_Q(q) + 0.1*h_trace
        q_assoc = q_assoc / q_assoc.norm().clamp(min=1e-8)
except Exception:
    q_assoc = None                                # bonus is optional; never fatal
```

Per candidate, next to the existing `contextual` term (`droid.py:569-580`):

```python
if q_assoc is not None:
    assoc = torch.dot(q_assoc, fact_emb).item()
    m["similarity"] = m["similarity"] + 0.15 * max(0.0, assoc)
```

Guardrails:

- Weight `0.15` is a starting point; tune against
  `scripts/verify_remote_sensing_e2e.py` staying green.
- The `sim < 0.48` dense gate at `droid.py:564` stays authoritative — the bonus
  re-ranks true positives only (decision 6).
- With an empty `S` (fresh droid) `retrieve()` returns ≈`0.1 * h_trace`, so the
  bonus is near zero. A cold-start no-regression test is mandatory.
- This is intentionally the *minimal* version of 09 Phase 6 channel A; the full
  plastic-prior re-ranker (fitted features, write-back on recall) stays in 09.

## A4 — WebUI mode

- Add a third option beside the two at `app.py:73`:
  `"Droid + Raw Byte RTU (Fused)"`.
- Fused mode reuses the droid chat handler unchanged; only the slow path
  differs (backend assigned). Fast path answers stay byte-identical to the
  plain droid mode.
- Sidebar/memory-console badge: RTU `dim`/`layers`, checkpoint path,
  anchor real-vs-mock, and whether the last answer was generative (slow) or
  extractive (fast).
- The standalone "Raw Byte RTU" mode keeps its own dim/layers widgets
  (`app.py:209-215`) and its byte loop (`:364-382`) — no behavior change.

## Tasks

### Phase A — backend & plumbing (no behavior change)

- [ ] A1.1 Create `src/model/byte_rtu_backend.py` per §A1.
- [ ] A1.2 Test: empty prompt → `""`; non-ASCII prompt → valid `str`.
- [ ] A1.3 Test: priming with prompt bytes changes the output vs no priming
        (state carries context — this is the core claim of the whole fusion).
- [ ] A1.4 Add `byte_rtu_backend=` param to `DroidEngine.__init__`
        (`droid.py:295-302`); when both it and `gguf_backend` are supplied,
        raise `ValueError` instead of silently picking one.
- [ ] A1.5 `DroidManager.create_droid(..., byte_rtu=True)` convenience arg
        (`droid_manager.py:45`).

### Phase B — persistence

- [ ] B1 Extend `save_profile` / `load_profile` (`droid.py:1217` / `:1309`) per §A2.
- [ ] B2 Test: save → load → `state.0` buffers and all `m.*` parameters equal;
        identical generation behavior must be asserted via state tensors, not
        sampled text (see honesty notes).

### Phase C — associative recall bonus

- [ ] C1 Implement §A3 in `recall()`.
- [ ] C2 Test: two candidates with equal dense similarity must re-order when the
        bonus is present, plus a cold-start (empty `S`) no-regression test.

### Phase D — WebUI + docs

- [ ] D1 Engine-mode option + badge (`app.py:73`, memory console).
- [ ] D2 `README.md`: add the fused branch to the architecture diagram.
- [ ] D3 `AGENTS.md`: record "RTU weights are frozen at inference; all learning
        lives in plastic memory" as a rule, plus the new file ownership.

### Phase E — verification gates

- [ ] E1 `PYTHONPATH=. /home/asus/miniforge3/envs/ai/bin/pytest tests` — full
        suite green, zero changes to existing contracts.
- [ ] E2 `PYTHONPATH=. /home/asus/miniforge3/envs/ai/bin/python scripts/verify_remote_sensing_e2e.py`
        — green with the fused backend attached.
- [ ] E3 Benchmark one fused `chat("/slow <query>")`; if it exceeds 10 s, lower
        `max_prompt_bytes` and record the measured ceiling in this file.

## New tests — `tests/test_droid_rtu_fusion.py`

1. `test_byte_rtu_backend_matches_raw_ui_loop` — backend `generate()` output
   equals the inline loop in `app.py:366-377` for the same primed state.
2. `test_fused_chat_is_grounded` — teach a domain paragraph, then
   `chat("/slow …")` output contains at least one taught fact substring.
3. `test_no_free_generation_without_facts` — fresh/empty droid: fused slow path
   returns the `droid.py:676` fallback, never generated text (decision 5).
4. `test_rtu_weights_frozen_during_inference` — checksum of every `m.*`
   parameter and of `state.*`/`decaytrace.*`/`embedtrace.*` before and after a
   fused chat are identical (proof of decision 1).
5. `test_fused_profile_roundtrip` — `save_profile`/`load_profile` keeps the RTU
   checkpoint equal, and raises loudly if `rtu.safetensors` is missing while
   config declares it.
6. `test_associative_bonus_reranks_and_cold_start_is_neutral` — C2 above.

## Risks

| Risk | Mitigation |
|---|---|
| A byte RTU trained on a small corpus emits garbled text, dragging perceived quality | Fast path stays extractive; slow path is opt-in per query (`/slow`); document the expected quality ceiling in README. |
| Someone later "trains" the RTU and reintroduces catastrophic forgetting | Frozen by construction; any training variant requires snapshot + `rollback_to()` and its own plan (decision 1). Test 4 is the guard. |
| Python-loop latency (bytes × layers, frozen calls) | Caps on prompt/output bytes; E3 benchmark gate; record the ceiling. |
| Associative bonus injects false positives | Dense gate stays authoritative; weight ≤ 0.15; cold-start test (decision 6). |
| UTF-8 truncation mid-character | `errors="ignore"`; test A1.2. |
| `select_strategy` drifts to `hidden`/`kv_cache` stubs | Config pin `injection_strategy: "prompt"` + trivial `tokenize()` shim (decision 3). |
| Random-init RTU silently used because no checkpoint exists | `Model.load` no-ops on missing file, so `load_profile` must explicitly error when config declares a checkpoint that is absent (A2). |
| Scope creep into the 09 rename | This plan touches only `byte_rtu_backend.py` (new), small hooks in `droid.py`, `droid_manager.py`, `app.py`, and `plans/` docs. |

## Acceptance criteria

- Existing suite green with zero contract modifications.
- Fused mode: teach a remote-sensing paragraph → `/slow` answer contains taught
  content; `/fast` answer identical to non-fused droid; standalone raw mode
  byte-identical to pre-change behavior.
- RTU parameter and state checksums unchanged across inference (test 4).
- `scripts/verify_remote_sensing_e2e.py` passes with the fused backend.
- Any measured latency/quality number is written into this file before the plan
  is marked implemented.

## Open questions

1. `max_prompt_bytes` 1024 vs 512 — depends on the E3 measurement; a 1024-byte
   prompt leaves only the last ~32 steps of half-life reaching the state
   (decision 4), so a *shorter*, higher-signal context may rank better.
2. Should the fused slow path expose `stop_threshold` per profile config, or
   keep the hardcoded 0.35 that the standalone UI uses (`app.py:375`)?
3. Does `recamem`/09 rename `ByteRtuBackend` → `EmbeddedRTUBackend` directly
   (recommended) or keep an alias? Not decidable until Phase 1 of 09 starts.
