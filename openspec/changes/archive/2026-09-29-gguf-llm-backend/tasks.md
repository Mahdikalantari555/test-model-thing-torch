# Tasks

## 1. GGUF Backend Core

- [ ] 1.1 Add `llama-cpp-python` to `requirements.txt` as optional extra `[gguf]`
- [ ] 1.2 Create `src/model/gguf_backend.py` with `GgufBackend` class wrapping `llama_cpp.Llama`
- [ ] 1.3 Implement `__init__(model_path=None, repo_id=None, filename=None, n_ctx=4096, n_gpu_layers=-1, **params)`
- [ ] 1.4 Implement `generate(prompt, max_tokens=512, temperature=0.7, top_p=0.9, stream=False, callback=None)`
- [ ] 1.5 Implement `tokenize(text)`, `detokenize(tokens)`, `get_kv_cache()`, `set_kv_cache(kv)`
- [ ] 1.6 Implement `download_model(repo_id, filename, local_dir)` using `huggingface_hub`
- [ ] 1.7 Verify: `tests/test_gguf_backend.py` loads model, generates, streams

## 2. Memory Injection Strategies

- [ ] 2.1 Create `src/model/memory_injection.py` with three strategy functions
- [ ] 2.2 Implement `inject_prompt(query, facts, max_context_tokens)` → formatted prompt string
- [ ] 2.3 Implement `inject_hidden(backend, query, h_trace, layer_idx)` → forward hook adding projected $h_t$
- [ ] 2.4 Implement `inject_kv_cache(backend, facts, query)` → encode facts, prime KV cache
- [ ] 2.5 Implement `select_strategy(fact_count, query_type, profile_config)` → strategy name
- [ ] 2.6 Verify: unit tests for each strategy with mock backend

## 3. Confidence Estimation

- [ ] 3.1 Create `src/model/confidence.py` with `estimate_confidence()` and components
- [ ] 3.2 Implement `retrieval_confidence(similarity, surprise, health_metrics)` → float [0,1]
- [ ] 3.3 Implement `generation_confidence(token_logprobs, self_consistency, entropy)` → float [0,1]
- [ ] 3.4 Implement `fuse_confidence(Cr, Cg, weights)` → harmonic mean
- [ ] 3.5 Implement `confidence_tier(confidence)` → "high"/"medium"/"low"
- [ ] 3.6 Verify: unit tests with known inputs produce expected outputs

## 4. DroidEngine Integration — Dual-Mode Generation

- [ ] 4.1 Update `DroidEngine.__init__` to accept optional `gguf_backend` and `gguf_config`
- [ ] 4.2 Add `DroidEngine.generate_with_memory(query, strategy="auto")` using injection + GGUF
- [ ] 4.4 Add `DroidEngine.estimate_confidence(retrieval_conf, generation_conf)` 
- [ ] 4.5 Add `DroidEngine.decide_path(query)` using System-1 `DecisionHead` (fast_recall vs slow_synthesis)
- [ ] 4.6 Update `chat()` to call `decide_path()`, route to `recall()` (fast) or `generate_with_memory()` (slow)
- [ ] 4.7 Add `/fast` and `/slow` chat command overrides
- [ ] 4.8 Verify: integration test shows fast path <50ms, slow path grounded in facts

## 5. Profile & Manager Integration

- [ ] 5.1 Update `DroidManager.create_droid()` to accept optional `gguf_model` config
- [ ] 5.2 Update `config.json` schema: `gguf_model_path`, `gguf_repo_id`, `gguf_filename`, `gguf_params`
- [ ] 5.3 Add `DroidManager.download_model(name, repo_id, filename)` downloading to profile dir
- [ ] 5.4 Update `save_profile()`/`load_profile()` to persist GGUF config
- [ ] 5.5 Verify: manager test creates droid with GGUF, downloads model, generates

## 6. CLI & MCP Integration

- [ ] 6.1 Add `download-model` subcommand to `src/cli.py`
- [ ] 6.2 Add `--mode fast|slow|auto` to `chat` subcommand
- [ ] 6.3 Add `generate` subcommand for direct LLM generation with memory
- [ ] 6.4 Add MCP tool `generate_with_memory` exposing slow path
- [ ] 6.5 Verify: CLI commands work; MCP tool returns structured response

## 7. Tests & Validation

- [ ] 7.1 Create `tests/test_gguf_backend.py` (requires model fixture or skip if no model)
- [ ] 7.2 Create `tests/test_memory_injection.py` with mock backend
- [ ] 7.3 Create `tests/test_confidence.py`
- [ ] 7.4 Extend `tests/test_droid_engine.py` for dual-mode, injection, confidence
- [ ] 7.5 Run full test suite: `PYTHONPATH=. /home/asus/miniforge3/envs/ai/bin/pytest tests -v`
- [ ] 7.6 Benchmark: fast path <50ms, slow path 500–2000ms, memory injection adds <100ms