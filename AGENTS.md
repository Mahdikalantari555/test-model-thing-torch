# Repository Instructions for AI Agents

## Environment & Toolchain
- **Python Environment**: ALWAYS use the `ai` conda environment for all executions:
  - Python: `/home/asus/miniforge3/envs/ai/bin/python`
  - Pytest: `/home/asus/miniforge3/envs/ai/bin/pytest`
  - Streamlit: `/home/asus/miniforge3/envs/ai/bin/streamlit`
  - Always prepend `PYTHONPATH=.` when invoking python directly from project root.

## Verification & Commands
- **Run all tests**:
  ```bash
  PYTHONPATH=. /home/asus/miniforge3/envs/ai/bin/pytest tests
  ```
- **Run a single test file**:
  ```bash
  PYTHONPATH=. /home/asus/miniforge3/envs/ai/bin/pytest tests/test_droid_engine.py
  ```
- **Run End-to-End Verification**:
  ```bash
  PYTHONPATH=. /home/asus/miniforge3/envs/ai/bin/python scripts/verify_remote_sensing_e2e.py
  ```
- **Launch WebUI**:
  ```bash
  /home/asus/miniforge3/envs/ai/bin/streamlit run app.py
  ```

## Architecture & Code Boundaries
- `src/model/rtu.py`: 1:1 PyTorch port of the MLX RTU block, Model class, custom AdamW, and loss functions (`unbiased=False` on variance loss).
- `src/model/onnx_anchor.py`: INT8 quantized `all-MiniLM-L6-v2` loaded via ONNX Runtime (< 25 MB). Models and tokenizer cached in `~/.cache/huggingface`.
- `src/model/droid.py`: `DroidEngine` coupling ONNX anchor to plastic RTU memory blocks with automatic learning rate and zero catastrophic collapse.
- `src/model/droid_manager.py`: Multi-profile manager for named droids in `droids/<name>/` (`config.json`, `memory.safetensors`, `knowledge.json`, `train_log.json`).
- `src/model/teacher.py`: Stdlib `urllib` client for OpenAI-compatible endpoints to distill unstructured text into propositions.
- `app.py`: Streamlit control station with dual modes (Droid Engine vs. Raw Byte RTU).

## Critical Implementation Quirks & Gotchas
- **PyTorch In-Place Parameter Edits**: When updating `nn.Parameter` values (e.g. `decay`) from disk or buffers, wrap in `with torch.no_grad():` or write to `.data` to prevent autograd leaf Variable runtime errors.
- **Safetensors Shared Memory**: Do NOT save duplicate references to `blocks.i.states` and `state.i`. Extract model weights via `named_parameters()` rather than `state_dict()`.
- **Hugging Face Cache**: Quantized ONNX weights and tokenizers reside in `~/.cache/huggingface`. Do not re-download if present.
- **OpenSpec**: Managed via `/home/asus/.local/share/pnpm/bin/openspec`. Specs reside in `openspec/specs/` and archived changes in `openspec/changes/archive/`.
