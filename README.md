# Test-Model-Thing (TMT) — PyTorch & Droid Lifelong Engine

[Original YouTube Demo](https://youtu.be/9UERVVwpNew) | [Original MLX Repo](https://github.com/jrz97619761/test-model-thing)

This repository provides a 1:1, contract-verified **PyTorch** implementation of Test-Model-Thing (TMT) alongside the **Droid Lifelong Engine** — an ultra-lightweight (< 30 MB) continual learning system that combines a frozen ONNX semantic anchor with plastic Recurrent Trace Units (RTUs) for conversational domain teaching without catastrophic forgetting.

---

## Key Features

1. **Pure PyTorch Implementation**:
   - 100% parity with original MLX math (variance loss `unbiased=False`, custom AdamW optimizer, dual gradient hooks, decay trace updates).
   - Runs everywhere: Linux, macOS, WSL2, and Windows (CPU or CUDA).
   - Preserves original MLX scripts as `main_mlx.py` and `benchmark_mlx.py`.

2. **Droid Lifelong Learning Engine**:
   - **Semantic Anchor**: Quantized INT8 `all-MiniLM-L6-v2` via ONNX Runtime (~22 MB on disk, sub-15ms CPU inference, cached in `~/.cache/huggingface`).
   - **Plastic RTU Memory**: Recurrent trace unit accumulating proposition vectors with automatic learning rate decay $\eta = \eta_0 / \sqrt{1 + \text{steps}/20}$.
   - **Zero Catastrophic Collapse**: Base language syntax is anchored; learning specialized domain paragraphs (e.g. Remote Sensing, Hydrology, Medicine) adapts episodic memory instantly without corrupting previous knowledge.
   - **In-Chat Conversational Teaching**: Teach the model simply by chatting, sending text paragraphs, or using `learn: <text>`.

3. **Interactive Streamlit WebUI**:
   - Switch between named Droid profiles (`droids/<name>/`) and raw byte RTU.
   - Live chat, instant domain paragraph ingestion, transparent audit logs, and RTU memory/decay visualization.

---

## Quickstart

### 1. Installation

```bash
git clone https://github.com/jrz97619761/test-model-thing-torch.git
cd test-model-thing-torch

# Install dependencies (Python 3.10+)
pip install -r requirements.txt
```

### 2. Launch WebUI (Streamlit)

```bash
streamlit run app.py
```
Open `http://localhost:8501` to chat, switch Droids, teach new domains, and inspect internal states.

### 3. CLI Usage

#### PyTorch CLI
```bash
# Train on raw bytes
python main.py <checkpoint.safetensors> train

# Interactive chat
python main.py <checkpoint.safetensors> chat

# Inference only (frozen weights)
python main.py <checkpoint.safetensors> chat --frozen

# Benchmark with CoLA
python benchmark.py <checkpoint.safetensors> <epochs> <slice>
```

#### Droid Lifelong Engine in Python
```python
from src.model.droid_manager import DroidManager

mgr = DroidManager(base_dir="droids")
droid = mgr.get_droid("droid-geospatial")

# Teach a domain paragraph (absorbed in < 30ms on CPU)
droid.teach(
    "Remote sensing is the acquisition of information about an object or phenomenon "
    "without making physical contact with the object, in contrast to on-site observation."
)

# Recall accurately in chat
reply = droid.chat("What is remote sensing?")
print(reply)
# -> "Remote sensing is the acquisition of information about an object..."

# Save profile
mgr.save_droid("droid-geospatial")
```

---

## Architecture Overview

```
                      +-----------------------------+
                      |   Input Text / Query        |
                      +--------------+--------------+
                                     |
                                     v
               +-------------------------------------------+
               |  ONNX MiniLM Anchor (~22 MB, INT8)        |
               |  (Frozen Base Semantic Feature Extractor) |
               +---------------------+---------------------+
                                     |
                          Embedding z in R^384
                                     |
                                     v
               +-------------------------------------------+
               |  Plastic RTU Memory Block (dim=384)       |
               |  h_t = decay * h_{t-1} + z_t              |
               |  x = LayerNorm + Linear + SiLU            |
               +---------------------+---------------------+
                                     |
                                     v
               +-------------------------------------------+
               |  Episodic Knowledge Bank & Dual Recall    |
               |  -> Grounded domain synthesis             |
               |  -> Zero catastrophic collapse            |
               +-------------------------------------------+
```

---

## Verification & Tests

Run the complete test suite (43 contract, unit, and lifelong learning tests):

```bash
pytest tests
```

Run end-to-end domain teaching verification (Remote Sensing acquisition & recall):

```bash
python scripts/verify_remote_sensing_e2e.py
```

---

## License

MIT
