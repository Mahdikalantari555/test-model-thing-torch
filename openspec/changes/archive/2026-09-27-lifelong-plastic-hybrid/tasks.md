# Tasks: Improved TMT Droid Engine

## 1. ONNX Semantic Anchor & RTU Coupling

- [x] 1.1 Implement `OnnxMiniLM` in `src/model/onnx_anchor.py` loading INT8 MiniLM from `~/.cache/huggingface`, verified by `tests/test_onnx_anchor.py`
- [x] 1.2 Implement `DroidEngine` in `src/model/droid.py` coupling ONNX semantic vectors to RTU plastic memory with auto-tuned parameters, verified by `tests/test_droid_engine.py`

## 2. In-Chat Conversational Teaching & Domain Specialization

- [x] 2.1 Implement in-chat knowledge absorption (`teach()` method) with zero-collapse guarantees, verified by `tests/test_conversational_teaching.py`
- [x] 2.2 Implement multi-droid profile manager (`droids/<name>/`), verified by `tests/test_droid_profiles.py`
- [x] 2.3 Implement optional OpenAI-compatible teacher synthesis helper in `src/model/teacher.py`

## 3. WebUI Control Station & End-to-End Verification

- [x] 3.1 Update `app.py` with Droid profile selector, in-chat teaching buttons, live transparent logs, and ONNX status
- [x] 3.2 End-to-end verification: teach Droid the Remote Sensing paragraph in chat and verify instant factual recall without model collapse
