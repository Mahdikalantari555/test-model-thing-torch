# Research Brief: Comprehensive Engineering & Product Improvement Roadmap

**Date**: 2026-09-27  
**Project**: `test-model-thing-torch` (TMT Droid / Remote Sensing Plastic Lifelong Learning Engine)  
**Roles**: Principal Software Engineer & Lead Product Manager  
**Depth Mode**: Deep (Multi-angle parallel investigation with cross-verification)

---

## 1. Context & Baseline Assessment

### Current Baseline Architecture
- **Semantic Anchor**: ONNX `all-MiniLM-L6-v2` (INT8, ~22MB, CPU inference <15ms).
- **Plastic Recurrent Layer**: PyTorch Recurrent Trace Unit (`RTULayer`, 384-dim hidden state, decay trace, embed trace).
- **Episodic Knowledge Store**: In-memory JSON list (`knowledge.json`) with naive linear-scan cosine similarity (`torch.cosine_similarity`).
- **Proposition Extraction**: Regex-based sentence boundary splitter with abbreviation protection and Markdown formatting stripping.
- **Teacher/Student Synthesis**: Stdlib `urllib` client supporting OpenAI-compatible endpoints (Ollama, LM Studio, Groq, OpenRouter, OpenAI) with local fallback.
- **UI & Multi-Tenancy**: Streamlit UI (`app.py`) with thread locking (`threading.Lock`) and profile directories under `droids/<droid-name>/`.
- **Test Suite**: 15 test suites, 45 passing unit tests.

### Key Pain Points & Bottlenecks
1. **Retrieval Scalability**: Linear O(N) memory scan; performance drops and memory footprint grows linearly as facts increase beyond hundreds.
2. **Underutilized Recurrent Plastic State**: Recurrent memory state `h_t` tracks sequence recurrence during training, but factual recall uses pure semantic cosine similarity to avoid centroid pollution. Needs a principled mathematical grounding (e.g. predictive coding, surprise gating, recency/frequency modulation).
3. **Knowledge Ingestion & Anaphora**: Regex splitting chops multi-sentence arguments and creates dangling pronouns ("It uses sensors..."). Lacks semantic chunking, relationship triples, or truth maintenance (contradiction resolution).
4. **Product Usability & Edge Integrations**: Locked inside a Streamlit web app; lacks headless CLI, MCP (Model Context Protocol) server interface, and REST API for IDE/agent integration.
5. **Geospatial & Multimodal Gap**: Repo claims "remote sensing plastic model", but current runtime is strictly text-based. Lacks lightweight raster/spectral embedding integration.

---

## 2. Research Angles

- **Angle 1: High-Performance Edge Retrieval & Hybrid Search (SWE Focus)**
  - SQLite-vec, FTS5 + BM25, HNSW, SIMD quantization, zero-external-dependency indexing for sub-10ms queries over 100k+ facts on CPU.
- **Angle 2: Advanced Plastic Memory, Recurrent States & Lifelong Learning (ML/SWE Focus)**
  - Fast-slow memory systems, Titans architecture (long-term memory with neural memory modules), RWKV-7, predictive coding, adaptive plasticity, and forgetting mechanisms.
- **Angle 3: Propositional Extraction, Knowledge Graphs & Conflict Resolution (PM/SWE Focus)**
  - Proposition-based RAG, truth maintenance systems (TMS), belief revision (handling contradictory user inputs), knowledge graph triple extraction without heavy LLMs.
- **Angle 4: Non-Autoregressive System-1 Decision & Tool-Calling Engines (PM/Architecture Focus)**
  - Function calling, classification, and structured actions without generative token collapse (Jev, Laya, Cactus Needle 3, Outlines, guidance on edge).
- **Angle 5: Product Surface, MCP Server Protocol & Developer Experience (PM Focus)**
  - MCP (Model Context Protocol) tool integration for Claude Code / Cursor / Windsurf, lightweight headless daemon (FastAPI/Uvicorn or stdlib HTTP), export/import portable brain packages (`.droid`).
- **Angle 6: Remote Sensing Multimodal Edge Foundations (Domain Focus)**
  - Integrating lightweight satellite vision models (Clay, Prithvi, SatMAE, or lightweight ViT/MobileNet spectral encoders) into the dual plastic memory pipeline.

---

## 3. Stopping Criteria & Deliverables
- Output `research/project-improvements/log.tsv` tracking all research iterations.
- Output `research/project-improvements/findings/F1.md` through `F6.md`.
- Output comprehensive `research/project-improvements/REPORT.md` with:
  1. Executive Summary & Product Vision (The "Edge Droid" Value Proposition).
  2. Technical Architecture & Engineering Upgrades (Detailed specs & code snippets).
  3. Product Roadmap (Phased implementation: v1.1, v1.2, v2.0).
  4. Concrete Next Steps & Immediate High-ROI Implementation Targets.
