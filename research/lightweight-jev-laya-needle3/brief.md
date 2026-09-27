# Deep Research Brief: Ultra-Lightweight System-1 Option-Selection Architecture (JEV + Laya + Needle 3 + Plastic RTU)

## Date
2026-09-27

## Objective
Investigate how to build an ultra-lightweight (< 30 MB), highly intelligent decision-making foundation model that excels at choosing the right options, executing typed decisions, and retaining lifelong memory without catastrophic collapse. Specifically investigate Jev, Laya, and Cactus Needle 3, and synthesize a concrete blueprint for combining all three with our plastic RTU memory.

## Scope
- In-Scope:
  - Jev (`jev` PyPI package, TypeSafe AI System-1 engine).
  - Laya (`laya` PyPI package, multilingual non-autoregressive decision engine).
  - Needle 3 (`needle` / `cactus-compute/needle`, 8-29 MB edge foundation model).
  - Mathematical & architectural unification with RTU (Recurrent Trace Unit) plastic memory.
  - Concrete Python implementation path under 30 MB RAM.
- Out-of-Scope:
  - 7B+ autoregressive generative chatbots.
  - Cloud-only APIs requiring constant internet connection.

## Angles
1. **Architectural Deconstruction**: Internal design, parameter sizes, and inference mechanics of JEV, Laya, and Needle 3.
2. **Decision Intelligence Paradigm**: Non-autoregressive System-1 inference vs. autoregressive token generation for option selection and tool calling.
3. **Synergy & Hybrid Blueprint**: Combining Needle 3 (quantized backbone), Laya (calibrated non-autoregressive heads), Jev (typed function compile interface), and RTU (plastic episodic memory).
4. **Implementation & Benchmark Roadmap**: Hardware footprint, latency, and step-by-step assembly in PyTorch/Python.
