# Plans — test-model-thing → PyTorch

This folder holds the **specification** before any code is written.
Guiding decisions, fixed up front:

1. **Numerical similarity matters more than architectural novelty.**
   A translation that produces the same loss curve beats one that is
   prettier. Bit-exact reproduction is *not* expected — see
   `07_numerical_verification.md` for the three sources of drift that are
   unavoidable (RNG, default init, `var`/dtype conventions).

2. **Phase 9 (continual learning) is gated on Phase 8 passing.** See
   `08_phase9_gate.md` for the concrete gate. It must not become an
   excuse to start experimenting before reproduction lands.

3. **Remote-sensing "plastic model" is a separate track.** See
   `09_remote_sensing_plastic_model.md`. It defines interface
   requirements that the text-path implementation must satisfy *without*
   adding architectural complexity to the reproduction itself.

Read order: `01_source_analysis.md` → the contracts → verification.

Every spec used to mark items that must be confirmed against the MLX runtime
as `[VERIFY]`. **All seven are now RESOLVED** — MLX cannot execute in this
environment (Linux wheel is CUDA-only, no GPU), so the defaults were confirmed
by reading the installed `mlx` 0.32.2 package source directly. Citations and
the exact consequences for the port are in `01_source_analysis.md` under
"RESOLVED — MLX 0.32.2 defaults".

One decision is still open and blocks Phase A: the trace buffers
(`states`/`decaytrace`/`embedtrace`) are genuinely in MLX's
`trainable_parameters()`, so the optimizer clobbers the manually-set
`stop_gradient` state by ~1.6e-3/step. Bug-for-bug vs. fix — see 01 #7.