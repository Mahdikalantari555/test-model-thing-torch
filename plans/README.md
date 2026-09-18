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

Every spec marks items that must be confirmed against the MLX runtime as
`[VERIFY]`. Those are the things I could not derive purely from reading
source and that would silently break numerical similarity if guessed wrong.