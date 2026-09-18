# 06 Remote Sensing Plastic Model (Separate Track)

## Status

A design track, NOT part of the reproduction. Zero code yet.
Must not add architectural complexity to the reproduction.

## What it is

A future model for the remote sensing domain that learns along the way
(online / continual adaptation). Uses the RTU memory mechanism as its
core, but with different I/O than the byte-level text model.

## Interface requirements the text implementation must expose

These are constraints on the reproduction's design, chosen so the
plastic model can reuse it later without a rewrite:

1. The RTU Layer must be I/O-agnostic. Its core (state update, trace
   buffers, custom grads) must not hard-code the byte vocabulary.
   The 256 in embedtrace should be a configurable `vocab_size` parameter,
   defaulting to 256 for the text path.

2. The Encoder/Decoder pair must be swappable. The text path uses
   Embedding(256, dim) -> Linear(dim, 256). The RS path would use a
   different input projection and output head. These should be separate
   modules passed in, not baked into Model.

3. The loss function must be decomposable. Currently four terms
   (var, pred mse, ce, stop mse). The RS path will reuse var + pred mse
   (latent prediction) but replace ce/stop with domain-specific heads.
   Loss terms should be individually switchable, not one monolithic fn.

4. The custom gradient hooks must work for any input representation,
   not just one-hot bytes. embedtrace uses onehot(c); this should accept
   an arbitrary input embedding vector.

5. Model.reset() must be exposed and callable (it already is).

## What the plastic model needs that the text model does not

- Input: multivariate time series (e.g. spectral bands, time stamps)
- Output: predictions over continuous or categorical targets
- Online adaptation: update state on real data without full retraining
- Possibly: multiple sensors / spatial patches as parallel streams

## Constraint

These requirements are recorded here as design intent. They must NOT
influence the reproduction implementation beyond items 1-5 above.
The reproduction stays byte-level, dim=512, layers=16, four loss terms.
If item 1-5 require a change, it must be a parameter default, not a
structural change.