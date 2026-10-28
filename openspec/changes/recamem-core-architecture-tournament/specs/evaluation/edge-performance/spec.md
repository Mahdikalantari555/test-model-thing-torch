# Spec Delta

## ADDED Requirements

### Requirement: Cross-tokenizer held-out bits-per-byte
Whenever language-modeling quality is evaluated or compared across tokenizers (such as UTF-8 bytes `V=256` versus compact BPE `V=4096`), the benchmark harness SHALL compute and report `held_out_bpb` defined as total held-out negative log-likelihood in nats divided by `(evaluated_utf8_bytes * ln(2))` over the exact same excluded UTF-8 byte spans, enforcing the 64-byte window contamination check prior to reporting.

#### Scenario: Byte and BPE models are compared on identical held-out text
- **WHEN** a byte-level checkpoint and a BPE checkpoint are evaluated on the same held-out split
- **THEN** both reports SHALL include `held_out_bpb`, `evaluated_utf8_bytes`, `evaluated_tokens`, and `tokenizer_vocab_size`
- **THEN** raw token-level perplexity SHALL NOT be used to rank models with different vocabularies

### Requirement: Prefill time-to-first-token and dynamic INT8 decode throughput
In addition to fp32 sequential generation throughput, the harness SHALL measure prefill time-to-first-token (`ttft_ms`) across prompt lengths ($T \in \{128, 512, 1024, 4096\}$) and post-quantization CPU decode throughput (`int8_tokens_per_second`, `int8_bytes_per_second`, and quantized artifact size in megabytes) under the mandatory warmup, median-of-runs, variance (`CV <= 0.10`), and system-load (`load_ratio <= 0.40`) guards.

#### Scenario: Quantized and prefill timings obey variance and load guards
- **WHEN** `ttft_ms` and dynamic-INT8 decode throughput are benchmarked on CPU
- **THEN** the report SHALL record median values, all timed runs, coefficient of variation, thread count, and RSS alongside fp32 baselines
- **WHEN** machine load or run-to-run variance exceeds the documented guard thresholds
- **THEN** the affected timing metrics SHALL be reported as `skipped` with the measured guard value and no numeric rate

### Requirement: Paired equal-byte and equal-wall-clock comparison tables
Every multi-candidate architecture benchmark report SHALL emit both an **Equal-Data-Budget** comparison table (matched UTF-8 training bytes consumed) and an **Equal-Wall-Clock-Budget** comparison table (matched CPU training seconds on identical thread counts) so sample efficiency and compute throughput are separated cleanly.

#### Scenario: Fast core vs sample-efficient core is visible in dual tables
- **WHEN** two candidate cores with different training bytes-per-second complete a tournament track
- **THEN** the report SHALL emit separate metrics at the shared byte milestone and at the shared wall-clock milestone
- **THEN** neither milestone SHALL be omitted or interpolated from unmeasured checkpoints
