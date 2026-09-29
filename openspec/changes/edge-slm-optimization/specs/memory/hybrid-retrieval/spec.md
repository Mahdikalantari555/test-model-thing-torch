# Spec Delta

## Purpose

Specify scalable dense candidate retrieval, bounded cross-encoder reranking, and adaptive-dimensional Matryoshka retrieval while preserving lexical/RRF behavior.

## ADDED Requirements

### Requirement: HNSW candidate retrieval at 100,000 facts
The system SHALL provide an `HNSWIndex` dense-candidate path for 100,000 normalized fact embeddings and SHALL preserve SQLite FTS5 lexical candidates and RRF fusion. On the documented reference CPU with a warm index, p95 dense retrieval latency SHALL target below 5 ms and recall@5 SHALL be at least 0.95 versus the exact dense baseline.

#### Scenario: Retrieve at 100k scale
- **WHEN** the index contains 100,000 active facts and a query is issued using the documented benchmark configuration
- **THEN** HNSW SHALL return dense candidates with p95 latency below 5 ms
- **THEN** recall@5 against exact dense search SHALL be at least 0.95
- **THEN** lexical candidates SHALL still contribute to RRF and exact-term queries SHALL remain retrievable

#### Scenario: Exact fallback
- **WHEN** HNSW is unavailable, invalid, or disabled in the profile
- **THEN** the store SHALL fall back to the exact dense candidate path without corrupting SQLite facts
- **THEN** the public recall result schema SHALL remain compatible

### Requirement: Cross-encoder reranking
The system SHALL optionally rerank a bounded candidate set with `CrossEncoderReranker`. On a versioned labeled query set, the target SHALL be at least 20% relative improvement in Precision@5 over the same pre-rerank candidate order.

#### Scenario: Rerank top candidates
- **WHEN** reranking is enabled for a query
- **THEN** only the configured top candidate limit SHALL be scored
- **THEN** the final order SHALL use cross-encoder scores with the configured fusion policy
- **THEN** pre/post ranks and timing SHALL be available for evaluation

#### Scenario: Measure reranker gain
- **WHEN** the fixed labeled retrieval benchmark runs with and without reranking
- **THEN** it SHALL report both Precision@5 values and relative gain
- **THEN** the +20% target SHALL be considered met only if the measured relative gain is at least 20%

### Requirement: 64-D Matryoshka fast path
The system SHALL support a validated 64-dimensional first-stage retrieval representation and a 384-dimensional slow/confirmation path. The 64-D path SHALL target at least 6x lower first-stage latency with no more than 5% relative retrieval-quality loss versus the 384-D baseline on the same benchmark.

#### Scenario: Fast-path retrieval and slow confirmation
- **WHEN** a query is processed with Matryoshka retrieval enabled
- **THEN** the system SHALL use a compatible 64-D representation to produce candidates
- **THEN** it MAY use the 384-D representation to rerank a bounded candidate set
- **THEN** measured first-stage latency and retrieval-quality loss SHALL be reported against the 384-D baseline

#### Scenario: Incompatible embedding version
- **WHEN** stored embeddings do not match the configured Matryoshka representation/version
- **THEN** the system SHALL re-embed or use a compatible exact fallback
- **THEN** it SHALL NOT silently truncate an arbitrary 384-D vector to 64 dimensions
