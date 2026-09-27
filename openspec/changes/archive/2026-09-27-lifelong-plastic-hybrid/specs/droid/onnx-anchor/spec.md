# Spec Delta: ONNX Semantic Anchor

## Purpose

Provides a sub-25MB, high-speed ONNX runtime semantic feature extractor that anchors the RTU memory layer to prevent syntactic and representation collapse.

## ADDED Requirements

### Requirement: ONNX Inference Pipeline
The system SHALL execute semantic feature extraction using an ONNX runtime session over quantized `all-MiniLM-L6-v2` loaded from `/home/asus/.cache/huggingface`.

#### Scenario: Sub-15ms CPU latent extraction
- **WHEN** a sentence or text chunk is passed to the ONNX anchor
- **THEN** the model SHALL output a 384-dimensional normalized semantic vector in under 15 milliseconds on CPU

#### Scenario: Cache persistence
- **WHEN** the model is initialized
- **THEN** weights and tokenizers SHALL load from `/home/asus/.cache/huggingface` without re-downloading if cached
