# portable-brain Specification

## Purpose

Portable `.droid` brain archive format bundling Droid configuration, plastic memory weights, and knowledge store into a single self-contained ZIP file with SHA-256 integrity verification.

## Requirements

### Requirement: ZIP archive format
The system SHALL package a Droid's complete state into a `.droid` ZIP archive containing `manifest.json`, `config.json`, `memory.safetensors`, and `knowledge.db`.

#### Scenario: Archive creation
- **WHEN** a Droid profile is exported
- **THEN** the `.droid` archive SHALL contain the following members:
  - `manifest.json`: Metadata including base anchor version, dimensions, creation timestamp, and SHA-256 checksums
  - `config.json`: Profile settings, decay parameters, learning rate schedules
  - `memory.safetensors`: Plastic RTU trace buffers (`states`, `decay`, `proj_weight`)
  - `knowledge.db`: SQLite database containing FTS5 tables, embeddings, and belief states
  - `train_log.json`: Auditable history of all learning sessions

#### Scenario: Archive restoration
- **WHEN** a `.droid` archive is imported
- **THEN** the system SHALL verify SHA-256 checksums of all archive members
- **THEN** the system SHALL extract members to the target Droid profile directory
- **THEN** the system SHALL load the Droid state from the extracted files

### Requirement: Integrity verification
The system SHALL verify SHA-256 checksums of all archive members during import to detect corruption or tampering.

#### Scenario: Checksum verification
- **WHEN** a `.droid` archive is imported
- **THEN** the system SHALL compute SHA-256 for each member and compare against `manifest.json`
- **THEN** a mismatch SHALL raise a clear integrity error and abort the import

### Requirement: Security hardening
The system SHALL prevent zip-slip path traversal attacks during archive extraction.

#### Scenario: Zip-slip protection
- **WHEN** a `.droid` archive contains a member with a path containing `../` or absolute paths
- **THEN** the system SHALL reject the member and abort extraction
- **THEN** the system SHALL log a security warning