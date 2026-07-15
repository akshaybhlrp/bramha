# Sprint 1 Execution Cards

This document tracks the retroactive execution cards for Sprint 1, built to align with the new Strict AI Execution Specifications (v8.0).

---

### Task ID
BRM-S1-001

### Title
Rust-only single binary setup

### Objective
Establish the foundational Rust-native, single-binary architecture with CLI execution capabilities, ensuring no Python or managed runtime dependencies.

### Scope
- `Cargo.toml`
- `src/main.rs`
- `src/cli.rs` (or equivalent CLI module)

### Non-Goals
- Do not implement HTTP servers or REST APIs.
- Do not implement complex storage layers yet.

### Dependencies
- None. This is the root setup.

### Inputs
- Initial `cargo new` template.

### Steps
1. Configure `Cargo.toml` with basic dependencies (e.g., `clap`, `tokio`).
2. Create `src/main.rs` as the single entry point.
3. Implement a basic CLI parser using `clap` to accept commands like `run`, `serve` (placeholder), and `init`.
4. Ensure the binary compiles and runs purely in Rust.

### Outputs
- `Cargo.toml` with initial deps.
- `src/main.rs` with tokio runtime and CLI entrypoint.
- CLI module.

### Acceptance Criteria
- [x] Binary compiles cleanly with `cargo build`.
- [x] Running `bramha --help` displays available CLI commands.
- [x] No Python, C++, or external runtime dependencies exist in the execution path.

### Failure Modes
- If CLI parsing fails, abort with a standard clap error message.

### Rollback
- Since this is the initial setup, rollback is reverting the repository to an empty state.

### Tests
- Unit: CLI parser correctly routes dummy commands.
- Manual: `cargo run -- --help` prints help text.

### Regression Risks
- Dependency bloat if unnecessary crates are added.

### Notes
- Architecture Invariant 1: Rust only. Invariant 3: Single binary by default.

---

### Task ID
BRM-S1-002

### Title
SQLite WAL metadata core

### Objective
Implement the `AnalyticsStore` and collection index tables backed by SQLite in Write-Ahead Logging (WAL) mode for fast, concurrent metadata control.

### Scope
- `src/storage/metadata_sql.rs` (or `analytics.rs`)
- `Cargo.toml`

### Non-Goals
- Do not store raw tensor data or embeddings in SQLite.
- Do not implement vector indexing here.

### Dependencies
- BRM-S1-001 (Rust single binary setup).

### Inputs
- Rusqlite dependency.
- Schema requirements for collections, sessions, and models.

### Steps
1. Add `rusqlite` to `Cargo.toml`.
2. Implement SQLite connection initialization with `PRAGMA journal_mode=WAL;`.
3. Create schemas for collections, index tables, and sessions.
4. Provide basic connection pooling or thread-safe access.

### Outputs
- SQLite connection logic with WAL mode.
- Initialization migrations/schemas.

### Acceptance Criteria
- [x] SQLite database file is created on disk upon initialization.
- [x] `journal_mode=WAL` is confirmed active.
- [x] Multiple concurrent reads/writes to metadata do not cause `database is locked` panics unexpectedly.

### Failure Modes
- If SQLite initialization fails (e.g., permission denied), the system must fail to start loudly.

### Rollback
- Revert schema migrations manually if schema is invalid. No feature flag needed for core init.

### Tests
- Unit: Initialize DB in memory, apply schema, and verify WAL mode.
- Integration: Concurrent write test to verify WAL concurrency.

### Regression Risks
- File locking issues on certain filesystems (e.g., NFS).

### Notes
- "Intelligence lives in the database." The metadata DB is the control plane.

---

### Task ID
BRM-S1-003

### Title
Model registry functionality

### Objective
Implement `RegistryEntry` and `TensorDB` logic to register, pull, and manage local LLM models and metadata securely.

### Scope
- `src/models/registry.rs`
- `src/models/tensor_db.rs`
- `src/cli.rs` (pull command)

### Non-Goals
- Do not execute inference in this task.
- Do not implement dynamic adapter routing.

### Dependencies
- BRM-S1-002 (SQLite WAL core for storing registry metadata).

### Inputs
- Basic CLI scaffolding.
- SQLite metadata store.

### Steps
1. Define `RegistryEntry` struct to represent a registered model (ID, architecture, parameter count, paths).
2. Implement `pull` CLI command to register dummy/local models.
3. Persist model metadata to the SQLite metadata DB.
4. Implement `TensorDB` to manage local paths for the model weights.

### Outputs
- Registry structures.
- DB persistence logic for models.
- `pull` CLI command implementation.

### Acceptance Criteria
- [x] `bramha pull <model_name>` successfully registers a model.
- [x] Model metadata is durably stored in the SQLite DB.
- [x] Registry can list available models cleanly.

### Failure Modes
- If model pull/registration fails, the DB transaction must rollback so no orphaned metadata exists.

### Rollback
- Delete the registered model entry from the DB.

### Tests
- Unit: Register a mock model and read it back from the registry.
- Integration: `pull` command adds to DB, `list` command retrieves it.

### Regression Risks
- Invalid paths stored in registry.

### Notes
- All models must be explicitly registered before they can be loaded by the inference engine later.

---

### Task ID
BRM-S1-004

### Title
In-process tokenizer

### Objective
Natively integrate a Rust-based tokenizer (e.g., HF tokenizers) to run in the same process without external HTTP calls or Python subprocesses.

### Scope
- `src/inference/tokenizer.rs`
- `Cargo.toml`

### Non-Goals
- Do not write a custom BPE algorithm from scratch; use the HuggingFace `tokenizers` crate.
- Do not stream tokens to clients yet.

### Dependencies
- BRM-S1-001 (Rust setup).

### Inputs
- `tokenizers` crate.
- Standard `tokenizer.json` file format.

### Steps
1. Add `tokenizers` crate to `Cargo.toml`.
2. Create `BramhaTokenizer` wrapper struct.
3. Implement `encode` (text to tokens) and `decode` (tokens to text) methods.
4. Ensure the tokenizer loads natively from disk.

### Outputs
- `BramhaTokenizer` implementation.
- Tests for encode/decode.

### Acceptance Criteria
- [x] Tokenizer loads a standard `tokenizer.json` successfully.
- [x] Encode produces deterministic token IDs.
- [x] Decode perfectly reconstructs the string for standard English text.
- [x] Tokenization is strictly in-process (no subprocesses).

### Failure Modes
- If `tokenizer.json` is missing or invalid, fail loudly upon load.

### Rollback
- Not applicable; this is a foundational requirement for inference.

### Tests
- Unit: Encode "hello world" and decode back to verify round-trip fidelity.

### Regression Risks
- Thread contention if tokenizer is not instantiated per-thread or protected correctly.

### Notes
- Tokenization must be fast and zero-overhead relative to Python wrappers.

---

### Task ID
BRM-S1-005

### Title
Atomic write helper for shard and manifest persistence

### Objective
Implement atomic write behavior for shard and manifest writes so a crash can never leave a partially written final file.

### Scope
- `src/storage/atomic_write.rs`
- `src/storage/decomposed_store.rs`
- `src/storage/manifest.rs`

### Non-Goals
- Do not change WAL format.
- Do not change snapshot format.
- Do not refactor unrelated storage code.

### Dependencies
- Existing storage paths and manifest persistence code must compile.

### Inputs
- Current shard write paths.
- Temp-file naming convention.

### Steps
1. Add a shared atomic write helper that writes to a temp file in the same directory.
2. Ensure file contents are flushed and synced before rename.
3. Rename temp file to final path using atomic rename semantics.
4. Replace direct final-path writes in storage modules with the helper.
5. Add crash-simulation tests for interrupted writes.

### Outputs
- New atomic write helper.
- Storage modules updated to use it.
- Crash and checksum tests.

### Acceptance Criteria
- [x] Crash during write never leaves a partially written final file.
- [x] Old file remains valid if temp write or rename fails.

### Failure Modes
- If temp write fails, abort write and preserve old file.
- If fsync fails, abort rename and preserve old file.

### Rollback
- Revert call sites to previous direct-write path only if helper causes blocking issues.

### Tests
- Unit: temp write success, fsync failure handling.
- Integration: persist shard and reload successfully.
- Recovery: simulate crash between temp write and rename.

### Regression Risks
- Slower writes on low-end disks.
- Cross-platform rename differences.

### Notes
- Final file must only become visible after successful durable temp write.

---

### Task ID
BRM-S1-006

### Title
WAL replay and transaction recovery

### Objective
Implement a Write-Ahead Log (WAL) replay manager to ensure durable transaction recovery for the intelligence database across crashes.

### Scope
- `src/storage/wal.rs`
- `src/storage/wal_manager.rs`

### Non-Goals
- Do not implement snapshotting or compaction yet.
- Do not replace SQLite's own WAL; this is for our custom append-only data stores (tensors/memories).

### Dependencies
- BRM-S1-005 (Atomic write helper).

### Inputs
- Custom data store designs (tensor DB, log-structured files).

### Steps
1. Define WAL entry serialization format (e.g., JSON or bincode).
2. Implement append-only WAL writer with `fsync`.
3. Implement WAL reader that scans logs sequentially on boot.
4. Implement replay logic that applies pending WAL entries to the in-memory state or disk.

### Outputs
- `WalManager` implementation.
- WAL append and replay methods.

### Acceptance Criteria
- [x] Appended entries are durably flushed to disk.
- [x] Upon startup, `WalManager` reads the log and perfectly reconstructs the un-compacted state.
- [x] Corrupted or partially written WAL entries at the tail are safely discarded without crashing.

### Failure Modes
- If WAL disk runs out of space, write fails and transaction aborts, leaving system consistent.

### Rollback
- None. WAL is foundational to data integrity.

### Tests
- Unit: Append 10 entries, read 10 entries back.
- Recovery: Append entries, simulate crash by intentionally truncating the last entry, boot and verify the first valid entries are replayed safely.

### Regression Risks
- Slow startup times if WAL grows too large (compaction needed in Sprint 4).

### Notes
- "No silent corruption — every storage path must fail loudly or degrade safely."

---

### Task ID
BRM-S1-007

### Title
Basic CRUD over collections, documents, chunks, sessions, models

### Objective
Implement complete Create, Read, Update, and Delete (CRUD) operations in the metadata DB for all core entities.

### Scope
- `src/storage/crud.rs` (or equivalent models/collections modules)
- `src/api/` (if basic accessors exist)

### Non-Goals
- Do not implement graph relationships or semantic embeddings yet.

### Dependencies
- BRM-S1-002 (SQLite WAL core).

### Inputs
- SQLite schemas for core entities.

### Steps
1. Write specific insert, select, update, and delete SQL functions for `Collection`.
2. Write CRUD for `Document` and `Chunk`.
3. Write CRUD for `Session`.
4. Ensure constraints (e.g., ON DELETE CASCADE) are enforced.

### Outputs
- Complete Rust data access layer for basic entities.

### Acceptance Criteria
- [x] Creation of each entity returns a valid UUID/ID and persists to DB.
- [x] Reading an entity returns exactly the data saved.
- [x] Deleting a collection cascades and deletes associated documents and chunks.
- [x] Updates safely modify target rows without corrupting adjacent data.

### Failure Modes
- If an entity violates a unique constraint or foreign key, return a clean `Result::Err` rather than a panic.

### Rollback
- Not applicable.

### Tests
- Integration: Full lifecycle test (Create Collection -> Add Document -> Add Chunks -> Update Doc -> Delete Collection -> Verify all are gone).

### Regression Risks
- Missed foreign key constraints leading to orphaned database rows.

### Notes
- CRUD is the basis of the Intelligence DB. "Intelligence lives in the database."
