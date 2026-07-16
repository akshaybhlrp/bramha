# Sprint 7 Execution Cards: Activation Views and Reuse

This document tracks the execution cards for Sprint 7, focusing on robust KV Cache checkpoints, activation replays, and zero-cost branching for long workflows.

---

### Task ID
BRM-S7-001

### Title
Activation Materialized Views & Telemetry

### Objective
Implement a checkpointing system to serialize and map KV Cache states to workflow/branch IDs, enabling instant retrieval of pre-computed activations.

### Scope
- `src/storage/activation_view.rs`
- `src/storage/metadata_sql.rs`

### Inputs
- Prefix KV cache state in memory.
- Current workflow and branch UUID.

### Steps
1. Define the `ActivationMaterializedView` structure holding branch relationships and token offsets.
2. Extend `metadata_sql.rs` SQLite schema to track `activation_views`.
3. Provide insert/query methods in the store to quickly lookup valid checkpoint paths.

### Outputs
- `activation_view.rs` definitions.
- Extended SQLite database schema.

### Acceptance Criteria
- [x] Schema migrations complete successfully without corrupting existing DB traces.
- [x] Views correctly insert and fetch disk paths corresponding to requested workflow IDs.

### Failure Modes
- If DB is locked, safely degrade without panicking (store fails silently).

### Rollback
- Bypass `metadata_sql` caching if schema versions drift.

### Tests
- Unit: Database insertion and retrieval of a mock Activation View record.

### Dependencies
- Existing `metadata_sql.rs` connection pool.

### Non-Goals
- Do not transmit views across the network.

### Regression Risks
- Slow DB locks blocking inference threads.

---

### Task ID
BRM-S7-002

### Title
Branch Checkpoint Replay

### Objective
Provide deterministic capability to reload KV caches up to a specified token index offset, validating integrity via rolling SHA-256 hashes.

### Scope
- `src/inference/paged_kv/branch_replay.rs`
- `src/inference/paged_kv/prefix_cache.rs`

### Inputs
- Required token prompt sequence.
- Candidate `ActivationMaterializedView`.

### Steps
1. Load the binary dump of the KV cache from disk based on the view path.
2. Perform rolling SHA-256 validation (Activation Replay Validation) to verify loaded token bounds match exactly with the incoming prompt subset.
3. Hook this replay logic seamlessly into the inference execution engine pipeline.

### Outputs
- `branch_replay.rs` implementation.
- Hardened `prefix_cache.rs` integration.

### Acceptance Criteria
- [x] Branch resumes exactly at the divergence point.
- [x] Hash mismatches strictly truncate the restored cache to prevent hallucinations.

### Failure Modes
- Corrupt disk file degrades cleanly to full prefill operation.

### Rollback
- None.

### Tests
- Unit: Corrupt cache file causes safe fallback; intact cache hits perfect divergence point.

### Dependencies
- BRM-S7-001 (Activation Views).

### Non-Goals
- Do not implement real-time delta diffing of KV states.

### Regression Risks
- In-memory state corruption if cache loading partially fails.

---

### Task ID
BRM-S7-003

### Title
Planner Cost Integration

### Objective
Inject cache-awareness into the Heterogeneous Planner so it can explicitly route operations to paths featuring cached activation views, scoring them near zero cost for prefill.

### Scope
- `src/planner/cost_model.rs`

### Inputs
- Incoming prompt metadata.
- SQLite query resolving valid `activation_views`.

### Steps
1. Query the `metadata_sql.rs` store during the planner optimization phase.
2. If an exact activation view match exists, deduct the prefill compute complexity from the total milliseconds estimate.
3. Output the updated execution plan.

### Outputs
- Modified `PlannerCostModel`.

### Acceptance Criteria
- [x] Planner demonstrably prefers paths with available branch checkpoints.
- [x] Cached branches win over exact routing latency thresholds.

### Failure Modes
- If cost models underestimate decode times, routing may oscillate.

### Rollback
- Disable cache lookup config flag.

### Tests
- Benchmark: Cost estimates with vs without materialized views.

### Dependencies
- BRM-S7-001 and BRM-S7-002.

### Non-Goals
- No arbitrary remote API integrations for planner.

### Regression Risks
- Over-estimating savings and starving parallel decodes.
