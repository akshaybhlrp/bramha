# Sprint 6 Execution Cards: Planner Engine

This document tracks the retroactive execution cards for Sprint 6, aligned with the Strict AI Execution Specifications (v8.0).

---

### Task ID
BRM-S6-001

### Title
Planner Policy and Warm-State Persistence

### Objective
Implement policy threshold configurations governing planning paths and handle thread-safe JSON serialization for warm-state persistence.

### Scope
- `src/planner/policy.rs`

### Non-Goals
- Do not build browser-based policy settings inside this module.

### Dependencies
- BRM-S1-001 (Stable Core setup).

### Inputs
- Policy threshold values.

### Steps
1. Create `PlannerPolicy` containing target latency, speculative limits, and caching options.
2. Build default configuration initializers.
3. Write serialization and deserialization routines to maintain policies across engine restarts.

### Outputs
- `src/planner/policy.rs` implementing planning policy state and JSON persistence.

### Acceptance Criteria
- [x] Policies deserialize from disk cleanly upon initialization.
- [x] Default values correctly populate missing properties.

### Failure Modes
- If loaded JSON files are corrupted, fall back to safe default policy thresholds.

### Rollback
- Revert policy options to fixed compile-time constants.

### Tests
- Unit: `planner::policy::tests::test_policy_default_and_roundtrip`

### Regression Risks
- Config lockups if parallel asynchronous tasks attempt overlapping writes to the policy file.

---

### Task ID
BRM-S6-002

### Title
Analytical Latency Cost Modeling

### Objective
Provide cost estimation metrics predicting latencies of alternative execution paths based on prompt sizing, target model properties, and hardware profiles.

### Scope
- `src/planner/cost_model.rs`

### Non-Goals
- Do not perform actual forward passes or dynamic GPU measurements in this mathematical cost analyzer.

### Dependencies
- BRM-S1-001 (Stable Core setup).

### Inputs
- Candidate route profiles, prompt token counts.

### Steps
1. Create `PlannerCostModel` defining mathematical latency models.
2. Build scoring formulas comparing cached exits, speculative draft matching, and parent exact loops.
3. Order paths analytically by projected execution time.

### Outputs
- Cost-scoring and path-ordering calculations in `cost_model.rs`.

### Acceptance Criteria
- [x] Cache path timing is scored as lowest-latency choice.
- [x] High token complex bounds prioritize speculative paths when accept rate criteria are satisfied.

### Failure Modes
- Handle overflow cases safely when processing extremely large token limits.

### Rollback
- Return simplistic default ordered vectors containing cache first, then speculative, then exact.

### Tests
- Unit: `planner::cost_model::tests::test_cost_model_relative_ordering`

### Regression Risks
- Inaccurate routing choices if real-world latency profiles diverge severely from cost formulas.

---

### Task ID
BRM-S6-003

### Title
Execution Path Optimizer and Fallback Chain

### Objective
Select the optimal query execution path by evaluating prompt complexity, policies, caches, and persistent accept rates, falling back to exact decode on errors.

### Scope
- `src/planner/optimizer.rs`
- `src/inference/engine.rs`

### Non-Goals
- Do not block engine execution if optimization steps encounter errors.

### Dependencies
- BRM-S6-001 (Planner policy).
- BRM-S6-002 (Cost model).

### Inputs
- Queries, policies, cache statuses, historical accept rates.

### Steps
1. Create `ExecutionPathOptimizer` evaluating planning paths.
2. Integrate decision routing inside `InferenceEngine::generate`.
3. Build fallback handling executing exact paths if speculative execution fails midway.

### Outputs
- Dynamic route planning and failover orchestrations in `optimizer.rs` and `engine.rs`.

### Acceptance Criteria
- [x] Generates expected route decisions based on input bounds.
- [x] Bypasses cache paths cleanly if exact-only mode is forced via overrides.
- [x] Failures on optimized paths degrade gracefully to exact decodes.

### Failure Modes
- Dynamic routing failures catch and log errors, seamlessly continuing query execution on the exact path.

### Rollback
- Force environment settings to bypass optimization and run exact decoding for all queries.

### Tests
- Unit: `planner::optimizer::tests::test_optimizer_decision_flow`
- Integration: `inference::engine::tests::test_planner_exact_only_override`
- Integration: `inference::engine::tests::test_heterogeneous_scheduler_midway_fallback`

### Regression Risks
- Query latency overheads if optimization loops add timing delays.

---

### Task ID
BRM-S6-004

### Title
Deterministic Context-Hashing Cache

### Objective
Build a deterministic cached-answer structure using unique context-hashing keys to support instant early-exits for exact prompt matches.

### Scope
- `src/storage/answer_cache.rs`

### Non-Goals
- Do not cache fuzzy or non-exact matches inside this deterministic cache.

### Dependencies
- BRM-S1-005 (Atomic writes).

### Inputs
- Prompts, versions, RAG context vectors.

### Steps
1. Implement `compute_context_hash` generating deterministic hashes.
2. Build cache search, insert, and age-based expiration routines.
3. Enforce directory isolation using the database's configured `planner_cache_path`.

### Outputs
- Grounded answer caching and lookup engine in `answer_cache.rs`.

### Acceptance Criteria
- [x] Cache retrieves responses perfectly on identical contexts.
- [x] Modifying the context hash results in clean cache misses.
- [x] Expired entries are safely evicted.

### Failure Modes
- Cache file corruptions degrade gracefully to cache-miss flows.

### Rollback
- Revert cache lookups to standard execution.

### Tests
- Unit: `storage::answer_cache::tests::test_answer_cache_insert_retrieval_and_expiry`
- Integration: `inference::engine::tests::test_planner_cache_hit_path`

### Regression Risks
- False cache hits if key hashing experiences collisions.

---

### Task ID
BRM-S6-005

### Title
SQLite Persistent Trace Telemetry

### Objective
Maintain thread-safe SQLite traces of planning telemetry, recording selected execution routes, model versions, and speculative acceptance rates.

### Scope
- `src/storage/metadata_sql.rs`

### Non-Goals
- Do not run continuous heavy table index rebuilds.

### Dependencies
- BRM-S1-002 (SQLite WAL core).

### Inputs
- Selected paths, accept rate floats, and query texts.

### Steps
1. Add telemetry schema tracking routes and accept rates.
2. Implement transaction-safe record additions.
3. Compute historical speculative token accept rate averages.

### Outputs
- Database tables and analytical queries tracking planning metrics in `metadata_sql.rs`.

### Acceptance Criteria
- [x] Execution telemetry saves details accurately into SQLite.
- [x] Historical accept rate calculations return correct metrics across recent executions.

### Failure Modes
- Database locks or write failures log warning messages without interrupting queries.

### Rollback
- Disable analytics database writes and log metrics directly to text logs.

### Tests
- Unit: `storage::metadata_sql::tests::test_planner_sql_persistence_and_stats`

### Regression Risks
- Storage amplification if query trace logs grow indefinitely without cleanup rules.
