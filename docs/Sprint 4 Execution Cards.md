# Sprint 4 Execution Cards: Database Intelligence

This document tracks the retroactive execution cards for Sprint 4, aligned with the Strict AI Execution Specifications (v8.0).

---

### Task ID
BRM-S4-001

### Title
Memory DB Tiering with Decay and Reinforcement

### Objective
Implement a multi-tiered memory storage backend (working, episodic, semantic) that decays unused memories and reinforces frequently accessed records.

### Scope
- `src/cognitive/memory.rs`

### Non-Goals
- Do not store raw large files inside memory tiers.
- Do not implement custom file systems.

### Dependencies
- BRM-S1-001 (Stable Core setup).

### Inputs
- Telemetry events.
- Active workspace memory entities.

### Steps
1. Create memory structures representing semantic, episodic, and working memory slots.
2. Implement memory reinforcement formulas applied on reads to boost retrieval scores.
3. Build automatic decay loops reducing the prominence of older unread items.

### Outputs
- `src/cognitive/memory.rs` with multi-tiered memory management and reinforcement.

### Acceptance Criteria
- [x] Memory accesses correctly reinforce target record scores.
- [x] Decayed elements are pruned or demoted when values drop below configured thresholds.

### Failure Modes
- If memory allocations reach maximum memory caps, drop oldest episodic items first.

### Rollback
- Bypass decay processing and keep all items active indefinitely.

### Tests
- Unit: `cognitive::memory::tests::test_proactive_memory_injection`
- Unit: `cognitive::memory::tests::test_memory_tier_lifecycle_decay_reinforcement`

### Regression Risks
- Slow search iteration if memory decay collections grow very large.

---

### Task ID
BRM-S4-002

### Title
Semantic Memory Promotion and Consolidation

### Objective
Provide automated background consolidation loops to promote persistent episodic experiences into consolidated semantic memories.

### Scope
- `src/cognitive/memory.rs`

### Non-Goals
- Do not create external operating system threads outside Tokio runtime tasks.

### Dependencies
- BRM-S4-001 (Memory DB).

### Inputs
- Epoched episodic entries.

### Steps
1. Implement a consolidation scheduler running inside standard async background tasks.
2. Analyze episodic records to extract clusters of high-association facts.
3. Promote highly consolidated items into the semantic tier and clean up duplicate details.

### Outputs
- Consolidation job scheduler and semantic promotion rules.

### Acceptance Criteria
- [x] Ephemeral details consolidate cleanly into semantic representations.
- [x] Consolidation jobs execute cleanly without locking or blocking interactive queries.

### Failure Modes
- Handle and log database lock contentions without causing job terminations.

### Rollback
- Revert semantic promotion and preserve raw episodic inputs as standard records.

### Tests
- Unit: Integrated testing of memory consolidation routines in `memory.rs`.

### Regression Risks
- Background compute overhead during episodic consolidation.

---

### Task ID
BRM-S4-003

### Title
Answer Trace Persistence and Route History

### Objective
Implement atomic answer trace logging and route history persistence to record execution decisions, latency profiles, and model pathways in SQLite.

### Scope
- `src/cognitive/analytics.rs`

### Non-Goals
- Do not log raw tensor values or model parameters.

### Dependencies
- BRM-S1-002 (SQLite WAL core).

### Inputs
- Planning choices, models queried, and response statistics.

### Steps
1. Add `analytics.rs` with schemas for tracking latency, metrics, and routing histories.
2. Build SQL statement bindings to insert query runs thread-safely.
3. Provide analytical queries to extract latency and routing percentiles.

### Outputs
- Persistent sqlite analytics logging structures.

### Acceptance Criteria
- [x] Answer queries record their exact routing history into SQLite.
- [x] Average timing and percentile calculations yield correct math on stored rows.

### Failure Modes
- Logging failures degrade silently to print statements to prevent query blockages.

### Rollback
- Write timing profiles directly to text logs instead of SQL databases.

### Tests
- Unit: `cognitive::analytics::tests::test_analytics_persistence_and_stats`

### Regression Risks
- Database write latency if disk performance is heavily constrained.

---

### Task ID
BRM-S4-004

### Title
Feedback Events and Reusable Workflow Objects

### Objective
Enable reinforcement feedback logging and structured workflow graph storage to support repeatable task routing.

### Scope
- `src/cognitive/controller.rs`
- `src/cognitive/goal_graph.rs`
- `src/cognitive/router.rs`

### Non-Goals
- Do not construct graphical workflow builders or node-editor user interfaces.

### Dependencies
- BRM-S3-005 (Goal graph).

### Inputs
- Feedback grades and sequence steps.

### Steps
1. Create controller endpoints and structures to parse execution feedback events.
2. Save sequence schedules as reusable workflow graphs in memory.
3. Query controller records to tune future dynamic routing decisions.

### Outputs
- Structured feedback loops and reusable workflow definitions.

### Acceptance Criteria
- [x] System parses, saves, and reinforces routes using execution feedback.
- [x] Reusable workflow objects serialize and reload successfully.

### Failure Modes
- Malformed feedback events return clean HTTP error payloads safely.

### Rollback
- Disable dynamic adaptation and execute fixed baseline routing rules.

### Tests
- Unit: `cognitive::controller::tests::test_adaptive_controller_adaptation_loops`
- Unit: `cognitive::goal_graph::tests::test_goal_graph_decomposition_and_execution`

### Regression Risks
- Routing oscillations if feedback rules are tuned too aggressively.
