# Sprint 5 Execution Cards: Multi-Model System

This document tracks the retroactive execution cards for Sprint 5, aligned with the Strict AI Execution Specifications (v8.0).

---

### Task ID
BRM-S5-001

### Title
Model Capability Registry and Backend Profiles

### Objective
Implement the catalog of architectural model families (Llama, Gemma, Mistral), context limits, and target backend profiles to guide routing choices.

### Scope
- `src/cognitive/adapter.rs`

### Non-Goals
- Do not download weights from model hubs inside registry listings.

### Dependencies
- BRM-S1-001 (Stable Core setup).

### Inputs
- Model metadata schemas.

### Steps
1. Create `ModelCapabilityRegistry` storing parameters, token bounds, and quantization options.
2. Implement `BackendCapabilityProfile` describing CPU vs. GPU configurations.
3. Provide spec verification methods for registered models.

### Outputs
- Registered model capability and backend profiling structures in `adapter.rs`.

### Acceptance Criteria
- [x] Correctly identifies capability mismatches for raw models.
- [x] Resolves precise layer counts and hardware constraints for Llama/Gemma.

### Failure Modes
- Default unsupported architectures to standard baseline capability limits safely.

### Rollback
- Revert capability mappings to simple parameter size checks.

### Tests
- Unit: `cognitive::adapter::tests::test_model_capability_registry`
- Unit: `cognitive::adapter::tests::test_backend_capability_profile`
- Unit: `cognitive::adapter::tests::test_llama_adapter_specifications`
- Unit: `cognitive::adapter::tests::test_gemma_adapter_specifications`

### Regression Risks
- Missing model capability specifications causing routing stalls.

---

### Task ID
BRM-S5-002

### Title
Dynamic SLA-Based Router with Benchmark Integration

### Objective
Select execution routes dynamically by checking prompt context sizes and target query SLA profiles against benchmark timing matrices.

### Scope
- `src/cognitive/router.rs`

### Non-Goals
- Do not build complex user timing settings inside this backend routing step.

### Dependencies
- BRM-S5-001 (Model capability registry).

### Inputs
- Target latency constraint, prompt details, and active model lists.

### Steps
1. Create `ModelRouter` implementing complexity routing evaluations.
2. Read timing information from previous benchmarks or active runtime trackers.
3. Choose the optimal model family satisfying context boundaries and SLA constraints.

### Outputs
- SLA-based model routing engine in `router.rs`.

### Acceptance Criteria
- [x] Fast low-latency routes are dynamically chosen for small prompts.
- [x] Complex or long-context queries dynamically fall back to deep/large models.

### Failure Modes
- If no model satisfies the strict SLA targets, gracefully route queries to the fastest baseline model.

### Rollback
- Revert routing behavior to utilize a single static default model.

### Tests
- Unit: `cognitive::router::tests::test_model_router_complexity_routing_rules`
- Unit: `cognitive::router::tests::test_benchmark_based_routing_sla_fallback`

### Regression Risks
- Sub-optimal model selections if benchmark timings are heavily skewed.

---

### Task ID
BRM-S5-003

### Title
Multi-Model RAG Pipeline Execution

### Objective
Orchestrate a sequential multi-model pipeline running draft generations, context lookup, sentence grounding scans, and verifications.

### Scope
- `src/cognitive/pipeline.rs`

### Non-Goals
- Do not block or lock the database during pipeline executions.

### Dependencies
- BRM-S3-002 (Hybrid retrieval).
- BRM-S5-002 (SLA router).

### Inputs
- Query requests, SLA rules.

### Steps
1. Build `MultiModelPipeline` coordinating multi-step processing sequences.
2. Query the draft model for speedy initial response sketches.
3. Retrieve supporting text, build overlap maps, and check validation boundaries.

### Outputs
- Unified pipeline manager executing multi-model flows.

### Acceptance Criteria
- [x] Pipeline executes all sequential steps and produces valid results.
- [x] Intermediate results pass properly across stages without data loss.

### Failure Modes
- If draft models fail, pass execution straight to baseline models.

### Rollback
- Bypass multi-step pipelines and directly execute raw queries on baseline models.

### Tests
- Unit: `cognitive::pipeline::tests::test_multi_model_pipeline_flow`

### Regression Risks
- Cumulative processing delays if multiple stages suffer from latency issues.

---

### Task ID
BRM-S5-004

### Title
Self-Correction Grounding Verifier Mode

### Objective
Add a verifier mechanism that evaluates generated sentences for contextual grounding and triggers automated self-correction loops when errors occur.

### Scope
- `src/cognitive/verifier.rs`

### Non-Goals
- Do not run infinite correction loops (apply a max execution limit).

### Dependencies
- BRM-S3-003 (Evidence sentence mapping).

### Inputs
- Overlap evidence maps, draft completions.

### Steps
1. Parse evidence results and compute grounding index boundaries.
2. If grounding ratios drop below target thresholds, formulate self-correction instructions.
3. Run the verifier critic model to rewrite ungrounded claims into contextually validated facts.

### Outputs
- Self-correcting verifier loop in `verifier.rs`.

### Acceptance Criteria
- [x] Ungrounded sentences trigger corrective rewrite cycles correctly.
- [x] Grounded outputs exit early without triggering unnecessary corrections.

### Failure Modes
- Exiting correction loops immediately if the verifier encounters errors.

### Rollback
- Disable verification checks and output the original raw drafts immediately.

### Tests
- Unit: `cognitive::verifier::tests::test_verifier_critic_self_correction_rewrite`

### Regression Risks
- Double or triple execution latency when corrections are triggered.
