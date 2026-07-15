# EXECUTION_ROADMAP.md — Bramha + SPANDA

> This document describes what we are building now.
> For the long-term vision, see `VISION.md`.
> For post-v1.0 distributed scale, see `HYPERSCALE.md`.

---

## 1. Relationship Definition

- **SPANDA** is a standalone Rust inference engine crate (`spanda-engine`).
- **Bramha** is the intelligence database that consumes SPANDA as its inference backend.
- SPANDA ships first. Bramha layers on top.
- **Crate boundary**: `spanda::InferenceSession` consumed by `bramha::InferenceOrchestrator`.
- **Version pinning**: Bramha locks to a SPANDA release tag, not a branch.

---

## 2. SPANDA Roadmap (Standalone)

### Phase 0: Static Sparse Validation

- **Objective**: Prove that static sparsity on the target model does not destroy quality.
- **Deliverable**: `spanda-tools` converter runs Qwen2-0.5B with static 2:4 sparsity on a fixed fixture of 1000 prompts.
- **Gate**: Top-1 token agreement > 99% vs. dense reference on ≥95% of prompts. Perplexity delta < 0.5%.
- **Fallback**: If gate fails, SPANDA does not ship static sparse. Re-evaluate target model or ship dense-only.
- **What this replaces**: The old "shadow mode" concept. SPANDA is a library; there is no production traffic to shadow.

### Phase 0.5: Viability Gate

- **Objective**: Determine if Qwen2-0.5B can be made to fit and run correctly on a 4GB GPU using any technique in the catalog.
- **Deliverable**: Dense Qwen2-0.5B loads, runs a forward pass, and produces correct logits on a 4GB GPU (or equivalent fixture).
- **Gate**: Model loads without OOM. Golden vector test passes.
- **Fallback**: If no combination of quantization, paging, or sparsity makes the target model viable on 4GB, SPANDA does not ship for this target. Honest kill.

### Phase 1a: Dense WGPU Baseline

- **Objective**: WGPU matmul kernel for dense Qwen2-0.5B. No sparsity, no paging.
- **Deliverable**: `spanda-engine` runs dense inference on wgpu backend.
- **Gate**: Golden vector test passes on wgpu path. Runs on 4GB GPU fixture.
- **Fallback**: If wgpu path fails, implement CPU-only fallback and re-gate.

### Phase 1b: Block-Sparse Kernel

- **Objective**: Add 4x4 block-sparse mask parsing to the WGPU kernel.
- **Deliverable**: WGPU compute shader parses 4x4 bitmasks. Decompressed weight MSE < 1e-5 vs. raw tensors.
- **Gate**: Golden vector test passes with block-sparse weights. No regression from 1a.
- **Fallback**: Revert to dense kernel. Static sparsity is applied at conversion time but dense kernel is used.

### Phase 1c: Host Paging for Sparse Blocks

- **Objective**: Page sparse weight blocks between host RAM and GPU VRAM.
- **Deliverable**: LRU page cache in host RAM. GPU holds working set. Async copy from host to GPU on cache miss.
- **Gate**: P99 latency ≤ dense baseline +15% on golden fixture. No OOM on 4GB target.
- **Fallback**: Static hot-block preload (keep top-N blocks resident, page rest on demand).

### Phase 2: Static Preload + Lookahead-1 Prefetch

- **Objective**: Hide page-load latency behind compute.
- **Deliverable**:
  - **Static hot-block preload**: Keep most-frequently accessed blocks resident in GPU memory.
  - **Lookahead-1 prefetch**: While GPU executes layer *i*, CPU async-copies layer *i+1* blocks into pinned host staging.
- **Gate**: 80% of layer transitions have next-layer weights ready before GPU sync. P99 bound holds.
- **Fallback**: Static preload only (no prefetch). If that fails, revert to synchronous lazy loading.
- **What was deleted**: Phase 2.2 (A* trajectory prefetch). If lookahead-1 cannot hide latency, A* cannot save it.

### Phase 3: Host RAM Offloading and Double-Buffered Swap

- **Objective**: Coordinate memory between host RAM, GPU VRAM, and disk.
- **Deliverable**: Double-buffered swap chain. Async compute streams for transfer. Disk spill for cold pages.
- **Gate**:
  - P99 latency ≤ dense baseline +15%
  - Top-1 token agreement > 99% on golden dataset
  - Perplexity delta < 0.5% vs. dense
- **Fallback**: Downgrade to static preload + lookahead-1.
- **Note**: Renamed from "L3 RAM Offloading" — L3 is CPU cache (MB scale), not host RAM.

---

## 3. Bramha Sprint Plan (Reconciled)

**Status legend:**
- `[x]` = Merged, tested, benchmarked, rollback confirmed, no known regressions.
- `[ ]` = Not started or blocked.
- `[~]` = In progress.
- `[D]` = Deferred to named version.

### Sprint 1 — Stable Core
- [x] Rust-only single binary
- [x] SQLite WAL metadata core
- [x] Model registry
- [x] Tokenizer in-process
- [x] Atomic writes
- [x] WAL replay
- [x] Basic CRUD for collections, documents, chunks, sessions, models

### Sprint 2 — Fast Local Inference
- [x] CPU backend (pure Rust, >20 TPS)
- [x] SIMD optimization
- [~] wgpu backend (pending SPANDA Phase 1a integration)
- [D] Speculative decode → v0.5
- [D] Prefix KV cache → v0.5 (SPANDA integration)
- [D] Flash attention → v0.5
- [D] INT4/INT8 support → v0.5
- [x] Criterion benchmarks
- [D] Persistent GPU buffers → v0.5
- [D] Heterogeneous scheduler v1 → v0.5

### Sprint 3 — Retrieval and Evidence
- [x] IVF/HNSW/BM25
- [x] Hybrid retrieval
- [x] Evidence mapping
- [x] Citation grounding
- [x] Graph pre-filter
- [x] Multi-hop retrieval
- [x] Retrieval-conditioned planner inputs

### Sprint 4 — Database Intelligence
- [x] Memory DB (unified, not separate subsystem)
- [x] Graph DB (unified, not separate subsystem)
- [D] Semantic memory promotion → v0.5
- [x] Forgetting and consolidation jobs
- [x] Answer trace persistence
- [x] Feedback events
- [x] Reusable workflow objects
- [x] Route history persistence

### Sprint 5 — Multi-Model System
- [x] Model capability registry
- [x] Model adapters
- [x] Router mode
- [D] Pipeline mode → v0.5
- [D] Verifier mode → v0.5
- [x] Benchmark-based routing
- [x] Backend capability profiles

### Sprint 6 — Planner Engine
- [x] Planner policies
- [x] Cost model
- [x] Execution path optimizer
- [x] Exact fallback chain
- [x] Planner telemetry
- [x] Stored plan traces
- [x] Local backend target selection
- [x] Planner warm-state persistence

### Sprint 7 — Activation Views and Reuse [DEFERRED to v0.5]
- [D] Activation materialized views
- [D] Deterministic answer cache
- [D] Reusable workflow cache
- [D] Activation replay validation
- [D] Planner integration
- [D] Branch checkpoint replay

### Sprint 8 — Model Storage Efficiency (Foundation)
- [x] Storage manifest layer
- [x] Content-addressed storage (Blake3 dedup)
- [x] Multi-tier storage system (Hot/Warm/Cold)
- [x] Integration guide and documentation
- [x] End-to-end validation (BRM-S9-003)

### Sprint 9 — Storage Integration & Validation
- [x] Manifest integration into `tensor_db.rs` (BRM-S9-001)
- [x] Multi-tier routing in planner (BRM-S9-002)
- [x] End-to-end storage benchmark (BRM-S9-003) — **GATE FOR SPRINT 8 CLAIMS PASSED**
- [D] Advanced compression (SVD, columnar codec, differential) → v0.5

### Sprint 10 — SPANDA Integration + DS4 Features
- [x] Integrate `spanda-engine` v0.1 as Bramha backend
- [x] Power throttling (`--power N` via measure + sleep)
- [x] KV cache persistence (`cold/continued/evict/shutdown` lifecycle)
- [x] `--dump-logprobs` and `--trace` diagnostic flags
- [x] Frontier-based benchmarking (per-context-size CSV)
- [x] Planner can select SPANDA path or CPU fallback
- [x] SPANDA P99 bound preserved under Bramha telemetry

### Sprint 11 — Adaptive Learning [DEFERRED to v0.5]
- [D] Adapter learning pipeline
- [x] Planner learning
- [x] Reinforcement/forgetting
- [D] Memory confidence updates
- [D] Contradiction resolution
- [x] Route-quality reinforcement

### Sprint 12 — Operator UX
- [x] Dashboard
- [x] Model orchestration visibility
- [x] Evidence explorer
- [x] Planner trace viewer
- [x] Memory explorer
- [x] Graph explorer
- [x] Backend target visibility
- [x] Route path visibility

### Sprint 13 — REMOVED

> See `HYPERSCALE.md`. Sprint 13 content moved there. Do not open until v1.0 entry gate is passed.

---

## 4. Architecture Invariants

1. **Rust only** — No Python, no C++ runtimes in core engine. Build tooling may use Python, but end-user binary must not require Python.
2. **`bramha-engine` is a pure library** — No transport-layer code inside it.
3. **Single binary by default** — No mandatory sidecars or subprocesses.
4. **CPU fallback is complete** — Every core feature must work CPU-only.
5. **wgpu accelerates** — It never defines the only correct path.
6. **Remote workers scale out** — They extend the execution model, not replace it.
7. **Decompose at ingest** — Inference must not depend on raw dense weight loading in the hot path.
8. **KV growth must be bounded or spill safely.**
9. **Anything worth reusing becomes a database object.**
10. **Every optimization path must have an exact fallback.**
11. **Planner decisions must be traceable.**
12. **Retrieval and memory are part of execution planning, not bolt-ons.**
13. **No silent corruption** — Every storage path must fail loudly or degrade safely.
14. **Every benchmark claim must name the exact fixture and hardware class.**
15. **Experimental research paths must be isolated behind feature flags.**
16. **Multi-model routing must degrade safely if preferred models are unavailable.**
17. **CPU, accelerated, and remote paths must preserve equivalent semantics** unless explicitly marked experimental.
18. **Observability is mandatory** for every runtime decision that changes user-visible behavior.
19. **Storage deduplication must be transparent to inference.**
20. **Multi-tier storage must never lose data** — Promotion/demotion must be safe under crash, with WAL recovery for in-flight transfers.
21. **Storage tier classification must be deterministic from the manifest alone.**
22. **Gate Discipline** — No phase begins until the previous phase's gate is passed. If a gate fails, execute the fallback immediately.
23. **No Retries** — Retries are jitter. All fallbacks are path switches, not re-attempts.
24. **Banker Mode** — When in doubt, ship the conservative option.
25. **P99 Bound** — Latency must never exceed dense baseline +15% at the 99th percentile.
26. **Layer-serial execution while paging** — Each transformer layer is a distinct GPU dispatch with inter-layer sync. Layer-batching across multiple layers into a single command buffer is forbidden until paging is proven correct and a separate gate is passed. *(Reference: ds4 issue #384)*
27. **Conversion-time memory budget** — The converter computes `pageable_cache_budget = 0.8 * target_gpu_mem - fixed_overhead`. If the budget is impossible, conversion fails loudly.
28. **Build simplicity** — No build-time code generation. Metal kernels compiled at runtime by driver. CUDA kernels compiled at build time with `nvcc`. No `build.rs` shader generation.
29. **SPANDA scope** — One model family per release. Tensor layouts, quantization mixes, and metadata formats may change between releases to optimize for the current target.
30. **Golden vector gate** — Every SPANDA phase gates on a logit-level regression test against a trusted reference. Weight-level MSE is necessary but not sufficient.

---

## 5. Execution Invariants

1. Every non-trivial task must have an execution card before implementation starts.
2. Every risky change must define rollback before code is written.
3. Every storage change must define recovery behavior.
4. Every planner or router change must define safe fallback behavior.
5. Every feature flag must preserve baseline behavior when disabled.
6. No task may change out-of-scope modules unless explicitly approved.
7. No task may close without outputs matching the task card.
8. No task may rely on human interpretation to determine success.
9. Naive attention is test-only once flash attention is enabled for production decode.
10. No task may proceed if the previous phase's gate is unpassed.
11. Absolute Logging: Each and every function must have logs. Any function that is written or modified must output logs to trace execution flow and easily identify the root cause of crashes.

---

## 6. Success Criteria

### v0.1 — Single-Model Local Intelligence Database (Shippable)
- [ ] Qwen2-0.5B runs end-to-end on CPU and wgpu
- [ ] CRUD for collections, documents, sessions, models
- [ ] WAL replay and atomic writes proven
- [ ] Tokenizer fully in-process and model registry working
- [ ] Retrieval (IVF/HNSW/BM25) with evidence grounding
- [ ] Memory DB + graph DB functional (unified SQLite, not separate subsystems)
- [ ] Planner with exact-decode-only path
- [ ] Storage manifest + dedup integrated
- [ ] No multi-model, no SPANDA advanced features, no adaptive learning, no distributed

### v0.5 — Fast Local Engine
- [ ] Sprint 7 complete (activation views, answer cache)
- [ ] Planner uses cached-answer + speculative paths
- [ ] Storage optimizations validated with benchmarks
- [ ] SPANDA static sparse baseline integrated

### v1.0 — Local Intelligence System
- [ ] Multi-model orchestration: router + pipeline + verifier modes stable
- [ ] Planner v2: online cost model updates from real latency measurements
- [ ] Full operator UX: dashboard, evidence explorer, planner trace viewer, memory/graph explorers
- [ ] CPU-only full feature path: zero GPU dependency for any core feature
- [ ] CPU + iGPU + dGPU acceleration: heterogeneous scheduler selects optimal path per operation
- [ ] SPANDA Phase 2–3 integrated: prefetcher + host RAM offload operational
- [ ] Adapter learning pipeline: LoRA fine-tuning for task specialization
- [ ] v1.0 API freeze: backward-compatible HTTP + UDS APIs

### vNext — Hyper-Scale

**Entry Gate**: See `HYPERSCALE.md`. vNext begins only when ALL of the following are true:
- [ ] v1.0 API is frozen for 30 days with zero breaking changes
- [ ] Single-node throughput validated at ≥10 req/s sustained on 4GB GPU target
- [ ] 7-day stress test: zero memory leaks, zero data corruption, P99 latency stable
- [ ] Distributed design doc approved (not implemented — just designed)
- [ ] Team has 2+ engineers with distributed systems experience, OR single engineer has completed v1.0 solo

**If gate fails:** Extend v1.0 stabilization. Do not begin distributed work.

---

## 7. Strict AI Execution Specification

### 7.1 Purpose

Every development task must be written so an AI can execute it without guessing. A task is invalid if it is vague, mixes multiple outcomes, omits pass/fail checks, or fails to name the exact modules allowed to change.

### 7.2 Mandatory Task Card Format

**Task ID**
Unique stable identifier: `BRM-<SPRINT>-<NUMBER>` or `SPANDA-<PHASE>-<NUMBER>`.

**Title**
Short, concrete, single-outcome description.

**Objective**
One sentence describing the exact outcome to achieve.

**Scope**
Exact files, modules, crates, endpoints, or subsystems allowed to change.

**Non-Goals**
Explicitly list what this task must not change.

**Dependencies**
List prerequisite tasks, modules, fixtures, or benchmark baselines that must already exist.

**Inputs**
The starting state required for the task: existing code modules, configuration, fixtures, sample models, benchmark baselines, manifests, database state, API contracts.

**Steps**
Ordered implementation steps only. Each step must be directly verifiable. No vague wording such as "improve," "optimize," or "clean up" unless a measurable target is stated.

**Outputs**
The concrete artifacts produced: source files changed, tests added, benchmarks added, manifest/schema updates, metrics emitted, docs updated.

**Acceptance Criteria**
Binary pass/fail conditions. No human interpretation required.

**Failure Modes**
What must happen if the task cannot complete safely: fallback behavior, abort conditions, preservation of old state, logging requirements, degraded mode behavior.

**Rollback**
How to disable or revert the feature if regressions appear: feature flag, config switch, migration reversal, fallback path.

**Tests**
Required validation: unit tests, integration tests, property tests, benchmark tests, recovery tests. Manual verification only if automation is impossible.

**Regression Risks**
List specific things that could break: correctness, latency, memory, storage compatibility, semantic drift, routing drift, dashboard truthfulness.

**Notes**
Optional implementation details, invariants, or warnings.

### 7.3 Task Quality Rules

A valid task must satisfy all of the following:

1. One task = one outcome.
2. Every task must name the exact modules allowed to change.
3. Every task must define the expected output state.
4. Every task must define what counts as regression.
5. Every task must be testable without human interpretation.
6. Every task must define rollback for risky changes.
7. Every task must fit within one AI execution plan.
8. Experimental features must be isolated behind feature flags.
9. Hot-path changes must name correctness invariants.
10. Benchmark claims must include exact measurement method.

### 7.4 Forbidden Task Language

The following are invalid:

- "Improve performance"
- "Make storage better"
- "Refactor inference"
- "Add memory support"
- "Fix routing"
- "Optimize GPU path"

Replace with task cards specifying exact modules, measurable targets, benchmark or correctness gates, fallback behavior, and test plans.

### 7.5 Done Means Done

A task is complete only when:

- Implementation is finished
- Outputs are present
- All required tests pass
- Rollback path is documented
- No known regressions remain open for the task scope
- Observability is added if runtime behavior changed
- Storage/schema changes include compatibility handling if needed
- Acceptance criteria are all checked
- No out-of-scope changes remain in the diff

---

## 8. Benchmark Fixtures

### Fixture 1: Qwen2-0.5B Cold Start

- **Model**: Qwen2-0.5B (Instruct)
- **Prompt**: 2048 tokens from public-domain text (e.g., Project Gutenberg)
- **Hardware class**: 4GB VRAM GPU + NVMe SSD + 16GB system RAM
- **Metrics**:
  - `model_load_time_ms`: Baseline 500ms | Target 50ms
  - `first_token_latency_ms`: Baseline 1200ms | Target 400ms
  - `sustained_tps`: Baseline 0.42 | Target TBD

### Fixture 2: Golden Vector

- **Model**: Qwen2-0.5B
- **Prompt**: Fixed 256-token prompt (deterministic seed)
- **Reference**: HuggingFace Transformers, greedy decoding, temperature=0
- **Gate**: SPANDA logits must match reference within 1e-3 relative tolerance per token.

### Fixture 3: Long Context Stability

- **Model**: Qwen2-0.5B
- **Context**: 8192, 16384, 32768 tokens
- **Gate**: Generation completes without OOM. Output is coherent (perplexity within 5% of 2048-token baseline).

---

## 9. Storage Layout (v0.1)

```
storage/
├── meta.db         ← SQLite control plane (metadata, sessions, models, collections, memories, graph, planner state)
├── payloads/       ← Content-addressed store (Blake3 keys)
│   ├── weights/    ← model weights, quantized variants, adapters
│   ├── vectors/    ← embeddings, chunk payloads
│   ├── kv/         ← paged KV blocks + prefix cache
│   ├── cache/      ← deterministic answers, planner warm-state
│   └── sessions/   ← resumable session checkpoints (KV + rendered text + tool maps)
├── wal/            ← write-ahead logs
├── snapshots/      ← restore points
└── manifest.json   ← model, planner, routing, quantization metadata
```

**Note:** v0.1 merges Metadata DB, Vector DB, Memory DB, Model DB, Graph DB, and Cache DB into a single SQLite control plane (`meta.db`) with separate tables. The Payload Store handles content-addressed blobs. Specialized subsystems (separate Graph DB, Memory DB) deferred until v0.5 if metrics prove the unified path is insufficient.

---

## 10. Risk Register (v0.1)

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Qwen2-0.5B does not fit on 4GB GPU even with quantization | Medium | Project kill | Phase 0.5 Viability Gate; honest kill if gate fails |
| wgpu backend has driver bugs on target hardware | Medium | Sprint delay | CPU fallback is complete; ship CPU-only if needed |
| Blake3 dedup corrupts reference counting | Low | Data loss | WAL + checksum verification on every read; crash recovery tests |
| Planner selects wrong path and degrades quality | Medium | Quality loss | v0.1 planner is exact-decode-only; no path selection risk |
| SQLite WAL grows unbounded | Low | Disk full | Automated WAL checkpoints; size limits with safe degradation |
| Static sparse weights degrade model quality | Medium | Quality loss | Phase 0 gate; if agreement < 99%, ship dense-only |
| Layer-batching + paging causes silent corruption | Low | Correctness | Invariant 26: layer-serial execution while paging |
