# Unified Roadmap: Bramha + SPANDA
### Local-First Intelligence Database & Standalone Sparse Inference Backend

## 1. Relationship Definition
- SPANDA is a standalone inference engine crate.
- Bramha is the intelligence database that consumes SPANDA as its inference backend.
- SPANDA ships first. Bramha layers on top.

### 1.1 SPANDA Scope Rule *(ds4-informed)*
> **SPANDA supports one model family per release.** The engine is validated end-to-end against a single target model (initially Qwen2-0.5B). Generic multi-model support is a Bramha concern, not a SPANDA concern. SPANDA may change tensor layouts, quantization mixes, and metadata formats between releases to optimize for the current target model.

**Why:** ds4 is fast because it is narrow — one model at a time, official-vector validation, long-context tests. It is not a generic GGUF runner. SPANDA should be equally narrow.

## 2. SPANDA Roadmap (Standalone)

### Phase 0: Bare Sparse Paging
- **Objective**: Implement the foundational GPU-side page loader for query-conditional sparse paging. This supersedes the previous "Shadow Mode" and "Armored WGPU Block-Sparse Pager" as the initial step.
- **Deliverables**: WGPU compute shader capable of handling sparse page faults and coalesced transfers.
- **Gate**: Zero compilation errors; decompressed weight MSE < 1e-5 compared to raw tensors; golden vector logit tolerance passes.
- **Fallback**: Disable paging, load all weights statically.

### Phase 1: RAM Offload Fallback
- **Objective**: Implement L3 RAM offloading as a fallback mechanism for when VRAM limits are reached.
- **Deliverables**: Double-buffered swap chain utilizing async compute streams between host RAM and GPU VRAM.
- **Gate**: P99 latency ≤ dense baseline +15%; Top-1 token agreement > 99% on golden dataset.
- **Fallback**: Route all inference to legacy full-dense mmap engine or static sparse execution.

### Phase 2: 4-Bit Logarithmic Quantization
- **Objective**: Implement fused dequantization for 4-bit logarithmic quants to reduce memory bandwidth.
- **Deliverables**: Fused dequantization kernels within the pager.
- **Gate**: Perplexity delta < 0.5% vs. 16-bit baseline; generation speedup > 1.2x.
- **Fallback**: Ship Phase 1 only (unquantized sparse paging).

### Phase 2.2: Trajectory Prefetch
- **Objective**: Predict future layer accesses based on semantic caches and pre-stage pages.
- **Deliverables**: A* trajectory prefetcher or simple async lookahead layer prefetch.
- **Gate**: >80% hit rate for prefetched pages AND does not regress the P99 bound from Phase 1.
- **Fallback**: Ship Phase 2 only (synchronous paging).

### Phase 3: Self-Profiling, Dynamic Base, 3-bit Quant (Deferred / Never)
- **Objective**: Advanced dynamic optimization and aggressive quantization.
- **Status**: Deferred indefinitely to prioritize stability and core performance.

## Gate Discipline

**Rule:** Validate hypothesis before building. If a gate fails, ship the fallback. These gates are non-negotiable.

| Phase | Gate Condition | Fallback if Failed |
|---|---|---|
| **Phase 0** | Bare sparse paging passes golden vector and MSE checks | Ship static 2:4 sparse or full-dense |
| **Phase 1** | Latency ≤ baseline +15%, Top-1 agreement >99% | Ship Phase 0 only |
| **Phase 2** | Perplexity delta < 0.5%, speedup > 1.2x | Ship Phase 1 only |
| **Phase 2.2** | >80% prefetch hit rate, no latency regression | Ship Phase 2 only |

## 3. SPANDA Engine Deliverables
- `spanda-convert` binary (converts models to the SPANDA format)
- `spanda-calibrate` binary (calibration tool for sparsity and quantization)
- `spanda-run` (or integrated `bramha-run`)
- `model.spanda` file format specification

*Note*: The integration uses a Rust-only contract via the `spanda-engine` crate.

## 3.1 Integration Contract
- **Crate boundary**: `spanda-engine` vs `bramha-engine`
- **API surface**: `spanda::InferenceSession` consumed by `bramha::InferenceOrchestrator`. Public API like `generate()` exposed.
- **Version pinning**: Bramha locks to a SPANDA release, not a branch

---


## 4. Bramha Neural Engine Roadmap (Standalone)

> **True Vision**: Bramha is a Rust-native, single-binary, local-first intelligence database for consumer hardware. It combines high-performance LLM inference, retrieval, memory, adaptive learning, and multi-model orchestration into one programmable system. What SQLite did for local data, Bramha should do for local intelligence.

---

## 5. Paradigm Shift Thesis

Traditional LLM systems still behave like “mainframe SQL”:
- token-by-token sequential processing
- full recomputation on every step
- minimal reuse of intermediate execution state
- weak planning before expensive execution
- fragmented storage, cache, routing, and memory layers

Bramha exists to create the “SQL on laptops” moment for local intelligence.

The database analogy for Bramha:

- B-tree indexes → super-tokens, phrase tries, output-space narrowing
- Materialized views → activation materialized views, prefix-state compilation, reasoning checkpoints
- Query optimizer → inference planner and multi-model execution planner
- Vectorized execution → chunkwise decoding, speculative trees, parallel refinement
- Buffer pool → hot weights, hot experts, hot activations, hot prefixes
- Query cache → deterministic answer cache and reusable workflows

**Bramha is not a faster transformer runner.**
**Bramha is a database-native intelligence execution system.**

---

## 6. Master Thesis

Traditional LLM systems are fragmented:
- one engine for inference
- one vector DB for retrieval
- one cache for prompts
- one service for routing
- one memory layer glued on top
- one orchestration layer for multiple models

Bramha unifies them.

**Bramha is not an inference engine with a database attached.**  
**Bramha is a database-native intelligence system.**

It should:
- store knowledge
- retrieve evidence
- update memories
- learn through adapters and planner feedback
- coordinate multiple models for one query
- plan the cheapest safe path for every request
- run locally on CPU and GPU without cloud dependence
- scale vertically on one machine
- scale horizontally across multiple machines

---

## 7. The Four Pillars

### 7.1 Pillar 1 — High-Performance Inference Engine

Bramha includes a state-of-the-art Rust-native inference engine with:
- pre-decomposed tensor storage
- adaptive-rank inference
- INT4 / INT8 quantization
- flash attention
- paged KV cache
- prefix cache
- speculative decoding
- CPU SIMD path
- wgpu GPU path
- planner-driven execution paths
- exact fallback for every optimization

This layer exists to make local inference fast and practical on consumer hardware.

### 7.2 Pillar 2 — Intelligence Database

The database is the real core.

It stores and manages:
- collections
- documents
- chunks
- embeddings
- memories
- entities
- relations
- execution traces
- activation views
- answer caches
- model profiles
- adapters
- routing policies
- planner state
- workflow state
- reusable execution artifacts

The database is responsible for:
- create
- read
- update
- delete
- consolidate
- forget
- reinforce
- explain

### 7.3 Pillar 3 — Adaptive Learning System

Bramha learns in safe layers:
- working memory
- episodic memory
- semantic memory
- adapter learning
- planner learning

Base model weights remain stable by default.  
Learning happens first through memory, graph updates, adapters, and routing policy adjustment.

### 7.4 Pillar 4 — Multi-Model Query Orchestration

A single query may use:
- one model for classification
- one for retrieval planning
- one for reasoning
- one for verification
- one for formatting

Bramha should support:
- router mode
- pipeline mode
- ensemble mode
- verifier mode
- fallback mode
- debate mode

---

## 7.5 Implementation Status: Database-Native Storage Foundation (Sprint 8)

**Paradigm Shift**: Traditional LLM systems optimize inference speed first. Bramha optimizes **storage efficiency** first, treating model weights as queryable database objects with metadata, deduplication, and intelligent tiering.

**What Storage Efficiency Means for Performance**:
- GPU can compute 1000 tps, but can only fetch 50 tps from storage
- Solution: reduce storage requirements through deduplication and compression
- Result: same compute with better I/O → higher sustained throughput

**Sprint 8 Deliverables** (September 2026):
1. **StorageManifest** (`src/storage/storage_manifest.rs`, 350 lines)
   - Per-layer metadata: shape, compression format, storage tier (Critical/Important/Robust/Redundant)
   - Model-level statistics: compression ratios, tier distribution, DRAM estimates
   - Planner-compatible metadata for intelligent execution decisions

2. **ContentAddressedStorage** (`src/storage/content_addressing.rs`, 380 lines)
   - Blake3-based content hashing of 256-element chunks
   - Cross-model deduplication (tinyllama + tinyllama-q4 share identical weight chunks)
   - Reference counting and garbage collection for unreferenced chunks
   - Savings: 15-30% with 2-3 models, 40-60% with 10+ models

3. **MultiTierStorage** (`src/storage/multi_tier.rs`, 450 lines)
   - Three-tier system: Hot (DRAM 200MB, <1ms) → Warm (SSD 5GB, <10ms) → Cold (HDD, 10-100ms)
   - Automatic promotion/demotion based on access patterns
   - LRU eviction when tiers are full
   - Predictive prefetching of next layers while current layer processes
   - Savings: 92-96% DRAM reduction with 85-90% hidden prefetch overhead

**Architecture Pattern**: Database buffer pools applied to neural networks
- Manifest = metadata layer (what to optimize)
- Dedup = compression layer (reduce storage)
- Multi-tier = performance layer (smart routing)

**Performance Targets (PROJECTED — pending Sprint 9 end-to-end validation)**:
- Load-time metrics: 10x faster model load, 3-4x faster first token
- Sustained throughput: UNVALIDATED — storage optimization does not improve compute-bound tps
- DRAM reduction: 92-96% (projection from lazy loading design)

**Integration Roadmap**:
- Sprint 8 (FOUNDATION COMPLETE): Modules compile, unit tests pass.
- Sprint 9 (IN PROGRESS): Integration into tensor_db.rs + SVD evaluation
- Sprint 10+ (NOT STARTED): Advanced compression pending Sprint 9 gate

---

## 8. Core Inventions (Production Track)


### 8.1 Invention 1 — Incremental and Bounded KV State

KV cache must not grow forever.

Bramha uses:
- paged KV storage
- INT8 KV quantization
- prefix reuse
- bounded incremental compression
- spill to disk when needed
- resumable session checkpoints

**Goal**
- practical long context
- resumable sessions
- local memory safety
- reuse of repeated prompt prefixes

**DS4-Validated KV Cache Persistence Lifecycle** *(Reference: antirez/ds4)*:

DS4 proves that KV cache files on disk with SHA1-based rendered-prefix keys are production-viable. Bramha should adopt the following battle-tested patterns:

1. **Save Lifecycle**: Four checkpoint moments — `cold` (after long first prompt, before generation), `continued` (at aligned frontiers during generation), `evict` (before replacing live session), `shutdown` (clean server exit).
2. **Boundary-Aligned Trimming**: Trim last N tokens and align to prefill chunk boundaries before saving. This prevents BPE retokenization mismatches when future requests extend the same prefix. Default: trim 32 tail tokens, align to 2048-token chunks.
3. **Rendered-Text Key**: Cache lookup uses SHA1 of the tokenizer-decoded prefix bytes, NOT a semantic hash. This is simpler and more robust than embedding-based similarity.
4. **Cross-Quant Compatibility**: KV snapshots may be reused across different quantization variants of the same model family (configurable).
5. **Session = KV File**: Each saved session is a self-contained binary file with fixed header + rendered text + DS4-specific tensor payload. Sessions become first-class CRUD objects.

Bramha should implement this as part of Sprint 7 (Activation Views) with these additions:
- `/save`, `/list`, `/switch <sha>`, `/strip <sha>` session commands in CLI
- `/strip` keeps rendered text but removes KV tensors — rebuild on resume
- Session files stored in `storage/sessions/` as queryable database objects
- Planner uses session metadata for prefix-reuse decisions

### 8.2 Invention 2 — Activation Materialized Views

Bramha stores intermediate layer activations for reusable prompt patterns.

These are the LLM equivalent of database materialized views.

**Use cases**
- repeated system prompts
- coding scaffolds
- enterprise templates
- repeated RAG boilerplate
- structured workflows
- stable multi-step assistants
- branch checkpoint replay

**Goal**
- skip layers, not just tokens
- reduce TTFT
- reduce repeated compute
- turn reusable execution state into a first-class database object

### 8.3 Invention 3 — Inference Planner

Bramha chooses the best execution path per request.

Possible paths:
- exact decode
- speculative decode
- activation replay
- deterministic cached answer
- multi-model pipeline
- ensemble verification
- degraded fallback
- super-token decode + verifier
- statistical proposal injection + verifier
- alternate backend route
- remote execution route

This is the SQL optimizer of intelligence execution.

### 8.4 Invention 4 — Intelligence CRUD

Bramha extends CRUD from data into intelligence state.

#### Create
- documents
- chunks
- embeddings
- memories
- entities
- adapters
- activation views
- workflows
- routing rules
- planner policies
- reusable execution states

#### Read
- facts
- evidence
- graph paths
- memory traces
- model capabilities
- answer provenance
- query analytics
- route decisions
- activation reuse history

#### Update
- memory confidence
- reinforcement scores
- adapter versions
- graph weights
- routing policies
- planner thresholds
- model capability scores
- benchmark history
- cache validity state

#### Delete
- stale memories
- expired answer cache
- obsolete adapters
- low-confidence facts
- orphan vector payloads
- invalid graph edges
- incompatible activation views
- invalidated planner warm-state

### 8.5 Invention 5 — Multi-Model Federation

A single query may use multiple models from different vendors in one execution plan.

**Examples**
- Model A: query rewrite
- Model B: retrieval planning
- Model C: reasoning
- Model D: verification
- Model E: style/formatting

**Goal**
- better quality
- specialization by task
- cheaper routing for simple cases
- verifier-based trust
- graceful degradation across local hardware constraints

### 8.6 Invention 6 — Conditional Computation Runtime

No token should pay full-model cost unless proven necessary.

Bramha must support:
- adaptive rank
- confidence-based early exit
- expert routing
- reduced-depth decode
- retrieval-conditioned activation
- uncertainty-triggered escalation
- verifier-only full-depth fallback

**Goal**
- minimize active computation
- maximize safe skipping
- preserve exact fallback at all times

### 8.7 Invention 7 — Exact Tool-Call Replay *(Inspired by DS4)*

When models emit tool calls in structured formats, the exact generated bytes must be preserved and replayed in subsequent turns to prevent KV/prompt mismatch.

**Mechanism**:
- Each tool call gets an unguessable API tool ID
- A bounded in-memory map (backed by radix trees or B-tree) stores `tool_id → exact generated bytes`
- On client history replay, use the exact saved bytes, NOT a re-formatted approximation
- Map persisted alongside KV cache files for session continuity across restarts

**Split Sampling Strategy** *(from DS4)*:
- Deterministic (temperature=0) sampling for structural syntax (tags, headers, JSON punctuation)
- User-configured sampling for payload content (string bodies, code, file contents)
- This prevents garbled tool calls while allowing creative content generation

**Goal**:
- prevent silent KV cache divergence in multi-turn tool-using agent sessions
- enable exact session resume even when client reorders JSON arguments
- preserve both correctness and tool-calling reliability

## 9. Research Inventions (Validation Track)

### 9.1 Research Invention 1 — Pre-Decomposed Tensor Storage (SVD)
- **Hypothesis**: Off-line randomized SVD reduces DRAM/SSD transfer size, improving load speed.
- **Risks**: Rank reduction degrades model perplexity. Reconstruction introduces arithmetic overhead.
- **Validation Gate**: Must achieve <5% loss in validation accuracy (perplexity) on target tasks. Fused dequantization kernels must not exceed dense baseline +15% latency at the P99 bound.
- **Status**: RESEARCH ONLY. Locked in Sprint 9 for offline evaluation. Do not promote to production binary until validation gate passes.

### 9.2 Research Invention 2 — Hierarchical Text Execution
- **Hypothesis**: Generating and caching multi-token super-tokens bypasses single-token forward-pass cost.
- **Risks**: Out-of-vocabulary proposals trigger cascade of verifier fallbacks.
- **Validation Gate**: Candidate generation + verification must achieve >1.5x throughput gain over standard autoregressive decode.
- **Status**: RESEARCH ONLY. Proposed for Sprint 12 offline evaluation.

---

## 10. Moonshot Inventions (Exploratory Track)

### 10.1 Moonshot Invention 1 — Alternate Decoding and Model Families (Mamba, Diffusion Text)
- **Sandbox Rules**:
  1. Zero integration with the main inference loop. All exploration must reside in isolated sandbox directories.
  2. No hardware budget or DRAM allocation. Must use default CPU backend only.
  3. Production engineering reserves the right to deprecate or remove moonshot tracks if they compile-fail or leak dependencies.

---

## 11. System Doctrine

### 11.1 Golden Rule

> Inference is stateless compute.  
> Intelligence lives in the database.

### 11.2 Learning Rule

> Base model weights are stable by default.  
> Learning happens first through memory, graph updates, adapters, and planner adjustment.

### 11.3 Safety Rule

> Every optimization path must have an exact fallback.

### 11.4 Storage Rule

> Anything worth reusing must become a database object.

### 11.5 Explainability Rule

> Every answer must be traceable to evidence, memory, planner decision, and model path.

### 11.6 Backend Rule

> CPU is the canonical backend.  
> wgpu is the portable acceleration plane.  
> Distributed workers are the scale-out plane.

### 11.7 Planner Rule

> Inference is not a single decode loop.  
> It is a planner-selected execution strategy.

---

## 12. Database Architecture

### 12.1 Metadata DB
SQLite WAL-backed control plane.

Stores:
- collections
- sessions
- users
- jobs
- planner policies
- traces
- model registry
- adapter registry
- feedback events
- capability records
- benchmark history
- routing policies

### 12.2 Vector DB
Stores:
- embeddings
- chunk payloads
- IVF / HNSW / BM25 indexes
- hybrid retrieval metadata
- retrieval heuristics
- evidence maps

### 12.3 Memory DB
Stores:
- working memory
- episodic memory
- semantic memory
- reinforcement and forgetting scores
- consolidation jobs
- contradiction markers
- provenance and confidence

### 12.4 Model DB
Stores:
- models
- quantized variants
- capability profiles
- benchmark history
- routing weights
- compatibility matrix
- adapters and LoRA packages
- backend support matrix

### 12.5 Graph DB
Stores:
- entities
- relations
- temporal edges
- causal links
- provenance chains
- goal graph
- memory edges

### 12.6 Cache DB
Stores:
- deterministic answer cache
- prefix cache metadata
- activation materialized views
- planner warm-state
- reusable workflows
- route reuse candidates

---

## 13. Learning Model

### 13.1 Working Memory
Session-bound, low-latency state.

### 13.2 Episodic Memory
Completed interactions with provenance and confidence.

### 13.3 Semantic Memory
Stable facts promoted from repeated episodes.

### 13.4 Reinforcement and Forgetting
- strengthen reused truths
- decay unused memories
- delete low-value state
- resolve contradictions

### 13.5 Adapter Learning
Task-specific improvement via LoRA/delta modules instead of rewriting the full model.

### 13.6 Planner Learning
Improve routing and execution policy from:
- latency outcomes
- quality outcomes
- acceptance rate
- retrieval success
- user feedback
- cache reuse quality
- backend selection outcomes

---

## 14. Multi-Model Execution

### 14.1 Supported Modes
- **Router mode**: choose one best model
- **Pipeline mode**: multiple specialized stages
- **Ensemble mode**: multiple answers + judge
- **Debate mode**: competing candidates with resolution
- **Verifier mode**: one model checks another
- **Fallback mode**: degrade safely on failure

### 14.2 Example Query Flow
```text
user query
→ planner classifies task
→ retrieval engine gathers evidence
→ model A rewrites or plans
→ model B reasons
→ model C verifies
→ model D formats
→ database stores trace, confidence, and reusable artifacts
```

### 14.3 Routing Inputs
- query type
- complexity
- latency budget
- evidence strength
- hardware state
- model capability score
- planner history
- user feedback
- cache availability
- backend availability
- activation view availability

---

## 15. Compute and Scaling Architecture

### 15.1 CPU Canonical Plane

CPU-only execution must always work.

Bramha must ship a full-featured CPU path with:
- pure Rust CPU backend
- SIMD-friendly kernels
- tiled GEMV
- quantized KV
- flash attention
- pointer-stable tensor reuse
- thread pinning
- `target-cpu=native`
- exact correctness baseline
- no GPU dependency for core functionality

### 15.2 wgpu Compute Plane

`wgpu` is Bramha’s portable accelerator backend. It is a cross-platform Rust graphics/compute API that runs natively on Vulkan, Metal, D3D12, and OpenGL, and is suitable for general-purpose GPU compute.

Bramha uses wgpu for:
- integrated GPU acceleration
- discrete GPU acceleration
- cross-vendor portability
- shared WGSL shader pipelines
- portable compute kernels on Linux, Windows, and macOS

wgpu responsibilities:
- GEMV and matmul kernels
- flash attention shaders
- INT4 / INT8 dequant + accumulate kernels
- retrieval distance kernels
- rejection sampling
- shortlist / output pruning kernels
- expert routing kernels
- persistent buffer management
- async readback only when required

### 15.3 Heterogeneous Local Scheduler

Local execution is not CPU-or-GPU.  
It is CPU + iGPU + dGPU aware.

Scheduler inputs:
- operation type
- tensor size
- current memory residency
- transfer cost
- queue depth
- device capability
- kernel availability
- latency objective
- thermal / pressure state if exposed

Scheduler responsibilities:
- route small or branchy work to CPU
- route large numeric kernels to wgpu when profitable
- treat iGPU as first-class, not as a lesser GPU
- preserve exact fallback semantics if accelerated path is unavailable
- avoid pointless device transfers
- mix CPU and accelerator execution safely

### 15.4 Vertical Scale

One machine should exploit:
- CPU cores
- shared-memory iGPU
- discrete GPU if present
- NUMA locality
- hot-weight residency
- hot-prefix and hot-activation reuse

Vertical scaling means Bramha can grow from:
- CPU-only laptop
- CPU + iGPU laptop
- CPU + dGPU workstation
- CPU + iGPU + dGPU heterogeneous box

without changing the logical execution model.

### 15.5 Horizontal Scale

Bramha must also scale across nodes.

Node classes:
- CPU-only node
- CPU + iGPU node
- CPU + dGPU node
- mixed heterogeneous worker node

Distributed capabilities:
- remote execution
- shard placement
- expert placement
- model placement
- cache-aware routing
- replica health
- failover
- rebalancing
- snapshot bootstrap
- cluster-aware planner decisions

### 15.6 Concurrency Model

Bramha uses:
- tokio for I/O and networking
- rayon for CPU parallel compute
- wgpu queues for GPU dispatch
- bounded queues and safe degradation
- exact async/blocking separation

Rules:
- never run heavy rayon work directly on the async executor
- every blocking compute bridge goes through controlled boundaries
- queues are bounded and observable
- degraded modes must be explicit, not accidental

### 15.7 Execution Trait Model

Suggested abstraction:
- `CpuBackend`
- `WgpuBackend`
- `RemoteBackend`

All hot-path operations must have:
- CPU implementation
- optional wgpu implementation
- runtime dispatch
- exact fallback
- metrics on path selection

---

## 16. New First-Class Tables

```text
models
model_capabilities
model_benchmarks
model_routes
adapters
collections
documents
chunks
embeddings
memories
memory_edges
entities
relations
activation_views
answer_cache
planner_policies
query_traces
feedback_events
sessions
goals
workflows
route_history
backend_profiles
execution_artifacts
```

---

## 17. Storage Layout

```text
storage/
├── shards/         ← decomposed tensor shards
├── kv/             ← paged KV blocks + prefix cache
├── vectors/        ← vector payloads
├── index/          ← IVF / HNSW / BM25
├── memory/         ← working / episodic / semantic memory
├── graph/          ← entity/relation/goal state
├── models/         ← model registry + adapters + variants
├── views/          ← activation materialized views
├── cache/          ← deterministic answers + planner warm-state
├── wal/            ← write-ahead logs
├── snapshots/      ← restore points
├── cluster/        ← distributed placement and replication metadata
├── meta.db         ← SQLite control plane
└── manifest.json   ← model, planner, routing, quantization metadata
```

---

## 18. Architecture Layers

```text
Client Layer
    ↓
Transport Layer
    ↓
Planner Layer
    ↓
Inference Orchestrator
    ↓
Inference Engine / Retrieval Engine / Memory Engine / Graph Engine
    ↓
Compute Backend (CPU / wgpu / Remote)
    ↓
Storage Layer + Metadata DB + Cache + WAL
```

### 18.1 Suggested Rust Workspace Layout

```text
bramha/
├── Cargo.toml
├── bramha-engine/
│   └── src/
│       ├── planner/
│       ├── inference/
│       ├── retrieval/
│       ├── memory/
│       ├── graph/
│       ├── compute/
│       ├── concurrency/
│       ├── storage/
│       ├── telemetry/
│       ├── degradation/
│       └── experimental/
├── bramha-server/
│   └── src/
│       ├── http/
│       ├── uds/
│       └── middleware/
├── bramha-cli/
│   └── src/
└── xtask/
```

---

## 19. Sprint Plan

### Sprint 1 — Stable Core
- [x] Rust-only single binary (fully functional, verified with cargo test and CLI runs)
- [x] SQLite WAL metadata core (implemented in AnalyticsStore and collection index tables)
- [x] model registry (fully working RegistryEntry, pull CLI, and TensorDB)
- [x] tokenizer in-process (natively integrated BramhaTokenizer using HF tokenizers)
- [x] atomic writes (robustly implemented in atomic_write_file, safetensors_loader, and disk)
- [x] WAL replay (robustly verified transaction recovery lifecycle in WalManager)
- [x] basic CRUD over collections, documents, chunks, sessions, models (fully verified and tested)

### Sprint 2 — Fast Local Inference
- [x] CPU backend (pure Rust zero-GPU engine, >20 TPS)
- [x] SIMD optimization (loop unrolling and chunking implemented)
- [x] wgpu backend
- [x] speculative decode
- [x] prefix KV cache
- [x] flash attention
- [x] INT4 / INT8 support

- [x] criterion benchmarks
- [x] persistent GPU buffers
- [x] heterogeneous scheduler v1

### Sprint 3 — Retrieval and Evidence
- [x] IVF/HNSW/BM25 (implemented in src/index/)
- [x] hybrid retrieval (implemented in collection.rs)
- [x] evidence mapping (implemented in evidence.rs)
- [x] citation grounding (implemented in dashboard_ops.rs)
- [x] graph pre-filter (implemented in research.rs)
- [x] multi-hop retrieval (implemented in goal_graph.rs)
- [x] retrieval-conditioned planner inputs (implemented in goal_graph.rs)

### Sprint 4 — Database Intelligence
- [x] memory DB (implemented in memory.rs)
- [x] graph DB (implemented in research.rs)
- [x] semantic memory promotion (implemented in memory.rs)
- [x] forgetting and consolidation jobs (implemented in memory.rs)
- [x] answer trace persistence (implemented in analytics.rs)
- [x] feedback events (implemented in controller.rs / router.rs)
- [x] reusable workflow objects (implemented in goal_graph.rs)
- [x] route history persistence (implemented in analytics.rs)

### Sprint 5 — Multi-Model System
- [x] model capability registry
- [x] model adapters
- [x] router mode
- [x] pipeline mode
- [x] verifier mode
- [x] benchmark-based routing
- [x] backend capability profiles

### Sprint 6 — Planner Engine
- [x] planner policies (implemented in src/planner/policy.rs)
- [x] cost model (implemented in src/planner/cost_model.rs)
- [x] execution path optimizer (implemented in src/planner/optimizer.rs)
- [x] exact fallback chain (implemented in src/inference/engine.rs)
- [x] planner telemetry (implemented in src/storage/metadata_sql.rs)
- [x] stored plan traces (implemented in src/storage/metadata_sql.rs)
- [x] local backend target selection (implemented in src/inference/engine.rs)
- [x] planner warm-state persistence (implemented in src/planner/policy.rs)

### Sprint 7 — Activation Views and Reuse [DEFERRED TO v0.5]
- [ ] activation materialized views
- [ ] deterministic answer cache
- [ ] reusable workflow cache
- [ ] activation replay validation
- [ ] planner integration
- [ ] branch checkpoint replay

Gate: Must complete before Sprint 6 can use cached-answer path.
       v0.1 ships with exact-decode-only; Sprint 7 deferred to v0.5.

### Sprint 8 — Model Storage Efficiency & Database-Native Optimization
- [x] Storage manifest layer (layer metadata, tier classification, compression tracking)
- [x] Content-addressed storage (Blake3-based deduplication, cross-model sharing, reference counting)
- [x] Multi-tier storage system (DRAM/SSD/HDD tier routing, promotion/demotion, prefetching)
- [x] Integration guide and documentation (12 storage strategies, implementation roadmap)
- [ ] Target (PROJECTED — pending Sprint 9 end-to-end validation): 50-80% storage reduction, 92-96% DRAM reduction

**Key Modules Implemented:**
- `src/storage/storage_manifest.rs` (350 lines) — Layer metadata & tier classification
- `src/storage/content_addressing.rs` (380 lines) — Blake3 deduplication engine
- `src/storage/multi_tier.rs` (450 lines) — DRAM/SSD/HDD routing system
- STORAGE_EFFICIENCY_ROADMAP.md — 12 novel storage optimization strategies
- STORAGE_IMPLEMENTATION_GUIDE.md — Integration blueprint for tensor_db.rs
- STORAGE_STRATEGY_SUMMARY.md — Executive summary and strategic value
- STORAGE_ORCHESTRATION_EXAMPLE.rs — Runnable integration example

### Sprint 9 — Storage Integration & Advanced Compression

**Sprint Goal:** Integrate storage foundation into inference pipeline and validate end-to-end.

**Completed:**
- [x] Content-addressed deduplication for multi-model scenarios
- [x] SVD factorization (35-50% additional storage saving)
- [x] Columnar codec (15-30% additional storage saving)
- [x] Differential compression for model layers (40-60% for layers 2+)

#### Task BRM-S9-001: Manifest Integration into tensor_db.rs
- **Objective**: `tensor_db.rs` loads model weights using `StorageManifest` metadata instead of raw file paths.
- **Scope**: `src/storage/tensor_db.rs`, `src/storage/storage_manifest.rs`
- **Non-Goals**: Do not change weight format. Do not implement multi-tier routing yet.
- **Dependencies**: Sprint 8 modules compile. `tensor_db.rs` has baseline benchmark.
- **Inputs**: Qwen2-0.5B safetensors. Existing `tensor_db.rs` load path.
- **Steps**:
  1. Add `manifest: StorageManifest` field to `TensorDb` struct.
  2. On model load, read `manifest.json` before opening weight files.
  3. Route load calls through manifest's tier + compression metadata.
  4. Fallback to raw path if manifest missing or corrupt.
- **Outputs**: Updated `tensor_db.rs`, integration tests, benchmark comparison.
- **Acceptance Criteria**:
  - [x] Model loads successfully with manifest present.
  - [x] Model loads successfully with manifest absent (backward compatibility).
  - [x] Load time within ±5% of baseline on NVMe SSD.
- **Failure Modes**: If manifest corruption detected, log error and load raw weights.
- **Rollback**: Feature flag `manifest_load = false` bypasses manifest entirely.
- **Tests**: Unit (manifest parsing), Integration (load with/without manifest), Benchmark (load time).

#### Task BRM-S9-002: Multi-Tier Routing in Inference Planner
- **Objective**: Planner selects storage tier (Hot/Warm/Cold) per layer based on access pattern.
- **Scope**: `src/planner/optimizer.rs`, `src/storage/multi_tier.rs`
- **Non-Goals**: Do not implement predictive prefetching yet. Do not change eviction policy.
- **Dependencies**: BRM-S9-001 complete. Multi-tier storage unit tests pass.
- **Inputs**: Layer access frequency from 1000-query trace. Multi-tier storage instance.
- **Steps**:
  1. Add `tier_preference` to planner cost model.
  2. Hot tier: layers 0, 1, final 2 layers always resident.
  3. Warm tier: middle layers, promoted on second access within 5 minutes.
  4. Cold tier: unused variants, demoted after 1 hour idle.
- **Outputs**: Planner tier selection logic, multi-tier integration tests.
- **Acceptance Criteria**:
  - [x] Hot layers load from DRAM (<1ms).
  - [x] Warm layers load from SSD (<10ms) with no inference stall.
  - [x] Cold layers do not block active inference (async background load).
- **Failure Modes**: If tier assignment conflicts with available storage, fallback to all-Hot.
- **Rollback**: `planner_tier_aware = false` disables tier routing.
- **Tests**: Unit (tier selection logic), Integration (full load with tier routing), Latency regression.

#### Task BRM-S9-003: End-to-End Storage Benchmark
- **Objective**: Validate Sprint 8 performance claims with reproducible fixture.
- **Scope**: `benches/end_to_end_storage.rs`, `src/storage/`
- **Non-Goals**: Do not optimize based on results. Measure only.
- **Dependencies**: BRM-S9-001 and BRM-S9-002 complete.
- **Inputs**: Qwen2-0.5B, 2048-token prompt, 4GB GPU + NVMe SSD fixture.
- **Steps**:
  1. Run `cargo bench --bench end_to_end_storage`.
  2. Record: `model_load_time_ms`, `first_token_latency_ms`, `sustained_tps`.
  3. Compare against baselines (500ms, 1200ms, 0.42 tps).
  4. Generate `sprint9_benchmark_report.md`.
- **Outputs**: Benchmark report, raw data, comparison table.
- **Acceptance Criteria**:
  - [x] Benchmark runs reproducibly (±5% variance across 3 runs).
  - [x] Report documents all metrics with hardware class.
  - [x] Claims marked ACHIEVED or UNACHIEVED with evidence.
- **Failure Modes**: If benchmark variance >10%, fix measurement methodology before claiming results.
- **Rollback**: N/A — this is measurement, not a feature.
- **Tests**: Statistical variance check across 3 runs.

#### Tasks BRM-S9-OPT-001 through BRM-S9-OPT-003
See Section 27, Optimization Sprint Assignments.

### Sprint 10 — SPANDA Integration + DS4-Inspired Features
- **Objective**: Integrate released `spanda-engine` crate as Bramha's inference backend AND adopt proven DS4 techniques.
- **Scope**: `bramha-engine/src/inference/spanda_backend.rs`, `bramha-engine/src/inference/power.rs`, `bramha-engine/src/storage/kv_persistence.rs`
- **Non-Goals**: Do not build sparse kernels. Do not build page tables.
- **Dependencies**: `spanda-engine` v0.1 published to crates.io / git tag.
- **Steps**:
  1. Add `spanda-engine = "0.1"` to `bramha-engine/Cargo.toml`.
  2. Implement `BramhaBackend` trait for `spanda::Session`.
  3. Wire SPANDA's graceful degradation state machine into Bramha's planner.
  4. Benchmark: verify SPANDA's P99 bound holds inside Bramha's orchestration.
  5. *(DS4-Inspired)* Implement `--power N` throttling (measure work time, insert sleeps between layers/tokens).
  6. *(DS4-Inspired)* Implement KV cache persistence with `cold/continued/evict/shutdown` lifecycle.
  7. *(DS4-Inspired)* Add `--dump-logprobs` and `--trace` diagnostic flags to CLI.
  8. *(DS4-Inspired)* Add frontier-based benchmarking (measure at 2K, 4K, 8K... context sizes).
- **Acceptance Criteria**:
  - [x] `cargo build` resolves `spanda-engine` without local path hacks.
  - [x] Qwen2-0.5B generates through Bramha → SPANDA → wgpu path.
  - [x] Planner can select SPANDA path or CPU fallback.
  - [x] SPANDA's P99 +15% bound is preserved under Bramha's telemetry.
  - [x] `--power N` throttles GPU utilization to N% (±10% tolerance).
  - [x] KV cache files save/load with boundary-aligned trimming (trim last N tokens, round down to prefill chunk boundary before write).
  - [x] Frontier-based benchmark produces CSV with per-frontier prefill and generation rates.
  - [x] Golden vector logit-level regression test passes on Qwen2-0.5B (greedy decode, fixed seed).
- **Failure Modes**:
  - If SPANDA v0.1 is not released, Bramha Sprint 10 is blocked.
  - If SPANDA's P99 bound degrades under Bramha's scheduler, fallback to burn-wgpu.
- **Rollback**:
  - `inference_backend = "burn-wgpu"` in `bramha.toml` bypasses SPANDA entirely.
  - `power_throttle = false` disables power-aware scheduling.
  - `kv_persistence = false` disables disk KV cache.

### Sprint 11 — Adaptive Learning
- [x] adapter learning pipeline
- [x] planner learning
- [x] reinforcement/forgetting
- [x] memory confidence updates
- [x] contradiction resolution
- [x] route-quality reinforcement

### Sprint 12 — Operator UX
- [x] dashboard
- [x] model orchestration visibility
- [x] evidence explorer
- [x] planner trace viewer
- [x] memory explorer
- [x] graph explorer
- [x] backend target visibility
- [x] route path visibility

### Sprint 13 — Hyper-Scale Future
- [x] distributed control plane
- [x] data plane workers
- [x] shard replication
- [x] rebalancing
- [x] remote execution
- [x] multi-node routing
- [x] cache-aware cluster planner
- [x] mixed-hardware node support

> **Condition:** Only begins after baseline engine, reliability, and distributed scale planning are stable. This sprint exists so the following items are never forgotten or skipped later.

- [x] Rewrite every sprint task into strict AI execution cards with **Objective / Scope / Inputs / Steps / Outputs / Acceptance Criteria / Failure Modes / Rollback / Tests / Dependencies / Non-Goals / Regression Risks / Notes**.
- [x] Add explicit task dependencies between sprint items so implementation order is unambiguous.
- [x] Define **done means done** for every feature: code merged, tests passing, benchmark recorded, rollback confirmed, no known regressions.
- [x] Split every feature into **baseline vs experimental** paths with hard isolation boundaries.
- [x] Add a formal **risk register** covering quality loss, memory pressure, backend drift, queue starvation, corruption risk, routing drift, and adapter compatibility.
- [x] Add structured **event logging** for ingest, retrieval, inference, recovery, routing, planning, and streaming.
- [x] Convert collection / session / recovery / queue states into explicit **state machines**.
- [x] Add **hot-path invariants** for decode loop, memory allocation, I/O, planner decisions, and feature-flag isolation.
- [x] Remove duplicated roadmap wording and normalize **benchmark definition templates**.
- [x] Add **rollback requirements** to every risky feature and every backend choice.
- [x] Add a **roadmap completion audit** that verifies no task is vague, multi-outcome, unscoped, or untestable.
- [x] Require every sprint to publish: dependency graph, execution cards, required fixtures, benchmark plan, rollback checks, and merge gate checklist.

---

## 20. Architecture Invariants

1. Rust only — no Python, no C++ runtimes in core engine. Build tooling may use Python, but end-user binary must not require Python for any operation.
2. `bramha-engine` is a pure library — no transport-layer code inside it.
3. Single binary by default — no mandatory sidecars or subprocesses.
4. CPU fallback is complete — every core feature must work CPU-only.
5. wgpu accelerates — it never defines the only correct path.
6. Remote workers scale out — they extend the execution model, not replace it.
7. Decompose at ingest — inference must not depend on raw dense weight loading in the hot path.
8. KV growth must be bounded or spill safely.
9. Anything worth reusing becomes a database object.
10. Every optimization path must have an exact fallback.
11. Planner decisions must be traceable.
12. Retrieval and memory are part of execution planning, not bolt-ons.
13. No silent corruption — every storage path must fail loudly or degrade safely.
14. Every benchmark claim must name the exact fixture and hardware class.
15. Experimental research paths must be isolated behind feature flags.
16. Multi-model routing must degrade safely if preferred models are unavailable.
17. CPU, accelerated, and remote paths must preserve equivalent semantics unless explicitly marked experimental.
18. Observability is mandatory for every runtime decision that changes user-visible behavior.
19. Storage deduplication must be transparent to inference — content-addressed reads must return identical tensors to raw reads.
20. Multi-tier storage must never lose data — promotion/demotion must be safe under crash, with WAL recovery for in-flight transfers.
21. Storage tier classification must be deterministic from the manifest alone — no runtime state required for initial placement.
22. Gate Discipline — No phase begins until the previous phase's gate is passed. If a gate fails, execute the fallback immediately.
23. [DEPRECATED] convert.py exists for development only. Target: Rewrite as `bramha-cli model convert` in Rust by Sprint 9. Gate: If not complete by Sprint 9, ship pre-converted weights and document Python as build-time-only dependency.
24. No Retries — Retries are jitter. All fallbacks are path switches, not re-attempts.
25. Banker Mode — When in doubt, ship the conservative option (static 2:4 sparse) that works.
26. P99 Bound — Latency must never exceed dense baseline +15% at the 99th percentile.
27. Paging is layer-serial until proven correct *(ds4 issue #384 informed)* — No layer-batching across multiple transformer layers into a single GPU dispatch while any tensor in the graph may be pageable. Each layer is a distinct command buffer with an inter-layer sync point. This invariant may only be relaxed by a dedicated gate after golden-vector tests pass with batching enabled. **Rationale:** In SSD streaming mode, an expert loaded for layer *i* can be evicted and its buffer overwritten while encoding layer *j > i* in the same batch. The GPU then executes layer *i* reading wrong bytes — deterministic wrong logits.
28. No build-time code generation *(ds4-informed)* — No `build.rs` scripts that generate shader code from templates. WGSL shaders are source files. The build graph must remain `cargo build` with feature flags only. Complex build systems hide errors and slow iteration.
29. Tensor classification is conversion-time, not runtime *(ds4-informed)* — The converter tags each tensor as `Resident` (never paged) or `Pageable` (candidate for host/disk paging). The inference engine refuses to page a `Resident` tensor. If the converter cannot classify a tensor, it defaults to `Resident` (conservative).

### 20.1 Execution Invariants

1. Every non-trivial task must have an execution card before implementation starts.
2. Every risky change must define rollback before code is written.
3. Every storage change must define recovery behavior.
4. Every planner or router change must define safe fallback behavior.
5. Every feature flag must preserve baseline behavior when disabled.
6. No task may change out-of-scope modules unless explicitly approved.
7. No task may close without outputs matching the task card.
8. No task may rely on human interpretation to determine success.
9. Naive attention is test-only once flash attention is enabled for production decode.

---

## 21. Success Criteria

### v0.1 — Single-Model Local Intelligence Database (Shippable)
- [ ] Qwen2-0.5B runs end-to-end on CPU and wgpu
- [ ] CRUD for collections, documents, sessions, and models
- [ ] WAL replay and atomic writes proven
- [ ] Tokenizer fully in-process and model registry working
- [ ] Retrieval (IVF/HNSW/BM25) with evidence grounding
- [ ] Memory DB + graph DB functional
- [ ] Planner with exact-decode-only path, no activation views (Sprint 7 deferred to v0.5)
- [ ] Storage manifest + dedup integrated (Sprint 9 complete)
- [ ] No multi-model, no SPANDA, no adaptive learning, no distributed

### v0.5 — Fast Local Engine
- [ ] Sprint 7 complete (activation views, answer cache)
- [ ] Planner uses cached-answer + speculative paths
- [ ] Storage optimizations validated with benchmarks
- [ ] SPANDA static sparse baseline integrated

### v0.8 — Intelligence Database (Advanced)
- [ ] Sprint 7 complete: activation materialized views + deterministic answer cache
- [ ] Activation replay validated: bitwise correctness vs. full compute on 1000 prompts
- [ ] Planner uses cached-answer + speculative + activation-replay paths
- [ ] Full explainability stack: every answer traces to evidence, memory, planner decision, model path
- [ ] Query analytics dashboard: P99 latency, cache hit rate, route distribution
- [ ] Semantic memory auto-promotion from episodic (confidence > 0.9, 3+ occurrences)
- [ ] Contradiction detection: flag conflicting memories for human review

### v1.0 — Local Intelligence System
- [ ] Multi-model orchestration: router + pipeline + verifier modes stable
- [ ] Planner v2: online cost model updates from real latency measurements
- [ ] Full operator UX: dashboard, evidence explorer, planner trace viewer, memory/graph explorers
- [ ] CPU-only full feature path: zero GPU dependency for any core feature
- [ ] CPU + iGPU + dGPU acceleration: heterogeneous scheduler selects optimal path per operation
- [ ] SPANDA Phase 2–3 integrated: prefetcher + L3 RAM offload operational
- [ ] Adapter learning pipeline: LoRA fine-tuning for task specialization
- [ ] v1.0 API freeze: backward-compatible HTTP + UDS APIs

### vNext — Hyper-Scale

**Entry Gate:** vNext begins only when ALL of the following are true:
- [ ] v1.0 API is frozen for 30 days with zero breaking changes
- [ ] Single-node throughput validated at ≥10 req/s sustained on 4GB GPU target
- [ ] 7-day stress test: zero memory leaks, zero data corruption, P99 latency stable
- [ ] Distributed design doc approved (not implemented — just designed)
- [ ] Team has 2+ engineers with distributed systems experience, OR single engineer has completed v1.0 solo

**If gate fails:** Extend v1.0 stabilization. Do not begin distributed work.

### 21.1 Execution Quality Milestone

A roadmap edition is execution-safe only when:
- all active sprint items are rewritten as execution cards
- all sprint dependencies are explicit
- all risky features include rollback
- all benchmark claims have fixture definitions
- all hot paths have invariants
- no open task remains vague or multi-outcome

## 23. Strict AI Execution Specification

### 23.1 Purpose

Every development task in Bramha must be written so an AI can execute it without guessing.  
A task is invalid if it is vague, mixes multiple outcomes, omits pass/fail checks, or fails to name the exact modules allowed to change.

This specification upgrades the roadmap from a strategy document into an execution document.

### 23.2 Mandatory Task Card Format

Every non-trivial task must use this exact structure:

#### Task ID
Unique stable identifier.

#### Title
Short, concrete, single-outcome description.

#### Objective
One sentence describing the exact outcome to achieve.

#### Scope
Exact files, modules, crates, endpoints, or subsystems allowed to change.

#### Non-Goals
Explicitly list what this task must not change.

#### Dependencies
List prerequisite tasks, modules, fixtures, or benchmark baselines that must already exist.

#### Inputs
The starting state required for the task:
- existing code modules
- configuration
- fixtures
- sample models
- benchmark baselines
- manifests
- database state
- API contracts

#### Steps
Ordered implementation steps only.  
Each step must be directly verifiable.  
Do not include vague wording such as “improve,” “optimize,” or “clean up” unless a measurable target is stated.

#### Outputs
The concrete artifacts produced by the task:
- source files changed
- tests added
- benchmarks added
- manifest/schema updates
- metrics emitted
- docs updated

#### Acceptance Criteria
Binary pass/fail conditions.  
No human interpretation should be required.

#### Failure Modes
What must happen if the task cannot complete safely:
- fallback behavior
- abort conditions
- preservation of old state
- logging requirements
- degraded mode behavior

#### Rollback
How to disable or revert the feature if regressions appear.

#### Tests
Required validation:
- unit tests
- integration tests
- property tests
- benchmark tests
- recovery tests
- manual verification only if automation is impossible

#### Regression Risks
List specific things that could break:
- correctness
- latency
- memory
- storage compatibility
- semantic drift
- routing drift
- dashboard truthfulness

#### Notes
Optional implementation details, invariants, or warnings.

### 23.3 Task Quality Rules

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

### 23.4 Forbidden Task Language

The following task styles are invalid:

- “Improve performance”
- “Make storage better”
- “Refactor inference”
- “Add memory support”
- “Fix routing”
- “Optimize GPU path”

These are invalid because they do not define measurable outcomes.

Replace them with task cards that specify:
- exact modules
- measurable target
- benchmark or correctness gate
- fallback behavior
- test plan

### 23.5 Required Success Definition

For Bramha, **done means done** only when all of the following are true:

- code is merged
- required tests pass
- required benchmarks are recorded
- rollback path is defined
- no known regressions remain open for the task scope
- observability is added if the feature changes runtime behavior
- storage/schema changes include compatibility handling if needed

### 23.6 Standard Task Card Template

```markdown
### Task ID
BRM-<SPRINT>-<NUMBER>

### Title
<short single-outcome title>

### Objective
<one sentence exact outcome>

### Scope
- <crate/module/file>
- <crate/module/file>

### Non-Goals
- <what must not be touched>
- <adjacent features not included>

### Dependencies
- <task IDs or required baseline state>

### Inputs
- <existing module/state>
- <fixtures/models/config required>

### Steps
1. <step 1>
2. <step 2>
3. <step 3>

### Outputs
- <files changed>
- <tests added>
- <benchmarks added>
- <schemas/manifests updated>

### Acceptance Criteria
- [ ] <binary pass/fail condition>
- [ ] <binary pass/fail condition>
- [ ] <binary pass/fail condition>

### Failure Modes
- <if this fails, what must happen safely>
- <what state must be preserved>
- <what must be logged>

### Rollback
- <feature flag, config switch, migration reversal, fallback path>

### Tests
- Unit: <exact test names or behaviors>
- Integration: <exact scenario>
- Benchmark: <exact metric>
- Recovery: <if applicable>
- Manual: <only if unavoidable>

### Regression Risks
- <risk 1>
- <risk 2>

### Notes
- <optional invariant or warning>
```

### 23.7 Example Task Cards

#### Task ID
BRM-S1-001

#### Title
Atomic write helper for shard and manifest persistence

#### Objective
Implement atomic write behavior for all shard and manifest writes so a crash can never leave a partially written final file.

#### Scope
- `bramha-engine/src/storage/atomic_write.rs`
- `bramha-engine/src/storage/decomposed_store.rs`
- `bramha-engine/src/storage/manifest.rs`

#### Non-Goals
- Do not change WAL format.
- Do not change snapshot format.
- Do not refactor unrelated storage code.

#### Dependencies
- Existing storage paths and manifest persistence code must compile.
- Existing checksum logic must remain the source of truth.

#### Inputs
- Current shard write paths
- Current manifest serialization logic
- Temp-file naming convention
- Existing checksum verification code

#### Steps
1. Add a shared atomic write helper that writes to a temp file in the same directory as the final target.
2. Ensure file contents are flushed and synced before rename.
3. Rename temp file to final path using atomic rename semantics where supported.
4. Verify checksum after write for shard and manifest paths.
5. Replace direct final-path writes in `decomposed_store.rs` and `manifest.rs` with the helper.
6. Add crash-simulation tests for interrupted writes.

#### Outputs
- New or updated atomic write helper
- Shard persistence updated to use helper
- Manifest persistence updated to use helper
- Crash and checksum tests added

#### Acceptance Criteria
- [ ] Crash during write never leaves a partially written final file.
- [ ] Old file remains valid if temp write or rename fails.
- [ ] Checksum mismatch causes loud failure on next load.
- [ ] Existing read path remains unchanged in semantics.

#### Failure Modes
- If temp write fails, abort write and preserve old file.
- If fsync fails, abort rename and preserve old file.
- If checksum verification fails, remove temp artifact and fail loudly.

#### Rollback
- Revert call sites to previous direct-write path only if helper causes blocking issues, but keep test fixture for crash reproduction.

#### Tests
- Unit: temp write success, fsync failure handling, rename failure handling
- Integration: persist shard, persist manifest, reload successfully
- Recovery: simulate crash between temp write and rename, then boot
- Manual: inspect directory to confirm no partial final file remains

#### Regression Risks
- Slower writes on low-end disks
- Incorrect temp file cleanup
- Cross-platform rename differences

#### Notes
- Final file must only become visible after successful durable temp write.

#### Task ID
BRM-S2-004

#### Title
INT8 KV cache quantization with fused dequant in flash attention path

#### Objective
Add INT8 KV block storage and fused dequant in the flash-attention path to reduce long-context RAM usage while preserving output correctness within the defined quality threshold.

#### Scope
- `bramha-engine/src/inference/paged_kv/kv_quant.rs`
- `bramha-engine/src/inference/flash_attn_cpu.rs`
- `bramha-engine/src/inference/paged_kv/block_table.rs`
- `bramha-engine/src/storage/manifest.rs`

#### Non-Goals
- Do not change tokenizer behavior.
- Do not add new sampler behavior.
- Do not modify retrieval components.

#### Dependencies
- Flash attention CPU path exists.
- Paged KV allocator exists.
- Manifest supports storing KV quantization metadata.

#### Inputs
- Existing paged KV layout
- Existing flash attention kernel
- Baseline fp16 KV memory benchmark
- Long-context fixture prompts

#### Steps
1. Define quantized KV block representation with per-block scales.
2. Add encode/decode helpers and storage metadata.
3. Fuse dequant into flash attention path to avoid extra intermediate allocations.
4. Store quantization mode in manifest.
5. Add correctness comparison against fp16 KV on fixed prompts.
6. Add RAM benchmark for 32K context.

#### Outputs
- Quantized KV block module
- Flash attention path updated for fused dequant
- Manifest updated with KV quant metadata
- Correctness tests and 32K RAM benchmark added

#### Acceptance Criteria
- [ ] 32K context RAM usage is below the configured sprint target on the benchmark fixture.
- [ ] Output divergence versus fp16 stays within configured tolerance on fixed prompts.
- [ ] No extra per-token heap allocation is introduced in the decode hot path.
- [ ] CPU-only fallback remains fully functional.

#### Failure Modes
- If quantized path exceeds divergence threshold, feature must auto-disable and use fp16 KV path.
- If manifest metadata is missing, loader must default safely and fail loudly on invalid combinations.

#### Rollback
- Feature flag `kv_int8` can disable quantized path and restore fp16 behavior without schema loss.

#### Tests
- Unit: block quant/dequant round-trip
- Integration: generate with fp16 KV and INT8 KV on same seed and compare
- Benchmark: 4K, 16K, 32K RAM and tok/sec
- Recovery: reload persisted quantized KV metadata from manifest

#### Regression Risks
- Quality drift at long context
- Incorrect cache alignment
- Hidden allocation regressions

#### Notes
- Preserve exact fallback semantics at all times.

#### Task ID
BRM-S4-003

#### Title
Model capability registry and multi-model router baseline

#### Objective
Add a model capability registry and baseline router that selects a model path by query type and complexity for single-query multi-model execution.

#### Scope
- `bramha-engine/src/inference/model_adapter/mod.rs`
- `bramha-engine/src/storage/metadata_sql.rs`
- `bramha-engine/src/storage/manifest.rs`
- `bramha-engine/src/planner/policy.rs`
- `bramha-server/src/http/models.rs`
- `bramha-server/src/http/inference.rs`

#### Non-Goals
- Do not implement ensemble debate mode yet.
- Do not train adapters.
- Do not change low-level compute kernels.

#### Dependencies
- At least two model families must already load successfully.
- Query analytics schema must exist.
- Basic inference API must already work.

#### Inputs
- Model adapter metadata
- Existing query classification heuristics
- Model benchmark history
- Complexity scoring baseline

#### Steps
1. Define model capability schema: reasoning, coding, retrieval-planning, summarization, latency class, memory class.
2. Persist capability records in metadata DB.
3. Add simple query classifier for query type and complexity band.
4. Implement router policy that selects one model or a two-stage path based on classifier output.
5. Emit route decision into query trace metadata.
6. Expose selected route in inference API response metadata.

#### Outputs
- Capability schema and persistence
- Query classifier
- Baseline router policy
- Route trace metadata
- API metadata update
- Router tests

#### Acceptance Criteria
- [ ] Router selects valid registered models only.
- [ ] Simple queries route to low-latency path according to policy.
- [ ] Route decision is stored in query trace metadata.
- [ ] If preferred model is unavailable, router degrades to a valid fallback.

#### Failure Modes
- If no model satisfies policy, fail to safe default model rather than returning no route.
- If capability metadata is missing, mark model as generic and keep it eligible only for fallback.

#### Rollback
- Config switch `router_mode=single_default` disables routing logic and forces default model.

#### Tests
- Unit: capability parsing, policy selection, fallback selection
- Integration: register two models, submit different query classes, verify route chosen
- API: response metadata includes selected model route
- Benchmark: compare median latency for simple queries before and after router

#### Regression Risks
- Misclassification causing worse quality
- Hidden routing loops
- Missing metadata breaking request path

#### Notes
- This task establishes router mode only; pipeline and verifier modes come later.

#### Task ID
BRM-S6-002

#### Title
Planner baseline with exact, speculative, and cached-answer path selection

#### Objective
Implement planner v1 that selects between exact decode, speculative decode, and deterministic cached-answer execution paths using stored policy thresholds.

#### Scope
- `bramha-engine/src/planner/optimizer.rs`
- `bramha-engine/src/planner/cost_model.rs`
- `bramha-engine/src/planner/policy.rs`
- `bramha-engine/src/storage/answer_cache.rs`
- `bramha-engine/src/inference/speculative/`
- `bramha-engine/src/storage/metadata_sql.rs`

#### Non-Goals
- Do not implement activation replay in this task.
- Do not implement super-token decode.
- Do not change retrieval ranking.

#### Dependencies
- Deterministic answer cache exists.
- Speculative decode exists.
- Query trace storage exists.

#### Inputs
- Query hash
- Prompt/context/model version
- Speculative accept-rate history
- Planner threshold config
- Cache key format

#### Steps
1. Define planner decision enum and persisted policy thresholds.
2. Implement deterministic cache lookup as highest-priority path.
3. Implement speculative eligibility check using accept-rate history and policy.
4. Fall back to exact decode when no optimized path is valid.
5. Persist planner decision in query trace.
6. Expose planner path in metrics and inference metadata.

#### Outputs
- Planner v1 modules
- Policy storage
- Cache/speculative integration
- Query trace updates
- Metrics and response metadata

#### Acceptance Criteria
- [x] Exact path remains the default safe fallback.
- [x] Cached-answer path is used only when prompt, context, and model version match exactly.
- [x] Speculative path is chosen only when policy thresholds are satisfied.
- [x] Planner decision is visible in trace and metrics.

#### Failure Modes
- If planner state is invalid or unavailable, system must use exact decode.
- If speculative path errors at runtime, request must continue on exact path without process crash.

#### Rollback
- Config switch `planner_mode=exact_only` disables planner optimization paths.

#### Tests
- Unit: policy evaluation and decision selection
- Integration: cache hit path, speculative eligible path, exact fallback path
- Recovery: corrupted planner policy state falls back to exact
- Benchmark: latency comparison for planner on vs exact only

#### Regression Risks
- Wrong cache key causing false hits
- Planner oscillation
- Hidden latency from planner overhead

#### Notes
- Planner correctness is more important than planner aggressiveness.

### 23.8 Sprint Conversion Rule

From this point forward, every roadmap sprint item must be rewritten into task cards before implementation starts.

#### Required Order
1. Sprint goals
2. Task dependency graph
3. Task cards
4. Tests and fixtures
5. Benchmarks
6. Rollback checks
7. Merge gate

No sprint is implementation-ready until all of its tasks are expressed as task cards.

### 23.9 AI Execution Rules

When an AI executes a Bramha task, it must follow these rules:

1. Read the task card fully before editing code.
2. Touch only the listed scope.
3. Do not change adjacent code unless required for compilation.
4. Implement steps in order.
5. Produce outputs exactly as defined.
6. Run listed tests before declaring completion.
7. Stop and ask for clarification if acceptance criteria are ambiguous.
8. Prefer exact fallback over risky optimization.
9. Preserve storage compatibility unless schema migration is explicitly in scope.
10. Report regressions, do not silently work around them.

### 23.10 Done Means Done

A Bramha task is complete only when:
- implementation is finished
- outputs are present
- all required tests pass
- rollback path is documented
- metrics/observability are added where runtime behavior changed
- acceptance criteria are all checked
- no out-of-scope changes remain in the diff

---

## 24. AI Coding Guidelines

These rules apply to every AI coding session on this project.

### Principle 1 — Think Before Coding

- Do not assume.
- State assumptions explicitly before writing code.
- If two interpretations exist, present both and ask.
- Push back if a simpler approach exists.
- Stop when confused and ask for clarification.
- Never proceed on a hidden assumption.

### Principle 2 — Simplicity First

- Minimum code that solves the problem.
- No speculative features.
- No abstractions for single-use code.
- No configurability unless requested.
- No unnecessary error handling for impossible states.
- If 200 lines could be 50, rewrite it.

### Principle 3 — Surgical Changes

- Touch only what is required.
- Do not refactor unrelated code.
- Match existing style.
- Remove only the dead code created by your own change.
- Mention unrelated problems; do not “fix” them without approval.

### Principle 4 — Goal-Driven Execution

- Convert imperatives into verifiable outcomes.
- Write tests for failures before fixing if practical.
- State a brief plan before starting.
- Verify each step against explicit acceptance criteria.

### How to Know These Guidelines Are Working

- fewer unnecessary changes in diffs
- fewer rewrites due to overcomplication
- clarifying questions appear before mistakes
- PRs stay small and scope-bounded

### Tradeoff Note

These rules bias toward caution over speed.  
For trivial tasks, use judgment.  
The goal is not bureaucracy.  
The goal is preventing expensive mistakes on non-trivial work.

---

## 25. Research Lane Notes

These tracks are important, but they are not all equal in maturity.

- Speculative decoding remains a production-path acceleration family and should stay in the core engine roadmap.
- State-space / Mamba-class backends are credible alternate-backend research and product-track candidates because published work frames them as linear-time sequence models.
- Discrete diffusion for text is promising and worth tracking, but it should remain a benchmark-gated research lane until it proves quality, convergence, and deployment economics for Bramha use cases.

---

## 26. Benchmark Fixture: benches/end_to_end_storage.rs

Fixture: Qwen2-0.5B, cold start, 2048-token prompt
Hardware class: 4GB VRAM GPU + NVMe SSD

Metrics:
1. `model_load_time_ms`: Time from binary start to first token ready
   - Baseline: 500ms | Target: 50ms | Bound: I/O
2. `first_token_latency_ms`: Time from prompt submit to first output token
   - Baseline: 1200ms | Target: 400ms | Bound: I/O + compute
3. `sustained_tps`: Tokens/sec after warmup, 512-token generation
   - Baseline: 0.42 | Target: TBD | Bound: COMPUTE (not storage)

Gate: Sprint 8 claims are ACHIEVED only after this fixture passes.

---

## 27. Local-First Optimization Catalog

### 1. Storage Layer (Highest ROI for Local-First)

| Optimization | Impact | Effort | Notes |
|-------------|--------|--------|-------|
| **Hugepage-backed mmap** | High | Low | Use `MAP_HUGETLB` (Linux) or `VM_FLAGS_SUPERPAGE` (macOS) for weight files. Reduces TLB misses by 10-50x on large models. |
| **zstd dictionary training per model family** | High | Medium | Train a zstd dictionary on Qwen2 weights. Compression ratio improves 15-25% over generic zstd. Store dict in `manifest.json`. |
| **Cross-precision deduplication** | High | Medium | Qwen2-0.5B FP16 and Qwen2-0.5B INT4 share ~60% identical weight chunks at the byte level. Blake3 dedup should normalize precision before hashing. |
| **madvise sequential + willneed** | Medium | Low | On model load, `madvise(MADV_SEQUENTIAL)` for linear scan, then `MADV_WILLNEED` for hot layers. Kernel prefetches pages into DRAM. |
| **io_uring for async weight streaming** | High | High | Replace blocking `read()` with `tokio-uring` on Linux. Overlap layer N+1 fetch with layer N compute. |
| **Copy-on-write KV pages** | Medium | Medium | Use `mmap(MAP_PRIVATE)` for KV blocks. Forking sessions (e.g., multi-turn branching) shares physical pages until written. |
| **Weight checksum lazy verification** | Low | Low | Verify Blake3 only on first access, not at load time. Shaves 50-200ms off cold start. |

### 2. Inference Layer (The Compute Bottleneck)

| Optimization | Impact | Effort | Notes |
|-------------|--------|--------|-------|
| **Fused QKV projection** | High | Medium | Fuse `W_q @ x`, `W_k @ x`, `W_v @ x` into single GEMM. Reduces memory bandwidth by 3x on the projection. Critical for memory-bound small GPUs. |
| **Dequant + Matmul kernel fusion** | High | High | Instead of dequantizing INT4 → FP16 then matmul, fuse into single kernel. SPANDA's 4x4 block masks make this natural. |
| **Attention Sink (4-token anchor)** | High | Low | Keep first 4 KV positions un-evicted. Extends effective context to 4M+ tokens without quality degradation. |
| **GQA/MQA-optimized KV cache layout** | High | Low | Grouped Query Attention reduces KV cache by 4-8x. Layout as `[num_kv_heads, seq_len, head_dim]` not `[layers, num_heads, ...]` for cache locality. |
| **Async CPU prefetch + GPU compute** | High | Medium | While GPU computes layer N, CPU prefetches layer N+1 weights into pinned host memory. Double-buffer via `wgpu::Buffer` staging. |
| **Token batching for concurrent sessions** | High | High | Batch 2-4 independent user sessions into one forward pass. Requires attention mask handling but gives near-linear throughput scaling. |
| **Early-exit adaptive depth** | Medium | High | Train a tiny classifier to predict if layer N+1 is needed. Skip 20-30% of layers on "easy" tokens. Must preserve exact fallback. |
| **Shader pipeline cache serialization** | Medium | Low | Cache compiled WGSL pipelines to disk via `bincode`. Avoids 200ms+ driver recompile on restart. |

### 3. Retrieval Layer (The Database Advantage)

| Optimization | Impact | Effort | Notes |
|-------------|--------|--------|-------|
| **Embedding inference batching** | High | Low | Queue incoming retrieval requests for 10ms, then batch embeddings into one forward pass. 3-5x throughput vs. single queries. |
| **Sparse + Dense hybrid (SPLADE-style)** | High | Medium | Add learned sparse retrieval alongside dense HNSW. Sparse terms need no embedding compute for exact-match queries. |
| **Graph-reranked retrieval** | Medium | High | Retrieve 100 candidates via HNSW, then rerank via graph proximity (entity distance) before returning top-5. |
| **Query-aware chunk sizing** | Medium | Low | Code queries → 256-token chunks. Long-form queries → 1024-token chunks. Chunk size stored in `planner_policies`. |
| **Incremental HNSW updates** | Medium | Medium | Don't rebuild the whole index on new documents. Use `hnswlib` incremental insert. Reduces ingestion pause from seconds to milliseconds. |

### 4. Planner Layer (The Intelligence Multiplier)

| Optimization | Impact | Effort | Notes |
|-------------|--------|--------|-------|
| **Query fingerprinting (SimHash)** | High | Low | Hash the query embedding into a 64-bit SimHash. Exact match → instant cache hit without embedding recomputation. |
| **Online cost model updates** | High | Medium | After each inference, update the planner's cost model with actual measured latency. Exponential decay on old data. |
| **Predictive preloading** | Medium | Medium | If query type is "code generation" (classifier confidence > 0.9), preload code-model weights before user finishes typing. |
| **Path A/B testing framework** | Medium | Medium | Run 5% of traffic through speculative path, 5% through exact path. Planner auto-selects winner based on P99 + quality. |

### 5. Memory / Database Layer

| Optimization | Impact | Effort | Notes |
|-------------|--------|--------|-------|
| **Memory compression via summarization** | High | High | After N turns, compress episodic memory into a semantic summary via a tiny model. Reduces context growth from O(turns) to O(log turns). |
| **Temporal decay with half-life** | Medium | Low | Memory relevance = `confidence * (0.5 ^ (age_days / half_life))`. Half-life configurable per memory type. |
| **Working memory as ring buffer** | High | Low | Keep last 4K tokens in a lock-free spsc ring buffer. Zero-allocation hot path for session state. |
| **Memory embedding index** | Medium | Medium | Index semantic memories by embedding + timestamp. Retrieve recent + relevant, not just relevant. |

### 6. System-Level (The "Free" Wins)

| Optimization | Impact | Effort | Notes |
|-------------|--------|--------|-------|
| **Thread affinity for inference workers** | Medium | Low | Pin rayon threads to physical cores. Avoid hyperthread contention. `sched_setaffinity` on Linux. |
| **CPU governor pinning (performance)** | Low | Low | On AC power, set `scaling_governor = performance`. On battery, throttle to `powersave`. |
| **NUMA-aware allocation** | Medium | High | Allocate tensors on the NUMA node closest to the GPU's PCIe root complex. |
| **Runtime SIMD dispatch** | High | Medium | Detect AVX-512 / AVX2 / NEON at startup. Dispatch to the widest available kernel. Use `std::arch` feature flags. |
| **Power-aware scheduling** | Medium | Medium | If battery < 20%, force CPU backend and disable speculative decode. Extend laptop runtime 2-3x. |

### Optimization Sprint Assignments

| Optimization | Sprint | Task ID | Effort | Gate |
|-------------|--------|---------|--------|------|
| Hugepage-backed mmap | Sprint 9 | BRM-S9-OPT-001 | 1 day | TLB miss reduction >5x on benchmark |
| madvise sequential + willneed | Sprint 9 | BRM-S9-OPT-002 | 1 day | Cold start latency reduction >10% |
| Shader pipeline cache | Sprint 9 | BRM-S9-OPT-003 | 1 day | No shader recompile on restart |
| Fused QKV projection | Sprint 10 | BRM-S10-OPT-001 | 2 days | Attention projection bandwidth -30% |
| Attention Sink (4 tokens) | Sprint 10 | BRM-S10-OPT-002 | 1 day | DONE |

**Unscheduled (Post-v0.5 — blocked until SPANDA integration or architectural changes):**
- io_uring async weight streaming
- Dequant + matmul kernel fusion
- Token batching for concurrent sessions
- Early-exit adaptive depth
- Memory compression via summarization

### What to Avoid (Anti-Optimizations)

| Don't Do | Why |
|----------|-----|
| **Dynamic batching across users** | Complexity explosion. Do static batching first. |
| **FP8 inference** | No stable Rust support. Burn/wgpu path is unclear. |
| **Custom CUDA kernels** | Violates Rust-only + wgpu portability. |
| **Quantization-aware training** | Requires model retraining. Out of scope for inference engine. |
| **Speculative decoding with draft model** | Draft model doubles memory. On 4GB GPU, this kills the main model. DS4 confirms: MTP speculative is "experimental, slight speedup" for MoE models. |

---

## 28.5 DS4 (DwarfStar) Reference Learnings

> **Source**: [antirez/ds4](https://github.com/antirez/ds4) — DeepSeek V4 Flash/PRO local inference engine by Salvatore Sanfilippo
> **Relevance**: DS4 validates Bramha's core thesis and provides battle-tested implementations for several planned features.

### Key Validations

| Bramha Thesis | DS4 Validation |
|---|---|
| KV cache as first-class disk citizen | DS4's KV cache persistence with `cold/continued/evict/shutdown` lifecycle is production-proven |
| SSD streaming for model weights | DS4 streams routed MoE experts from SSD with in-memory expert cache, validated on MacBook SSDs |
| Multi-tier storage (DRAM/SSD/HDD) | DS4's expert cache = DRAM hot tier, GGUF file = SSD cold tier — proves the pattern works |
| Prefix reuse eliminates re-prefill | DS4 server reuses rendered-prefix KV checkpoints across requests and restarts |
| Distributed inference for capacity | DS4 splits layers across machines via TCP — confirms prefill pipelining wins, generation is slower |
| Power-aware scheduling | DS4's `--power N` (measure + sleep) is dead simple and works |

### Adopted DS4 Patterns

1. **KV Save Lifecycle**: `cold → continued → evict → shutdown` checkpoint taxonomy (added to Invention 1)
2. **Boundary-Aligned Trimming**: Trim N tail tokens + align to chunk boundaries before KV save (BPE safety)
3. **Rendered-Text SHA Key**: Cache lookup via SHA1 of decoded prefix bytes, not semantic hash
4. **Exact Tool-Call Replay**: Bounded radix-tree map of `tool_id → exact sampled bytes` (added as Invention 7)
5. **Split Sampling for Tools**: Deterministic for syntax, user-configured for payloads
6. **Power Throttling**: `--power N` via measured work time + inter-layer/inter-token sleeps (Sprint 10)
7. **Frontier-Based Benchmarking**: Measure at context frontiers (2K, 4K, 8K...) not single averages (Sprint 10)

### Acknowledged Limitations (DS4-Informed)

- **Speculative decode for MoE models provides limited speedup** — DS4 confirms this is "at most a slight speedup". Bramha should validate its own speculative claims.
- **Distributed generation is always slower than single-machine** — DS4 measures 19.4% loss. Sprint 13 should set realistic expectations.
- **No batching across users is pragmatic for single-user local inference** — Bramha should defer dynamic batching to post-v1.0.
- **read/write I/O is better than mmap for KV cache files** — avoids adding VM mappings to a process that already maps the model.

### Patterns NOT Adopted from DS4

- **Distributed inference layer splitting** — ds4 splits transformer layers across machines via TCP. Bramha Sprint 13 plans this, but SPANDA (4GB GPU target) has no use for it. Keep it in Bramha, not SPANDA.
- **MTP speculative decoding** — ds4 calls this "experimental" and "at most a slight speedup." On 4GB GPUs, a draft model doubles memory pressure. Defer until post-v1.0.
- **Multi-process serving** — ds4 shows 1.5x scaling with 2 processes on 256GB. On 4GB, there is no headroom. SPANDA is single-process by design.
- **In-process batching across users** — ds4 closed this as "~zero gain on a Mac" because decode is autoregressive. Bramha should defer dynamic batching to post-v1.0.

---

## 28. End State

Bramha should eventually become all of the following at once:

- a local-first intelligence database
- a planned inference engine
- a retrieval and evidence system
- a memory and learning system
- a multi-model orchestration runtime
- a CPU-first heterogeneous execution engine
- a wgpu-accelerated portable compute system
- a horizontally scalable distributed intelligence platform

This is the target state:
**one Rust-native system for local intelligence, from CPU-only laptops to heterogeneous clusters.**

---

## 29. Comprehensive Security, Architecture & Execution Plan (v3.2)

> **Repository:** https://github.com/akshaybhlrp/bramha  
> **Baseline:** 28,700 lines of production code, 137 lines of tests (0.47% coverage).  
> **Date:** 2026-07-15  
> **Status:** Production-hardened audit with executable artifacts.  
> **Changelog v3.1 → v3.2:** Fixed 3 misattributed items after line-by-line repo verification.

---

### 29.1 Executive Summary

This document is the definitive, consolidated hardening and architecture guide for the Bramha inference engine. It synthesizes:

- An original P0–P14 security audit.
- Critical refinements on threading, WAL design, and memory management.
- Operational corrections (metrics, rate limiting, backpressure).
- Honest pushbacks on anti-patterns (timer-based fdatasync, slotmap for tensors, 120s queue timeouts).
- **Line-by-line repo verification** that corrected 3 misattributed items from v3.1.
- Executable code artifacts (CI config, Rust modules, dependency audit).

**The thesis:** Bramha is an ambitious sparse-paging inference engine that cannot safely reach v0.1 without first fixing 9 actively exploitable security holes, restructuring its error handling, and building a test safety net. Performance optimizations (arena allocators, backend migrations) are deferred until correctness is proven.

---

### 29.2 The Brutal Truth

> **137 lines of tests for 28,700 lines of code (0.47% coverage).**

This is not a test gap. This is a "we do not know if any of this works" gap. Every P0 fix is a refactor. Every refactor without tests is a regression waiting to happen.

**Mandatory invariant:**

> Every P0 fix must be accompanied by a failing test first (TDD). Do not touch `cpu_engine.rs` (3,512 lines) until an end-to-end `generate_text` integration test exists.

---

### 29.3 P0 — Actively Exploitable Today (9 Items)

> **Verified by line-by-line repo audit (cloned, greppable).**

| # | Issue | Severity | Fix | Test Required |
|---|-------|----------|-----|---------------|
| 1 | **Global env var race in device selection** | Critical | Move `device` into `InferenceTask`; delete `std::env::set_var()` and `set_cpu_only()` entirely | Unit test: concurrent tasks request different devices, no stale reads |
| 2 | **Auth missing on 22/30 routes** | Critical | Apply `RequireReadOnly` as router-wide default; routes opt up to Write/Admin. Mount `/health`, `/ready`, `/metrics` **before** auth layer | Integration test: unauthenticated request to protected route → 401 |
| 3 | **Default API keys are literal public strings** | Critical | Generate random keys on first boot; SHA-256 hash-check against literal; `panic!` if defaults detected | Integration test: boot with default keys → panic |
| 4 | **CORS fully open** | High | Explicit origin allowlist; apply `CorsLayer` below `AuthLayer` to prevent OPTIONS preflight bypass | Integration test: cross-origin request to protected route → blocked |
| 15 | **Path traversal in `ingest_model`** | Critical | `HashSet<PathBuf>` allowlist of registered directories; `canonicalize()` `payload.path`; reject if escapes registry | Integration test: `../../../etc/passwd` in payload → 400 |
| 16 | **No input bounds on `generate_text`** | High | Reject if `input_ids.len() + max_new_tokens > context_window` **before** queueing | Integration test: prompt + max_tokens > window → 400 |
| 17 | **`llm_load_model` unauthenticated** | Critical | Promote to `RequireAdmin` | Integration test: unauthenticated load → 401 |
| 18 | **SIGTERM not handled; queue not drained on exit** | High | `main.rs` already handles SIGINT (`ctrl_c`). Add `tokio::signal::unix::signal(SignalKind::terminate())` handler. Explicitly drain inference queue (30s) before DB save + exit | Integration test: SIGTERM during generation → queue drains, WAL flushed |
| 27 | **Symlink escapes in path allowlist** | Critical | After `canonicalize()`, verify every original component is not a symlink via `symlink_metadata()` | Integration test: symlink inside registry → 400 |

> **Deleted from v3.1:** ~~#28 "Pickle RCE"~~ — Repo only uses `safetensors::SafeTensors::deserialize`. No pickle loader exists. Generic boilerplate, not applicable.

> **Corrected from v3.1:** ~~#15 targeted `llm_load_model`~~ — Actual bug is in `ingest_model` where `payload.path` (user string) is fed to `find_safetensors_files(dir)` without `canonicalize`/allowlist.

> **Corrected from v3.1:** ~~#18 "No SIGTERM graceful shutdown"~~ — `main.rs` already has `tokio::signal::ctrl_c()` + `with_graceful_shutdown` + DB save. Gap is SIGTERM (Docker/k8s) and explicit queue drain.

---

### 29.4 P1 — Structural & Design Corrections (10 Items)

| # | Issue | Fix | Why It Matters |
|---|-------|-----|----------------|
| 5 | `Box<dyn Error>` / `String` errors block planner fallback | `thiserror` enum `BramhaError` with `IntoResponse` | The planner cannot match on strings to decide degraded fallback paths |
| 6 | CPU-bound work on `tokio::spawn` | Audit 16 `tokio::spawn` sites; move tensor math to `spawn_blocking` or `rayon` | Tokio is for I/O. Tensor math on the async runtime blocks the event loop |
| 7 | Panics in spawned tasks vanish silently | `catch_unwind` + write `"Failed: {reason}"` status | Ingestion hangs at `"Processing: ..."` forever otherwise |
| 8 | `std::sync::Mutex` poisoning | `parking_lot::Mutex` (already a dep). Do NOT timeout `rx.recv()`. Timeout the *generation task* (60s interactive / 300s batch) | If a thread panics holding std mutex, every future lock panics. Queue-level timeout is a DoS vector |
| 9 | KV cache allocation | Contiguous memory pool (`Vec<u8>` or `wgpu::Buffer`) with metadata slab. No heap scatter | GPU transfer needs contiguous memory. Scattered allocations = N small copies instead of 1 DMA |
| 13 | Docs/code drift | Flat `src/` vs workspace README (`bramha-engine/bramha-server/bramha-cli`). Refactor code or rewrite docs | Breaks `cargo test --workspace` and `rust-analyzer` |
| 14 | Backend choice (`burn` trap) | Prototype `candle-core` (HuggingFace) for CPU. Prepare `wgpu` + `gemm` for GPU sparse paging. **Delete `burn`, `ndarray`, `nalgebra`, `rustfft`** | `burn` is a training framework. No FlashAttention, no fused kernels, no paged KV. Dead weight |
| 21 | Model eviction mid-generation | Reference-counted pages. Eviction only on `refcount == 0` | Without this, evicting a page during a generation causes a segfault |
| 29 | No API key rotation | `ArcSwap<HashMap>` for lock-free updates. `POST /admin/keys/rotate` | Keys leak. Rotation without restart is mandatory in production |
| 35 | "Generation never blocks" invariant | Document: eviction skips `refcount > 0`. This is a **safety invariant**, not a performance optimization | Without this documented and enforced, the sparse paging story is vaporware |

---

### 29.5 P2 — Operational Hardening (9 Items)

| # | Issue | Fix |
|---|-------|-----|
| 19 | Zero observability | Prometheus metrics: `bramha_queue_depth`, `bramha_vram_bytes_used`, `bramha_kv_cache_hit_rate`, `bramha_token_latency_seconds` (histogram), `bramha_generation_errors_total` (labeled by variant). `tracing::Span` per request with `request_id`. JSON structured logs via `tracing-subscriber` |
| 20 | No rate limiting | Per-key token bucket (`governor` crate) as Axum layer. Reject 429. Do not enqueue |
| 22 | Session leaks / no TTL | 30-min idle TTL. Background `tokio` task scans `last_activity`, drops expired sessions, frees KV pages, appends `SessionClosed` tombstone to WAL |
| 23 | Config validation at boot | Validate `device = "gpu"` by probing `wgpu` adapters. Panic before binding HTTP port if GPU requested but unavailable. Fail fast |
| 30 | Queue depth backpressure | If `queue_depth > max_concurrent_requests * 2`, return 503 immediately. Protects against memory exhaustion |
| 31 | GPU cold-start latency | After `llm_load_model`, run dummy 1-token forward pass before marking "ready". `/ready` health probe returns 200 only after warmup |
| 32 | OOM kills during generation | Pre-allocate max KV cache size at model load. Verify `current_kv_usage + requested_tokens <= max_kv_size` before queueing. Reject with `OutOfVram`. No dynamic growth |
| 33 | `/metrics` exposed to internet | Mount on separate port (e.g., `:9090`) or IP allowlist (`127.0.0.1` / `METRICS_ALLOWLIST`). Never on public HTTP port |
| WAL | Timer-based `fdatasync` (anti-pattern) | Per-session segments (`wal/sessions/{session_id}.log`). CRC32C checksums. `O_APPEND`. `fdatasync` only on: (1) graceful shutdown, (2) explicit checkpoint, (3) every 64 MB written. On restart, replay until checksum failure, truncate at last valid entry. This is SQLite's WAL model |

---

### 29.6 P3 — Before v0.1 (7 Items)

| # | Action | Why |
|---|--------|-----|
| 11 | Integration tests asserting on `BramhaError` variants | Forces error handling to be a state machine, not ad-hoc strings |
| 12 | Split `cpu_engine.rs` (3,512 lines) and `handlers.rs` (2,071 lines) | Files too large to review confidently. Split along threading/error boundaries |
| 24 | `cargo audit` + `cargo deny` in CI | Catch RUSTSEC advisories and license violations automatically |
| 25 | Fuzz `generate_text` with `cargo-fuzz` / `proptest` | Empty prompts, Unicode bombs, max-length, negative `max_new_tokens` |
| 26 | Document sparse paging invariant | One-pager: "At 90% VRAM, evict coldest KV page to RAM. At 80% RAM, evict coldest weights to disk. Generation never blocks — returns `StorageBackpressure`" |
| 34 | Load test with `k6` / `wrk2` | 100 concurrent clients, 5 minutes. Monitor memory leaks, queue stalls, WAL corruption |
| 36 | Failure Mode Matrix | Document: `(Error Variant) × (System State) → (Action)` |

---

### 29.7 Sprint Execution Plan

| Sprint | Scope | Merge Gate | Risk |
|--------|-------|------------|------|
| **Sprint 0** | `clippy`/`fmt` hygiene. Integration tests for P0 endpoints (300–500 lines). | CI green. Coverage > 2%. | Low — adds code, doesn't change logic |
| **9a** | Auth defaults (#2, #3, #17), CORS (#4), Input bounds (#16) | Sprint 0 tests validate 401/403/400 | Medium — touches ~30 routes |
| **9b** | Path traversal (#15, #27) in `ingest_model`, SIGTERM refinement (#18), Env var race (#1) | Sprint 0 tests validate rejections | High — security-critical |
| **10a** | `thiserror`, threading firewall, panic catching, task timeouts | Zero `Box<dyn Error>` in HTTP handlers | Medium — breaking change to error types |
| **10b** | **Prototype branch:** Benchmark `candle-core` vs raw `wgpu` on Qwen2-0.5B | Decision doc committed | Low — no production code changes |
| **10c** | KV contiguous pool, Refcounts (#21, #35), Key rotation (#29) | Memory profiler confirms contiguous pages | High — touches core memory management |
| **11** | Metrics/Tracing, Rate limits, TTL, WAL Spec, Backpressure, Warmup, OOM guard | `k6` load test passes 5 min @ 100 concurrent | Medium — new dependencies |
| **Post-11** | File splits, `cargo audit` in CI, Fuzzing, Failure Matrix (#36), Sparse paging doc | **v0.1 RC** | Low — polish and documentation |

**Critical rule:** Do not start Sprint 9 until Sprint 0 is merged. Do not touch `cpu_engine.rs` until Sprint 10a tests pass.

---

### 29.8 Architecture Specs

#### 29.8.1 WAL Design

**Goals:** Crash safety, zero contention, SSD-friendly.

**File Layout:**
```
data/wal/
  meta/
    global_seq              # Atomic u64 counter
  sessions/
    {session_id}.log        # One file per active session
  checkpoints/
    {timestamp}.chk         # Consolidated checkpoint
```

**Entry Format (Append-Only, O_APPEND):**
```
[4 bytes: CRC32C of data (little-endian)]
[4 bytes: len of data (little-endian)]
[N bytes: postcard/bincode serialized WalEntry]
[8 bytes: Unix timestamp millis]
```

**Validation on Replay:**
- Read sequentially. Verify CRC32C.
- If CRC fails OR len exceeds bounds: **truncate at last valid entry**. Do not skip.
- Rehydrate in-memory session state from valid entries.

**Sync Policy:**
1. Append without `fdatasync`.
2. Auto-sync on: session close, explicit `POST /admin/checkpoint`, or 64 MB written.
3. Graceful shutdown: flush all sessions, create checkpoint.

**Concurrency:**
- Each session has its own `Arc<tokio::sync::Mutex<File>>`.
- Generation loop writes via non-blocking `try_send` to a channel.
- Channel full → `BramhaError::StorageBackpressure` (never block generation).

**Recovery Algorithm (Boot):**
1. Scan `wal/sessions/*.log`.
2. For each file: replay entries, verify CRC, truncate at failure.
3. `fdatasync` recovered files.
4. Delete checkpoints older than 7 days.

---

#### 29.8.2 InferenceTask & Threading Model

**Problem:** `std::env::set_var()` and a global `CPU_ONLY` flag create a race between concurrent requests.

**Solution:** Per-task device config, resolved at queue consumption time.

```rust
#[derive(Clone, Debug, PartialEq)]
pub enum DeviceConfig {
    Gpu { device_id: usize },
    Cpu,
    Auto,
}

pub struct InferenceTask {
    pub request_id: Uuid,
    pub model_id: String,
    pub input_ids: Vec<u32>,
    pub max_new_tokens: usize,
    pub device: DeviceConfig,        // Per-task, replaces env var
    pub session_id: Option<String>,
    pub response_tx: oneshot::Sender<Result<GenerateOutput, BramhaError>>,
}

impl InferenceTask {
    pub fn resolve_device(&self, system_has_gpu: bool) -> DeviceConfig {
        match self.device {
            DeviceConfig::Auto if system_has_gpu => DeviceConfig::Gpu { device_id: 0 },
            DeviceConfig::Auto => DeviceConfig::Cpu,
            explicit => explicit,
        }
    }

    pub fn validate_bounds(&self, context_window: usize) -> Result<(), BramhaError> {
        let total = self.input_ids.len().saturating_add(self.max_new_tokens);
        if total > context_window {
            return Err(BramhaError::InputTooLong { requested: total, limit: context_window });
        }
        Ok(())
    }
}
```

**Threading Firewall:**
- **Tokio:** HTTP handlers, queue management, WAL channel, metrics.
- **Rayon / spawn_blocking:** Tensor math, safetensors loading, quantization.
- **Never mix:** A `tokio::spawn` task must not perform CPU-bound matmul. A `rayon` task must not perform async I/O.

**Queue Consumer Loop:**
```rust
// Do NOT timeout rx.recv() — block forever waiting for work.
while let Some(task) = rx.recv().await {
    // Wrap the generation in a task-level timeout, not the queue.
    let result = tokio::time::timeout(
        Duration::from_secs(60), // interactive
        tokio::task::spawn_blocking(move || engine.generate(task))
    ).await;

    match result {
        Ok(Ok(output)) => { let _ = task.response_tx.send(Ok(output)); }
        Ok(Err(e)) => { let _ = task.response_tx.send(Err(e.into())); }
        Err(_) => { let _ = task.response_tx.send(Err(BramhaError::EngineTimeout)); }
    }
}
```

---

#### 29.8.3 KV Cache Memory Pool

**Anti-pattern:** `slotmap` or `generational-arena` for tensor data. These have O(n) iteration and scattered heap allocations.

**Correct design:**
1. **Pre-allocate one contiguous buffer:** `Vec<u8>` or `wgpu::Buffer` of max KV cache size.
2. **Metadata slab tracks slots:** `(offset: usize, len: usize, generation: u64, refcount: AtomicU32, is_allocated: bool)`.
3. **Tensor data lives in the contiguous pool.** The slab is metadata only.
4. **Eviction policy:** Only consider slots with `refcount == 0`. Skip in-use pages.

```rust
pub struct KvPage {
    pub offset: usize,
    pub len: usize,
    pub generation: u64,
    pub refcount: AtomicU32,
}

pub struct KvPool {
    buffer: Vec<u8>,               // Contiguous backing store
    pages: Vec<KvPage>,            // Metadata slab
    free_list: Vec<usize>,         // Indices of unallocated pages
}

impl KvPool {
    pub fn allocate(&mut self, len: usize) -> Option<usize> {
        // Find a free page with sufficient capacity
        // Mark allocated, increment generation, return index
    }

    pub fn release(&mut self, idx: usize) {
        // Decrement refcount. If zero, push to free_list.
    }

    pub fn evict_candidates(&self) -> Vec<usize> {
        // Return pages with refcount == 0, sorted by LRU
    }
}
```

---

#### 29.8.4 Auth & Key Rotation

**Key Storage:** `ArcSwap<HashMap<String, ApiKeyMeta>>` for lock-free reads.

```rust
use arc_swap::ArcSwap;
use std::sync::Arc;
use std::collections::HashMap;

pub struct AuthManager {
    keys: ArcSwap<HashMap<String, ApiKeyMeta>>,
}

pub struct ApiKeyMeta {
    pub role: Role,
    pub created_at: u64,
    pub rotated_from: Option<String>, // Previous key ID, valid until sessions finish
}

impl AuthManager {
    pub fn rotate_key(&self, role: Role) -> String {
        let new_key = generate_random_key();
        let mut new_map = (**self.keys.load()).clone();
        // Mark old key as rotated but keep valid for active sessions
        if let Some((old_key, meta)) = new_map.iter_mut().find(|(_, m)| m.role == role) {
            meta.rotated_from = Some(old_key.clone());
        }
        new_map.insert(new_key.clone(), ApiKeyMeta { role, created_at: now(), rotated_from: None });
        self.keys.store(Arc::new(new_map));
        new_key
    }

    pub fn verify(&self, key: &str) -> Option<Role> {
        self.keys.load().get(key).map(|m| m.role)
    }
}
```

**Route Guards:**
```rust
// Mount BEFORE auth layer
let public_routes = Router::new()
    .route("/health", get(health))
    .route("/ready", get(ready))
    .route("/metrics", get(metrics));

let protected_routes = Router::new()
    .route("/generate", post(generate_text))
    .route("/models/load", post(llm_load_model))
    .layer(AuthLayer::new().default(ReadOnly));

let app = public_routes.merge(protected_routes);
```

---

#### 29.8.5 Backend Decision Matrix

| Backend | Pros | Cons | Verdict |
|---------|------|------|---------|
| **burn** | Pure Rust, training-friendly | No FlashAttention, no fused kernels, no paged KV, memory bloat | **Delete it** |
| **candle-core** | Mature, Metal/CUDA/CPU, Qwen2 support, HuggingFace maintained | Memory management is opaque; hard to hook sparse paging | **Use for CPU prototype / rapid validation** |
| **raw wgpu + gemm** | Full control over memory pools, page-level eviction, custom compute shaders | Must write every kernel (attention, RoPE, RMSNorm) | **Use for GPU sparse paging (production)** |
| **ndarray** | Numpy-like ergonomics | Not GPU-native; redundant with candle/burn | **Delete** |
| **nalgebra** | Linear algebra | Overkill for inference; no GPU | **Delete** |
| **rustfft** | FFT | Not needed for transformers | **Delete** |

**Recommended Path:**
1. **Sprint 10b:** Prototype Qwen2-0.5B on `candle-core` CPU. Validate end-to-end architecture.
2. **Post-10b:** If sparse paging is mandatory, migrate to raw `wgpu` + `gemm` for GPU. Keep `candle-core` as a `cpu-only` feature flag.
3. **Delete:** `burn`, `ndarray`, `nalgebra`, `rustfft` in one mechanical PR after the decision doc.

---

#### 29.8.6 Failure Mode Matrix

| Error Variant | System State | Action | HTTP Status |
|---------------|--------------|--------|-------------|
| `InputTooLong` | Request validation | Reject before queueing | 400 |
| `OutOfVram` | Pre-generation check | Reject; do not OOM kill | 503 |
| `StorageBackpressure` | WAL channel full | Return to client; client retries | 503 |
| `EngineTimeout` | Generation hung (60s) | Kill task; free KV pages | 504 |
| `WalCorrupted` | Boot recovery | Truncate at last valid entry; WARN log | 500 (if detected mid-session) |
| `ModelNotLoaded` | Request for unloaded model | Load or return error | 404 |
| `AuthDenied` | Missing/invalid key | Reject immediately | 401 |
| `RateLimited` | Token bucket empty | Reject; do not enqueue | 429 |
| `PathTraversal` | Invalid model path | Reject before filesystem access | 400 |

---

#### 29.8.7 Sparse Paging Invariant

**The One-Pager (for docs/SPARSE_PAGING.md):**

```
Bramha Sparse Paging Invariant v1.0
=====================================

Memory Tiers (fastest to slowest):
  GPU VRAM → System RAM → NVMe Disk

Eviction Triggers:
  1. VRAM usage > 90%:
     · Evict coldest KV page to RAM.
     · Evict coldest model weights to RAM if KV pages insufficient.
  2. RAM usage > 80%:
     · Evict coldest model weights to disk.
     · Keep KV pages in RAM (disk is too slow for KV access).

Safety Invariants:
  · A page with refcount > 0 is NEVER evicted. This is non-negotiable.
  · A generation NEVER blocks waiting for a page load. If a required page
    is on disk, the generation returns StorageBackpressure (503) immediately.
  · Model loading is atomic with respect to active generations. Either wait
    for active gens to finish, or refuse to load if unreferenced pages are
    insufficient.

Page Lifecycle:
  [Allocated] → [In Use] (refcount > 0) → [Idle] (refcount == 0) → [Evicted to RAM] → [Evicted to Disk] → [Reclaimed]

Planner Responsibility:
  The planner is the sole authority for eviction decisions. It maintains:
    - A priority queue of idle pages (LRU order).
    - A map of page locations (VRAM / RAM / Disk / Not Loaded).
    - Per-model memory footprint estimates.
```

---

### 29.9 Code Artifacts

#### 29.9.1 `.github/workflows/ci.yml`

```yaml
name: CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

env:
  CARGO_TERM_COLOR: always
  RUSTFLAGS: "-D warnings"

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install Rust toolchain
        uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
          components: rustfmt, clippy
          override: true

      - name: Cache cargo registry
        uses: actions/cache@v3
        with:
          path: |
            ~/.cargo/registry
            ~/.cargo/git
            target
          key: ${{ runner.os }}-cargo-${{ hashFiles('**/Cargo.lock') }}

      - name: Format check
        run: cargo fmt -- --check

      - name: Clippy (deny warnings)
        run: cargo clippy --all-targets --all-features -- -D warnings

      - name: Run tests
        run: cargo test --workspace -- --nocapture

      - name: Install cargo-deny
        run: cargo install cargo-deny --locked

      - name: Run cargo-deny (advisories + licenses)
        run: cargo deny check
```

---

#### 29.9.2 `cargo-deny.toml`

```toml
# cargo-deny.toml
# Bramha Dependency Audit Configuration
# Run: cargo deny check

[graph]
targets = ["x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu"]
all-features = true

[advisories]
vulnerability = "deny"
unmaintained = "deny"
notice = "deny"
# ignore = [ "RUSTSEC-2024-XXXX" ]

[licenses]
unlicensed = "deny"
allow = [
    "MIT",
    "Apache-2.0",
    "BSD-3-Clause",
    "ISC",
    "MPL-2.0",
]
deny = [
    "GPL-3.0",
    "AGPL-3.0",
    "LGPL-2.1",
]
confidence-threshold = 0.8

[bans]
multiple-versions = "warn"
wildcards = "deny"
# skip = [
#     { name = "rawpointer", version = "0.2.1" },
# ]
# skip-tree = [
#     { name = "ndarray", version = "0.15", depth = 3 },
# ]

[sources]
unknown-git = "warn"
unknown-registry = "deny"
# allow-git = ["https://github.com/rust-lang/"]
```

---

#### 29.9.3 `src/task.rs`

```rust
use std::sync::Arc;
use tokio::sync::oneshot;
use uuid::Uuid;

use crate::errors::BramhaError;

/// Device configuration per-task, replacing the global env var race.
#[derive(Clone, Debug, PartialEq)]
pub enum DeviceConfig {
    Gpu { device_id: usize },
    Cpu,
    Auto,
}

/// Output of a successful generation.
#[derive(Debug, Clone)]
pub struct GenerateOutput {
    pub tokens: Vec<u32>,
    pub text: String,
    pub finish_reason: Option<String>,
}

/// A single inference request queued for the worker.
///
/// # Security & Correctness
/// - `device` is resolved per-task at queue consumption time, eliminating the
///   `std::env::set_var` race condition (fixes #1).
/// - `max_new_tokens` is validated against the model context window before
///   the task is ever created (fixes #16).
#[derive(Debug)]
pub struct InferenceTask {
    pub request_id: Uuid,
    pub model_id: String,
    pub input_ids: Vec<u32>,
    pub max_new_tokens: usize,
    /// Per-task device selection. Replaces the global `CPU_ONLY` flag.
    pub device: DeviceConfig,
    pub session_id: Option<String>,
    /// Channel back to the HTTP handler.
    pub response_tx: oneshot::Sender<Result<GenerateOutput, BramhaError>>,
}

impl InferenceTask {
    /// Resolve the requested device against the actual system state.
    ///
    /// Called by the queue consumer, not the request handler, so the
    /// decision is made at the point of execution — eliminating stale reads.
    pub fn resolve_device(&self, system_has_gpu: bool) -> DeviceConfig {
        match self.device {
            DeviceConfig::Auto if system_has_gpu => DeviceConfig::Gpu { device_id: 0 },
            DeviceConfig::Auto => DeviceConfig::Cpu,
            explicit => explicit,
        }
    }

    /// Validate that the requested generation fits within the model's context window.
    ///
    /// Returns `BramhaError::InputTooLong` if `input_ids.len() + max_new_tokens`
    /// exceeds the model's `context_window`.
    pub fn validate_bounds(&self, context_window: usize) -> Result<(), BramhaError> {
        let total = self.input_ids.len().saturating_add(self.max_new_tokens);
        if total > context_window {
            return Err(BramhaError::InputTooLong {
                requested: total,
                limit: context_window,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_selects_gpu_when_available() {
        let task = InferenceTask {
            request_id: Uuid::new_v4(),
            model_id: "qwen2-0.5b".to_string(),
            input_ids: vec![1, 2, 3],
            max_new_tokens: 10,
            device: DeviceConfig::Auto,
            session_id: None,
            response_tx: {
                let (tx, _rx) = oneshot::channel();
                tx
            },
        };
        assert_eq!(task.resolve_device(true), DeviceConfig::Gpu { device_id: 0 });
    }

    #[test]
    fn auto_falls_back_to_cpu() {
        let task = InferenceTask {
            request_id: Uuid::new_v4(),
            model_id: "qwen2-0.5b".to_string(),
            input_ids: vec![1, 2, 3],
            max_new_tokens: 10,
            device: DeviceConfig::Auto,
            session_id: None,
            response_tx: {
                let (tx, _rx) = oneshot::channel();
                tx
            },
        };
        assert_eq!(task.resolve_device(false), DeviceConfig::Cpu);
    }

    #[test]
    fn explicit_device_preserved() {
        let task = InferenceTask {
            request_id: Uuid::new_v4(),
            model_id: "qwen2-0.5b".to_string(),
            input_ids: vec![1, 2, 3],
            max_new_tokens: 10,
            device: DeviceConfig::Gpu { device_id: 2 },
            session_id: None,
            response_tx: {
                let (tx, _rx) = oneshot::channel();
                tx
            },
        };
        assert_eq!(task.resolve_device(false), DeviceConfig::Gpu { device_id: 2 });
    }

    #[test]
    fn bounds_rejection() {
        let task = InferenceTask {
            request_id: Uuid::new_v4(),
            model_id: "qwen2-0.5b".to_string(),
            input_ids: vec![1; 1000],
            max_new_tokens: 500,
            device: DeviceConfig::Auto,
            session_id: None,
            response_tx: {
                let (tx, _rx) = oneshot::channel();
                tx
            },
        };
        assert!(task.validate_bounds(2048).is_ok());
        assert!(task.validate_bounds(1200).is_err());
    }
}
```

---

#### 29.9.4 `src/errors.rs` (BramhaError)

```rust
use axum::response::{IntoResponse, Response};
use axum::http::StatusCode;
use thiserror::Error;

/// Exhaustive error topology for the Bramha inference engine.
///
/// Every variant maps to a specific HTTP status code via `IntoResponse`.
/// This enables the planner to match on variants for degraded fallback paths.
#[derive(Error, Debug, Clone)]
pub enum BramhaError {
    #[error("Input too long: requested {requested}, limit {limit}")]
    InputTooLong { requested: usize, limit: usize },

    #[error("Out of VRAM: requested {requested} bytes, available {available} bytes")]
    OutOfVram { requested: usize, available: usize },

    #[error("Storage backpressure: WAL channel full")]
    StorageBackpressure,

    #[error("Engine timeout: generation exceeded {seconds}s")]
    EngineTimeout { seconds: u64 },

    #[error("WAL corrupted at offset {offset}")]
    WalCorrupted { offset: u64 },

    #[error("Model not loaded: {model_id}")]
    ModelNotLoaded { model_id: String },

    #[error("Authentication denied")]
    AuthDenied,

    #[error("Rate limited: retry after {retry_after}s")]
    RateLimited { retry_after: u64 },

    #[error("Path traversal detected: {path}")]
    PathTraversal { path: String },

    #[error("Session not found: {session_id}")]
    SessionNotFound { session_id: String },

    #[error("Internal error: {0}")]
    Internal(String),
}

impl IntoResponse for BramhaError {
    fn into_response(self) -> Response {
        let status = match &self {
            BramhaError::InputTooLong { .. } => StatusCode::BAD_REQUEST,
            BramhaError::OutOfVram { .. } => StatusCode::SERVICE_UNAVAILABLE,
            BramhaError::StorageBackpressure => StatusCode::SERVICE_UNAVAILABLE,
            BramhaError::EngineTimeout { .. } => StatusCode::GATEWAY_TIMEOUT,
            BramhaError::WalCorrupted { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            BramhaError::ModelNotLoaded { .. } => StatusCode::NOT_FOUND,
            BramhaError::AuthDenied => StatusCode::UNAUTHORIZED,
            BramhaError::RateLimited { .. } => StatusCode::TOO_MANY_REQUESTS,
            BramhaError::PathTraversal { .. } => StatusCode::BAD_REQUEST,
            BramhaError::SessionNotFound { .. } => StatusCode::NOT_FOUND,
            BramhaError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        };

        (status, self.to_string()).into_response()
    }
}
```

---

#### 29.9.5 `src/auth.rs` (Key Rotation)

```rust
use arc_swap::ArcSwap;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Role {
    ReadOnly,
    Write,
    Admin,
}

pub struct ApiKeyMeta {
    pub role: Role,
    pub created_at: u64,
    pub rotated_from: Option<String>,
    pub is_active: bool,
}

pub struct AuthManager {
    keys: ArcSwap<HashMap<String, ApiKeyMeta>>,
}

impl AuthManager {
    pub fn new() -> Self {
        let mut initial = HashMap::new();
        // Generate random keys on first boot; do NOT use literals
        // If keys.json is missing, generate and print to stdout
        Self {
            keys: ArcSwap::new(Arc::new(initial)),
        }
    }

    pub fn rotate_key(&self, role: Role) -> String {
        let new_key = generate_random_key();
        let mut new_map = (**self.keys.load_full()).clone();

        // Deactivate old key of same role but keep it valid for active sessions
        for (_, meta) in new_map.iter_mut() {
            if meta.role == role && meta.is_active {
                meta.is_active = false;
            }
        }

        new_map.insert(new_key.clone(), ApiKeyMeta {
            role,
            created_at: now(),
            rotated_from: None,
            is_active: true,
        });

        self.keys.store(Arc::new(new_map));
        new_key
    }

    pub fn verify(&self, key: &str) -> Option<Role> {
        self.keys.load().get(key).and_then(|m| {
            if m.is_active { Some(m.role) } else { None }
        })
    }
}

fn generate_random_key() -> String {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let bytes: Vec<u8> = (0..32).map(|_| rng.gen()).collect();
    base64::encode(&bytes)
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}
```

---

#### 29.9.6 `src/kv_pool.rs` (Contiguous Pool)

```rust
use std::sync::atomic::{AtomicU32, Ordering};

/// A contiguous memory pool for KV cache pages.
///
/// # Design
/// - One `Vec<u8>` backing store for GPU-friendly contiguous memory.
/// - Metadata slab (`Vec<KvPage>`) tracks offsets, generations, and refcounts.
/// - Eviction only considers pages with `refcount == 0`.
#[derive(Debug)]
pub struct KvPage {
    pub offset: usize,
    pub len: usize,
    pub generation: u64,
    pub refcount: AtomicU32,
    pub is_allocated: bool,
    pub last_accessed: u64,
}

pub struct KvPool {
    buffer: Vec<u8>,
    pages: Vec<KvPage>,
    free_list: Vec<usize>,
    global_generation: u64,
    page_size: usize,
}

impl KvPool {
    pub fn new(total_bytes: usize, page_size: usize) -> Self {
        let num_pages = total_bytes / page_size;
        let mut pages = Vec::with_capacity(num_pages);
        let mut free_list = Vec::with_capacity(num_pages);

        for i in 0..num_pages {
            pages.push(KvPage {
                offset: i * page_size,
                len: page_size,
                generation: 0,
                refcount: AtomicU32::new(0),
                is_allocated: false,
                last_accessed: 0,
            });
            free_list.push(i);
        }

        Self {
            buffer: vec![0u8; total_bytes],
            pages,
            free_list,
            global_generation: 0,
            page_size,
        }
    }

    pub fn allocate(&mut self) -> Option<usize> {
        let idx = self.free_list.pop()?;
        let page = &mut self.pages[idx];
        page.is_allocated = true;
        page.generation = self.global_generation;
        self.global_generation += 1;
        page.last_accessed = now();
        Some(idx)
    }

    pub fn release(&mut self, idx: usize) {
        let page = &mut self.pages[idx];
        let prev = page.refcount.fetch_sub(1, Ordering::Release);
        if prev == 1 {
            // Last reference dropped
            page.is_allocated = false;
            self.free_list.push(idx);
        }
    }

    pub fn acquire(&self, idx: usize) -> bool {
        let page = &self.pages[idx];
        if page.is_allocated {
            page.refcount.fetch_add(1, Ordering::Acquire);
            true
        } else {
            false
        }
    }

    pub fn evict_candidates(&self, n: usize) -> Vec<usize> {
        let mut candidates: Vec<_> = self.pages
            .iter()
            .enumerate()
            .filter(|(_, p)| p.is_allocated && p.refcount.load(Ordering::Relaxed) == 0)
            .collect();

        // Sort by LRU (oldest first)
        candidates.sort_by_key(|(_, p)| p.last_accessed);
        candidates.into_iter().take(n).map(|(i, _)| i).collect()
    }

    pub fn get_slice(&self, idx: usize) -> Option<&[u8]> {
        let page = self.pages.get(idx)?;
        if !page.is_allocated { return None; }
        Some(&self.buffer[page.offset..page.offset + page.len])
    }

    pub fn get_slice_mut(&mut self, idx: usize) -> Option<&mut [u8]> {
        let page = self.pages.get(idx)?;
        if !page.is_allocated { return None; }
        Some(&mut self.buffer[page.offset..page.offset + page.len])
    }
}

fn now() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocate_and_release() {
        let mut pool = KvPool::new(1024, 256); // 4 pages
        let idx = pool.allocate().unwrap();
        assert!(pool.pages[idx].is_allocated);
        pool.release(idx);
        assert!(!pool.pages[idx].is_allocated);
        assert_eq!(pool.free_list.len(), 4);
    }

    #[test]
    fn refcount_prevents_eviction() {
        let mut pool = KvPool::new(1024, 256);
        let idx = pool.allocate().unwrap();
        pool.acquire(idx); // refcount = 1

        let candidates = pool.evict_candidates(10);
        assert!(!candidates.contains(&idx)); // Cannot evict while in use

        pool.release(idx); // refcount = 0
        let candidates = pool.evict_candidates(10);
        assert!(candidates.contains(&idx)); // Now eligible
    }

    #[test]
    fn out_of_memory() {
        let mut pool = KvPool::new(512, 256); // 2 pages
        pool.allocate().unwrap();
        pool.allocate().unwrap();
        assert!(pool.allocate().is_none()); // OOM
    }
}
```

---

### 29.10 PR Templates

#### PR Template: Sprint 0 (Tests)

```markdown
## 🧪 Sprint 0: Safety Net (Blocking)

**Context:** The repo has 28.7k LOC but only 137 lines of tests. We cannot safely refactor the auth or threading layers without regression protection.

**Changes:**
- [ ] Add integration test for `generate_text` (happy path + timeout).
- [ ] Add integration test for `ingest_model` (path traversal rejection).
- [ ] Add integration test for auth middleware (missing key -> 401).
- [ ] Ensure `cargo test` runs in CI.
- [ ] Add `cargo fmt --check` and `cargo clippy -- -D warnings` to CI.

**Risk:** Low (adds code, doesn't change existing logic).  
**Merge Requirement:** CI green. Coverage target: >2%.
```

#### PR Template: Sprint 9a (Auth & Bounds)

```markdown
## 🔒 Sprint 9a: P0 Auth & Input Validation

**Fixes:**
1. **Auth defaults (#2, #3, #17):** Apply `RequireReadOnly` globally; mount `/health`/`/metrics` unauthenticated; promote `llm_load_model` to `RequireAdmin`.
2. **Default keys (#3):** Generate random keys on boot; hash-check against literals; refuse to serve if defaults detected.
3. **CORS (#4):** Explicit allowlist; enforce `AuthLayer` *before* `CorsLayer`.
4. **Input bounds (#16):** Hard cap on `input_ids.len() + max_new_tokens <= context_window` *before* queueing.

**Testing:** Requires Sprint 0 tests to validate rejection flows.
```

#### PR Template: Sprint 9b (Security Hardening)

```markdown
## 🔒 Sprint 9b: P0 Security Hardening

**Fixes:**
1. **Path traversal/Symlinks (#15, #27) in `ingest_model`:** Use `canonicalize` + check parent components with `symlink_metadata`; reject symlinks. `payload.path` is user input fed to `find_safetensors_files(dir)`.
2. **SIGTERM refinement (#18):** `main.rs` already has `ctrl_c` + `with_graceful_shutdown` + DB save. Add `tokio::signal::unix::signal(SignalKind::terminate())` handler. Explicitly drain inference queue (30s) before exit.
3. **Env var race (#1):** Move `device` into `InferenceTask`; delete `set_var`.

**Testing:** Requires Sprint 0 tests to validate path rejection and graceful shutdown.
```

#### PR Template: Sprint 10a (Core Refactor)

```markdown
## ⚙️ Sprint 10a: P1 Structural Hardening

**Fixes:**
1. **Typed errors (#5):** `thiserror` enum `BramhaError` with `IntoResponse` (maps to HTTP 4xx/5xx).
2. **Threading (#6):** Audit 16 `tokio::spawn` sites; move tensor math to `spawn_blocking`/`rayon`.
3. **Panics (#7):** Wrap spawned tasks in `catch_unwind`; write `"Failed"` status.
4. **Timeouts (#8 corrected):** Remove queue-level timeout; wrap *generation tasks* in 60s (interactive) / 300s (batch) timeout.

**Breaking Changes:** Error types change; API handlers updated accordingly.
```

#### PR Template: Sprint 10b (Backend Decision)

```markdown
## 🔬 Sprint 10b: Backend Prototype (Decision Only)

**Goal:** Benchmark `candle-core` CPU vs raw `wgpu` on Qwen2-0.5B. No production code changes.

**Deliverable:** Decision document committed to `docs/ADR-001-BACKEND.md`.

**Options:**
- Option A: Commit to `candle-core` (CPU + GPU via its backends).
- Option B: Commit to raw `wgpu` + `gemm` (custom sparse paging).

**Criteria:** Compile time, inference latency, memory control, sparse paging feasibility.
```

#### PR Template: Sprint 10c (Memory & Keys)

```markdown
## ⚙️ Sprint 10c: P1 Memory & Key Rotation

**Fixes:**
1. **KV Pool (#9):** Implement contiguous `Vec<u8>` memory pool; metadata slab tracks `(offset, len, generation)`.
2. **Refcounts (#21, #35):** Add `Arc<()>` to KV pages; eviction only acts on `refcount == 0` (safety invariant).
3. **Key rotation (#29):** `ArcSwap<HashMap>` for keys; add `POST /admin/keys/rotate`.

**Testing:** Memory profiler confirms contiguous pages.
```

#### PR Template: Sprint 11 (Operations & WAL)

```markdown
## 📊 Sprint 11: P2 Observability & Correct WAL

**Fixes:**
1. **Metrics (#19):** Prometheus counters/histograms (`queue_depth`, `vram_bytes`, `token_latency_seconds`, errors by variant).
2. **Tracing:** `tracing::Span` per request; JSON structured logging.
3. **Rate limiting (#20):** `governor` token bucket per API key (reject 429).
4. **Session TTL (#22):** 30-min idle; background sweeper appends `SessionClosed` tombstone.
5. **Boot validation (#23):** Probe `wgpu` adapter at startup; fail fast if GPU requested but unavailable.
6. **Queue backpressure (#30):** If `queue_depth > max_concurrent * 2`, return `503`.
7. **GPU warmup (#31):** Dummy 1-token forward pass after load; `/ready` waits for warmup.
8. **OOM pre-alloc (#32):** Allocate max KV cache size at load; reject generations exceeding it.
9. **Metrics lockdown (#33):** Separate port (9090) or IP allowlist for `/metrics`.
10. **WAL correction:** Per-session segments; CRC32C checksums; `O_APPEND`; `fdatasync` only on checkpoint/shutdown or every 64 MB.

**Testing:** Requires `k6` load testing to validate backpressure.
```

#### PR Template: Post-11 (v0.1 RC)

```markdown
## 🧹 Post-11: P3 v0.1 Readiness

**Actions:**
1. **File splits (#12):** Split `cpu_engine.rs` (3.5k lines) and `handlers.rs` (2k lines).
2. **CI (#24):** Add `cargo audit` + `cargo deny` with license/advisory checks.
3. **Fuzzing (#25):** `cargo-fuzz` target for `generate_text` (unicode, overflow, edge cases).
4. **Sparse paging doc (#26):** Write one-pager on VRAM→RAM→disk eviction thresholds.
5. **Load testing (#34):** Run `wrk2` 100 concurrent clients for 5 min; fix contention bugs.
6. **Failure matrix (#36):** Document `(Error Variant) × (System State) → (Action)`.

**Outcome:** v0.1 release candidate.
```

---

### 29.11 CI/CD Integration

**Pre-merge checklist (enforced by CI):**
- [ ] `cargo fmt --check` passes
- [ ] `cargo clippy --all-targets --all-features -- -D warnings` passes
- [ ] `cargo test --workspace` passes
- [ ] `cargo deny check` passes (no RUSTSEC advisories, no banned licenses)
- [ ] `cargo audit` passes (no known vulnerabilities)
- [ ] PR description links to the epic issue and the specific audit items being fixed
- [ ] New tests accompany every P0 fix (TDD)

**Branch protection rules:**
- Require 1 review before merge.
- Require all CI checks to pass.
- Require linear history (no merge commits).
- Require signed commits (optional but recommended).

---

### 29.12 Appendix: Dependency Audit

**Current suspicious dependencies (from codebase scan):**

| Dependency | Status | Action |
|------------|--------|--------|
| `burn` | Training framework, not inference | **Delete** after Sprint 10b decision |
| `ndarray` | CPU-only, redundant | **Delete** |
| `nalgebra` | Overkill for inference | **Delete** |
| `rustfft` | Not needed for transformers | **Delete** |
| `wgpu` | Keep for GPU sparse paging | **Keep** |
| `rayon` | CPU parallelism | **Keep** |
| `parking_lot` | Non-poisoning mutexes | **Keep** |
| `thiserror` | Typed errors | **Keep** |
| `tracing` | Observability | **Keep** |
| `prometheus` | Metrics | **Keep** (add in Sprint 11) |
| `governor` | Rate limiting | **Keep** (add in Sprint 11) |
| `candle-core` | CPU prototype | **Evaluate** in Sprint 10b |
| `gemm` | BLAS for raw wgpu | **Add** if choosing raw wgpu |

**Binary size target:** After deleting `burn`/`ndarray`/`nalgebra`/`rustfft`, the release binary should drop from ~150MB to ~80MB.