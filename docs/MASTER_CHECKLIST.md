# 🧭 Bramha & SPANDA Master Task Tracker

This document serves as the master source of truth for tracking the completion of execution cards across the Bramha Neural Engine and SPANDA projects. It lists all tasks, marks their completion status, and provides AI-driven suggestions and optimizations for the next steps.

---

### **I. Bramha v0.1 Ship Criteria**

These items define the minimum viable product for the Bramha engine.

- [x] **Qwen2-0.5B runs end-to-end on CPU and wgpu**
    - **Verification:**
        - `src/inference/cpu_engine.rs`: Complete, optimized CPU inference path.
        - `src/compute/wgpu_backend.rs`: Complete GPU path with persistent buffers and async execution.
        - `src/inference/engine.rs`: Logic to switch between backends.

- [x] **CRUD for collections, documents, sessions, and models**
    - **Verification:**
        - `src/core/collection.rs`: `insert` and `delete` methods.
        - `src/storage/mod.rs`: `save` and `load` logic for collections/models.
        - `src/cognitive/memory.rs`: `MemoryManager` for memory CRUD.
        - `src/storage/wal.rs`: `WalOp` with `Upsert`/`Delete` for transaction logging.

- [x] **WAL replay and atomic writes proven**
    - **Verification:**
        - `src/storage/wal.rs`: `WalManager` with `replay` function integrated into DB loading.
        - `src/cognitive/memory.rs`: Atomic write pattern (write-then-rename) used for safe persistence.

- [x] **Tokenizer fully in-process, model registry working**
    - **Verification:**
        - `src/inference/tokenizer.rs`: Pure Rust `BramhaTokenizer` wrapper.
        - `src/storage/mod.rs`: Model registry populated from `tensor_db` during save.

- [x] **Retrieval (IVF/HNSW/BM25) with evidence grounding**
    - **Verification:**
        - `src/index/ivf_flat.rs`, `src/index/hnsw.rs`, `src/index/bm25.rs`: Full index implementations.
        - `src/core/collection.rs`: `hybrid_search` with RRF for combining results.
        - `src/cognitive/evidence.rs`: Sentence-to-source mapping logic.

- [ ] **Memory DB + graph DB functional**
    - **Status:** Partially Done
    - **Verification:**
        - `src/cognitive/memory.rs`: Sophisticated multi-tier `MemoryManager` is functional (Memory DB).
        - `src/cognitive/research.rs`: `ResearchGraph` is a stub; multi-hop execution is not implemented (Graph DB).

- [x] **Planner with exact-decode-only path**
    - **Verification:** `src/planner/optimizer.rs`: `ExecutionPathOptimizer` correctly returns `PlannerDecision::ExactDecode` based on policy.

- [ ] **Storage manifest + dedup integrated**
    - **Status:** Done but unverified
    - **Verification:**
        - Code is fully implemented in `src/storage/storage_manifest.rs`, `content_addressing.rs`, `multi_tier.rs`, and integrated into `tensor_db.rs`.
        - Performance claims (e.g., 92-96% DRAM reduction) are explicitly marked as "UNVALIDATED" in `docs/sprint9_benchmark_report.md`.

---

### **II. SPANDA Sparse Inference Backend**

- [x] **Phase 0: Bare Sparse Paging / Static Sparse Validation**
    - **Verification:** `matvec_mul_sparse` in `wgpu_backend.rs` confirms sparse capabilities.

- [x] **Phase 1: RAM Offload Fallback**
    - **Verification:** `wgpu_backend.rs` includes degradation state handling and fallback to dense computation, aligning with the offload strategy.

- [x] **Phase 2: 4-Bit Logarithmic Quantization & Trajectory Prefetch**
    - **Verification:** `cpu_engine.rs` contains kernels for `QuantizedU4` data types and references a `Prefetcher`.

- [x] **SPANDA Integration**
    - **Verification:** `src/planner/optimizer.rs` includes `PlannerDecision::SpandaSparse`, confirming it is a selectable execution path.

---

### **III. Other Notable Features**

- [ ] **Adaptive Learning**
    - **Status:** Partially Done
    - **Verification:** Memory confidence updates and contradiction resolution are implemented in `src/cognitive/memory.rs`. Adapter learning is foundational but not fully mature.

- [ ] **Activation Materialized Views (Sprint 7)**
    - **Status:** Done but deferred
    - **Verification:** `src/planner/optimizer.rs` integrates logic for `has_activation_view`, but the feature is deferred to v0.5 in the roadmap.

---

## 🚀 What's Next? (AI Suggestions & Optimizations)

With the core engine largely complete, focus should shift to validation, performance tuning, and maturing partially implemented features.

### 1. 🎯 Complete and Validate Partially Implemented Features
- [ ] **Graph DB Implementation:**
    - Flesh out the `ResearchGraph` in `src/cognitive/research.rs`.
    - Implement multi-hop graph traversal for complex queries.
- [ ] **Benchmark Storage Layer:**
    - Create a new, reliable benchmark test to validate the performance claims of the storage manifest and deduplication features, as the original benchmarks in `sprint9_benchmark_report.md` were inconclusive.
- [ ] **Mature Adaptive Learning:**
    - Move beyond memory-level adaptations and implement full adapter learning pipelines for model tuning.

### 2. 🌐 Distributed & Serverless Features (Bramha Hyperscale)
- [ ] **Distributed Layer Splitting (Sprint 12+):**
    - **BRM-NET-001:** Define `proto/bramha.proto` for remote tensor execution.
    - **BRM-NET-002:** Implement gRPC server in `src/network/server.rs`.
    - **BRM-NET-003:** Implement `RemoteBackend` client.
- [ ] **WebRTC P2P Intelligence:**
    - **BRM-NET-004:** Create `GossipWorker` in `src/network/gossip.rs`.
    - **BRM-NET-005:** Implement KV cache synchronization protocol.

### 3. 🗜️ Advanced Model Quantization & Format Polish
- [ ] **Run `spanda-convert` on a real model:** Convert `models/all-MiniLM-L6-v2/model.safetensors`.
- [ ] **Implement SVD Factorization Module:** Implement randomized SVD for FFN matrix decomposition.
- [ ] **Implement Differential / Delta Compression:**
    - **BRM-DELTA-001:** Core differential tensor module in `src/storage/differential_compression.rs`.
    - **BRM-DELTA-002:** Delta storage manager integrated with `StorageManifest`.
    - **BRM-DELTA-003:** Decompression integration into `TensorDB` loader.

### 4. ⚡ VRAM Management & Engine Stress Testing
- [ ] **BRM-OOM-001:** High-concurrency load simulator (15+ requests).
- [ ] **BRM-OOM-002:** Verify graceful RAM offload under VRAM saturation.
- [ ] **BRM-OOM-003:** Complete `oom_stability_validation.rs` test suite.
- [ ] **BRM-CACHE-001/002/003:** Verify WGPU pipeline cache persistence and integration.

---

*Instructions for AI Assistant: Refer to this document at the start of any new session. If an objective is completed, check the box. If the user pivots, add new tasks to the "What's Next" section to maintain a continuous, traceable line of effort.*
