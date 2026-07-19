# Bramha — Local-First Intelligence Database

## What Is Bramha

Bramha is a **Rust-native, single-binary, local-first intelligence database** for consumer hardware. It unifies high-performance LLM inference, retrieval, memory, adaptive learning, and multi-model orchestration into one system.

> **What SQLite did for local data, Bramha does for local intelligence.**

Bramha is **not** an inference engine with a database attached.
Bramha **is** a database-native intelligence execution system.

---

## Architecture at a Glance

```
Client Layer
    ↓
Transport Layer (HTTP + UDS)
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

### Crate Boundary

| Crate | Role |
|-------|------|
| `bramha` | Monolithic intelligence database containing all modules |
| `spanda-engine` | Standalone sparse inference backend (consumed as a versioned workspace dependency) |

SPANDA ships independently. Bramha pins a specific `spanda-engine` release and wires it in via `BramhaBackend` trait.

---

## Project Structure

```text
bramha/
├── Cargo.toml
└── src/
    ├── api/              # Axum HTTP API & Handlers
    ├── cognitive/        # Memory, learning, and high-level intelligence
    ├── compute/          # CPU SIMD, wgpu shaders
    ├── concurrency/      # Threading, async/blocking bridge
    ├── core/             # Core data structures (Tensor, etc.)
    ├── index/            # IVF/HNSW/BM25 indexing
    ├── inference/        # Engine, CPU/wgpu backends, SPANDA integration
    ├── middleware/       # Auth, queueing, etc.
    ├── models/           # Model loading and quantization
    ├── network/          # P2P gossip and client logic
    ├── planner/          # Cost model, optimizer, policies
    ├── storage/          # Manifest, content-addressing, multi-tier, TensorDB
    └── bin/              # Crate binaries (serve, ingest, etc.)
```

---

## Core Capabilities

### Intelligence CRUD
Bramha extends CRUD from data to intelligence state: collections, documents, chunks, embeddings, memories, entities, relations, activation views, answer caches, adapters, routing policies, planner state, and execution traces.

### Inference Planner
Every request goes through a planner that selects the cheapest safe execution path:
exact decode → speculative decode → activation replay → cached answer → multi-model pipeline → degraded fallback.

### Storage Foundation (Sprint 8 — FOUNDATION COMPLETE)
- `StorageManifest` — per-layer metadata, tier classification, compression tracking
- `ContentAddressedStorage` — Blake3-based deduplication, cross-model weight sharing
- `MultiTierStorage` — DRAM/SSD/HDD routing with promotion, demotion, and prefetching

**Performance targets (PROJECTED — pending Sprint 9 benchmark validation):**
- Storage reduction: 50-80%
- DRAM reduction: 92-96%
- Model load time: 500ms → 50ms
- First token latency: 1.2s → 300-400ms

### SPANDA Sparse Inference Backend (v7 Architecture)
**The Memory Wall Problem**: LLM inference on consumer hardware is constrained by memory bandwidth and VRAM capacity, not compute. Large models cannot fit in VRAM, and loading them token-by-token from storage is too slow.
**SPANDA's Solution (Query-Conditional Sparse Paging)**: SPANDA introduces a database-native approach to inference. Instead of keeping the whole model in memory, it uses query-conditional sparse paging—loading only the necessary weight "pages" (experts) dynamically based on the query's path through the model. It includes:
- **VRAM Page Caching & RAM Offload**: Keeps hot experts in VRAM and gracefully falls back to L3 host RAM offloading when limits are reached.
- **Optional Quantization & Prefetch**: 4-bit logarithmic quantization and trajectory prefetching to hide memory latency and maximize generation speed.

### Retrieval & Evidence
IVF/HNSW/BM25 hybrid retrieval, evidence mapping, citation grounding, multi-hop graph retrieval.

### Memory System
Working memory (session-bound) → Episodic memory (completed interactions) → Semantic memory (stable promoted facts). Temporal decay with half-life, contradiction detection, confidence scoring.

---

## v0.1 Ship Criteria (Single-Model Local Intelligence Database)

- [ ] Qwen2-0.5B runs end-to-end on CPU and wgpu
- [ ] CRUD for collections, documents, sessions, and models
- [ ] WAL replay and atomic writes proven
- [ ] Tokenizer fully in-process, model registry working
- [ ] Retrieval (IVF/HNSW/BM25) with evidence grounding
- [ ] Memory DB + graph DB functional
- [ ] Planner with exact-decode-only path (Sprint 7 deferred to v0.5)
- [ ] Storage manifest + dedup integrated (Sprint 9 complete)

**Not in v0.1:** multi-model, SPANDA, adaptive learning, distributed.

---

## Building & Running

```bash
cd /home/akshay-bhalerao/.gemini/antigravity/scratch/bramha

# Build (release)
cargo build --release

# Convert a model to the SPANDA sparse format (Required before ingestion)
spanda-convert ./models/Qwen2-0.5B.safetensors -o ./models/Qwen2-0.5B.spanda

# Ingest a model
./target/release/bramha ingest ./models/Qwen2-0.5B.spanda

# Start the server
./target/release/bramha serve --port 8000

# Run storage benchmark
cargo bench --bench end_to_end_storage
```

---

## Architecture Invariants (Non-Negotiable)

| Invariant | Rule |
|-----------|------|
| **Rust-only** | No Python in the runtime binary. `convert.py` is deprecated; the canonical tool is `spanda-convert`. |
| **CPU fallback complete** | Every feature works CPU-only. wgpu accelerates, never gate-keeps. |
| **Exact fallback** | Every optimization path has a safe exact fallback. |
| **Gate discipline** | No phase begins until the previous gate passes. |
| **No retries** | Fallbacks are path switches, not re-attempts. |
| **P99 Bound** | Latency must never exceed dense baseline +15% at P99. |
| **Banker Mode** | When in doubt, ship the conservative option that works. |

---

## Active Sprint

**Sprint 9 — Storage Integration & Advanced Compression**

| Task | Status |
|------|--------|
| BRM-S9-001: Manifest integration into `tensor_db.rs` | [ ] Open |
| BRM-S9-002: Multi-tier routing in inference planner | [ ] Open |
| BRM-S9-003: End-to-end storage benchmark | [x] Done |
| BRM-S9-OPT-001: Hugepage-backed mmap | [ ] Open |
| BRM-S9-OPT-002: madvise sequential + willneed | [ ] Open |
| BRM-S9-OPT-003: Shader pipeline cache | [ ] Open |

Full task cards in the master roadmap (Section 19, Sprint 9).

---

## Roadmap

See [Bramha Neural Engine — Master Roadmap v8.0.md](./Bramha%20Neural%20Engine%20%E2%80%94%20Master%20Roadmap%20v8.0.md) for the authoritative roadmap including:
- SPANDA standalone roadmap (Section 2)
- Integration contract (Section 3)
- Sprint plan with task cards (Section 19)
- Architecture invariants (Section 20)
- Success criteria by version (Section 21)

For detailed SPANDA architecture and design, see:
- [SPANDA Design](./docs/SPANDA_Design.md)
- [SPANDA Integration](./docs/SPANDA_Integration.md)

---

*Last updated: 2026-07-10*
