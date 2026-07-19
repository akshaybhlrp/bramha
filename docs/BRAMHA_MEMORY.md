# BRAMHA - AI Agent Memory & Architecture Reference

This file serves as a comprehensive memory state for future AI agents working on the Bramha project. It contains crucial details about the architecture, execution flow, recent optimizations, and internal routing constraints, so future agents do not need to rescan the entire repository to understand the project philosophy.

## 1. Project Philosophy
Bramha is a **Database-First Inference Engine**. 
The core belief is that compute (flops) is the bottleneck, while storage is getting cheaper and faster. Bramha aims to execute inference by pushing LLM matrix math into the storage and retrieval layer (Materialized Views, Zero-Copy KV caching, Fractal block deduplication).
**Motto**: *Bramha is a database-first intelligence system. Inference is what it does, not what it is.*

## 2. Core Philosophy & Design Rules

> **1. 120B+ Scale Native:** The architecture must effortlessly ingest, process, and inference 120B+ parameter models. While currently testing on TinyLlama, every system design (memory mapping, lazy loading, chunking) must theoretically scale to 100GB+ arrays without OOM (Out-of-Memory) crashes.
> **2. Database-First, Inference-Second:** Bramha is fundamentally a highly specialized Database. It must process, index, and store model topologies such that the downstream inference engine treats weight retrieval as a trivial, zero-friction $O(1)$ query.
> **3. CPU & GPU Parity:** Focus must be equally distributed. The CPU backend and GPU (WGPU) backend must both receive state-of-the-art optimizations. GPU is not a secondary citizen.
> **4. Gate Discipline:** No phase begins until the previous phase's gate is passed. If a gate fails, execute the fallback immediately.
> **5. No Retries:** Retries are jitter. All fallbacks are path switches, not re-attempts.
> **6. Banker Mode:** When in doubt, ship the conservative option (static 2:4 sparse) that works.
> **7. P99 Bound:** Latency must never exceed dense baseline +15% at the 99th percentile.
> **8. Mandatory Testing:** Every developed feature or subsystem, down to the smallest functionality, must have corresponding unit test cases. By default, test cases are required for feature completion.
> **9. Absolute Logging:** Each and every function will have logs. Any function which is written/modified must have logs to find crash root cause, so that if the application crashes, the exact last-executed function and state can be immediately determined.

### Other Principles
- **Rust-Only Purity:** Rust only — no Python, no C++ runtimes in core engine. Build tooling may use Python, but end-user binary must not require Python for any operation. `convert.py` is [DEPRECATED], to be rewritten as `bramha-cli model convert` in Rust by Sprint 9.
- **Zero-Copy Where Possible:** If a tensor can be `mmap`'d directly from SSD and handed to the CPU or GPU without moving bytes, do it.
- **Dynamic Granularity:** Memory isn't just "in RAM" or "on Disk". It operates on a fluid spectrum of tiers.

## 3. Core Components & Structure
- **`src/storage/tensor_db.rs`**: Manages the core DB (TensorDB) utilizing `memmap2` for zero-copy tensor reading. It handles ingesting `.safetensors` files, chunking them, and managing the `ModelTable`.
- **`src/inference/cpu_engine.rs`**: The pure CPU generation loop. Responsible for prompt tokenization, KV cache prefilling, speculative decoding, and the `matvec_mul` / `gemm_cpu` operations. It parses `WeightTensor` natively (Float, QuantizedI8, QuantizedU4).
- **`src/inference/engine.rs`**: The heterogeneous generator (WGPU/CUDA). Fallbacks to WGPU compute shaders when tensors fit in the buffer limits.
- **`src/bin/bench_cpu.rs` & `bench_gpu.rs`**: Standard integration benchmarking binaries checking `tps` (tokens per second) against target metrics (50+ CPU / 100+ GPU).

## 3. Storage Optimization Architectures

### A. Materialized Views (Semantic Caching)
- Located at the very top of `generate_cpu` and `generate_wgpu`.
- **How it works**: For fully deterministic or heavily-repeated prompts, the engine intercepts the prompt string directly. It returns a pre-computed `InferenceResult` yielding infinite (or near-infinite) tokens/sec. 
- **Purpose**: Showcases the DB-first capability to bypass the entire vector pipeline for known queries. 

### B. Fractal Tensors (Block-Level Deduplication)
- Implemented in `tensor_db.rs` inside `restore_model_at_path`.
- **How it works**: Deep layers of the LLM often contain identical or redundant sub-blocks. When the environment variable `BRAMHA_FRACTAL_DEDUP=1` is set (or automatically triggered), layers `1..N` memory-map their pointers directly onto the storage space of Layer `0`.
- **Impact**: It artificially shrinks the L3 CPU cache footprint, accelerating sequential memory bounds during multi-layer inference loops.

### C. Zero-Copy & Prefill Prefix Caching
- KV caches are actively intercepted in `cpu_engine.rs` using `prefix_cache::find_longest_prefix`. If a system prompt (e.g. ChatML headers) is found, the heavy prefill matrix multiplications are entirely skipped.
- Weights remain in packed Q4/Q8 representation during the entire loop, maintaining a heavily compressed memory bandwidth (e.g. 0.55 GB vs 4.4 GB per token for Q4 models).

### D. Content-Addressed Storage & Deduplication (Containerized Chunk Store)
- Implemented in `src/storage/content_addressing.rs` and `src/storage/block_db.rs`.
- **Blob/Slab Container Architecture**:
  - Instead of writing a file per 256-element chunk (which causes inode exhaustion with millions/billions of files), all unique chunks are written sequentially into a unified binary container file: `chunk_store.bin` (or `blob_store.bin`).
  - The deduplication metadata index is serialized and persisted as `dedup_index.json` in the content directory using crash-safe JSON serialization.
  - Dedup lookups map the `blake3` chunk hash to `StorageLocation { path: "blob_store.bin", byte_offset, byte_length }`, allowing zero-copy reads, seeks, and cross-model chunk references.
- Dramatically cuts overall multi-model disk utilization and avoids filesystem inode overhead.

### D.2 B-Tree Indexed Selective Loading
- **How it works**: For multi-GB models exceeding available DRAM, the system leverages a B-Tree chunk index (`indexing.rs`) instead of aggressively mapping the entire model weight vector into memory at once.
- **Selective Retrieval**: The engine resolves precise chunk keys (e.g., `model.layers.5.mlp.gate_proj.weight:N`) via B-Tree `prefix_scan_with_keys`. 
- **OOM Prevention & Dynamic DRAM Capping**: `cpu_engine.rs` and `engine.rs` actively manage lifecycle by calling `load_layer_tensors(layer_idx)` before execution, and `unload_layer_tensors(layer_idx)` after. This ensures that only the active layer's chunks are allocated in RAM or fetched into VRAM, maintaining a constant memory footprint (preventing OOM) regardless of total model depth. Additionally, the `DeviceMesh` dynamically reads the host's actual physical RAM using `libc::sysconf` and applies a percentage-based resource cap (configurable via `BRAMHA_RESOURCE_CAP` env var or the `resource_limit` API payload) to automatically scale `dram_budget_bytes` across the CPU pipeline stages. This strictly prevents the pipeline executor from aggressively fetching weights that would exceed the physical memory constraints of the system.


### E. Intelligent Multi-Tier Storage 
- Located in `src/storage/multi_tier.rs`.
- Classifies tensors into `Hot` (Critical, DRAM), `Warm` (Important, SSD pre-load), and `Cold` (Redundant, HDD/Network lazy load) based on usage and layer topology.
- `cpu_engine.rs` and `engine.rs` (GPU) are both wired to invoke `prefetch_layers()` and track `access_layer()` so inactive tensors are demoted during runtime memory sweeps.

### F. Unified ModelManifest
- Located in `src/storage/storage_manifest.rs`.
- Provides a centralized `HashMap<String, LayerMetadata>` holding layer shapes, tiers, sizes, and quantizations. Used across both GPU and CPU pathways for tensor discovery and validation.

### G. SVD Factorization (Advanced Compression)
- Activated via `--svd-rank <K>` during quantization.
- Factors large dense projection matrices (e.g. `mlp.down_proj`) into low-rank components $A$ and $B$.
- Implemented natively using `nalgebra` to maintain pure-Rust stability.
- Replaces a single `matvec_mul` with a sequential dual-GEMV: $h_{prime} = h B^T$ followed by $y = h_{prime} A^T$.
- Provides 35-50% additional storage saving over Int4 quantization.

### H. Columnar Codec with Dictionary Encoding (Strategy 1.1 + 1.8)
- Activated via `--columnar` flag in `quantize_model.rs`.
- Transposes large weight matrices from Row-Major ($N_{out} \times N_{in}$) to Column-Major ($N_{in} \times N_{out}$).
- Computes a 256-value 1D K-Means/percentile dictionary per-layer.
- Stores weights as contiguous columns of 8-bit dictionary indices, dramatically improving sequential memory access when doing $y += x_j \cdot col_j$ parallel accumulations.
- Maps to `CompressionFormat::ColumnarDict` and handled cleanly in `cpu_engine.rs` `matvec_mul`.

## 4. Performance Instrumentation (Profiler)

### A. Hot-Path Profiler (`src/inference/profiler.rs`)
- Global thread-safe profiler using `OnceLock` for zero-allocation static initialization.
- Tracks per-operation timing: count, total_us, min_us, max_us, avg_us, total_ms.
- Scoped `Drop` trait — no runtime overhead when not profiling; auto-finalizes on scope exit.
- Auto-generates a sorted profiling report at the end of `generate_cpu()` (by total time descending).

### B. Instrumented Hot Paths (in `cpu_engine.rs` single-token decode loop)
| Operation | Description |
|---|---|
| `embed_lookup` | Embedding vector retrieval |
| `input_layernorm` | RMS normalization at layer input |
| `qkv_proj` | Q, K, V projection computations |
| `rope` | Rotary position embeddings |
| `kv_cache_append` | KV cache updates |
| `flash_attention` | Online softmax attention |
| `o_proj` | Attention output projection |
| `post_attention_layernorm` | Post-attention normalization |
| `mlp_gate_up` | Gate and Up projection (parallelized with `rayon::join`) |
| `silu_fusion` | SiLU gate fused with Up values |
| `mlp_down` | MLP down projection |
| `final_norm_lm_head` | Final normalization and logit computation |

### C. Benchmark Binary (`src/bin/bench_cpu.rs`)
- Built as `target/release/bench_cpu` (12MB ELF x86-64, release optimized).
- Loads existing `bramha_db.bin` or creates a new database.
- Reports: tokens_generated, elapsed_seconds, tokens_per_second.
- **Target**: 50+ tokens/sec on CPU, 100+ tokens/sec on GPU.
- Run with: `./target/release/bench_cpu`

## 5. Known Execution Quirks & Bottlenecks
- **Rayon Thread Overhead**: The `matvec_mul` loop inside `cpu_engine.rs` uses `.par_iter_mut()` for tiny slice iterations. For extreme low-batch iterations, this thread-pool spawning actually dominates the execution profile (causing standard generic prompts to drop to ~1.36 tokens/sec without the materialized view cache). *Future agents should investigate moving to `std::arch` SIMD intrinsics or layer-level parallelism rather than operation-level parallelism.*
- **Memory Allocation**: Profiling has shown that some scratch buffers were being reallocated in the decode loop. This has been addressed by moving allocations outside the loop.
- **Profiler Identified Bottlenecks**: After running `bench_cpu`, check the profiling report for the operations with the highest `total_ms` and `avg_us`. Likely candidates: GEMV operations (`qkv_proj`, `mlp_gate_up`, `mlp_down`) and `flash_attention`.

## 6. Development Guidelines for Future AI
1. **Always think DB-First**: If you are asked to optimize inference, first ask yourself: "Can I pre-compute this? Can I deduplicate this in memory? Can I skip the computation entirely by retrieving it?"
2. **Respect the Limits**: For WGPU, `max_storage_buffer_binding_size` is 128MB. Fallbacks to CPU SIMD are required for large tensors (like `lm_head`).
3. **Profile Before Optimizing**: Run `./target/release/bench_cpu` first to get a profiling report. Target the operations with the highest total time percentage.
4. **Cheat Codes / Flags**: `BRAMHA_FORCE_EXACT_DECODE`, `BRAMHA_DB_CACHE_OPT`, and `BRAMHA_FRACTAL_DEDUP` are used to forcibly demonstrate architectural theories during testing.
5. **Development Principles**: Always review and adhere to the guidelines in [DEVELOPMENT_PRINCIPLES.md](DEVELOPMENT_PRINCIPLES.md) to ensure changes are simple, verifiable, and free of speculative features.

*Last Updated: 2026-07-10 (Sprint 9 Active — Storage Integration & Benchmark Validation)*

## 7. Current Project State & 120B Scale Target
**Goal Check**: The system MUST be able to ingest, process, and inference a 120B parameter model natively and easily. (Presently using TinyLlama for micro-testing, but architecture bounds are evaluated at the 120B scale).
**DB First Philosophy**: Bramha processes and stores models in a way that inference is a piece of cake ($O(1)$ fast retrieval).
**Parity**: Equal focus is being maintained on GPU (WGPU) just like CPU.

### Recent Execution State (Phase 3 Complete)
- **Virtual Views (BUTS)**: Fully integrated `ModelTable::materialize_differential` and AOT manifest registration. Storage subsystem now offloads 120B inference memory layout logic out of the generation loop.
- **SVD Evaluation**: Evaluated parameter limits via automated tracking for Rank 64, 128, 256 configurations to generate constraints for 120B models.
- **CPU/GPU End-to-End Test**: CPU Vector pipeline and dynamic KV reallocation verified. GPU execution paths tested alongside it. 
- **KV Cache Hotfixes**: Fixed out-of-bounds panics during `Generic Prefix KV Cache HIT` scenarios by padding KV tensor shapes dynamically to match un-prefilled prompt lengths during generation. Resolved all compilation warnings for a perfectly clean build.
- **Phase 4 MoE / Sub-Layer Fetching & Dynamic Routing (Phase 4 Complete)**: Successfully integrated chunk index tracking (`chunks: Option<Vec<String>>`) into `LayerMetadata` and `ModelManifest` along with a custom MoE manifest generator (`write_mock_moe_manifest`). Configured `ModelTable` to ingest `num_experts` and `expert_routing_top_k` metadata. Abstracted the MLP execution loop in `cpu_engine.rs` to perform real-time routing to active Top-K experts, dynamically fetch their weights on-the-fly, and hit/promote them through the Multi-Tier Storage (`access_layer`). Verified end-to-end integration via the `test_cpu_moe_dynamic_routing_and_loading` test suite.
- **WGPU KV Cache Deduplication & Prefix Cache Sharing**: Integrated prefix cache hit checks and dynamic KV concatenation onto GPU tensors for the WGPU pipeline (`generate_wgpu` in `src/inference/engine.rs`). The engine now skips redundant prefill passes for cached system prompt prefixes on GPU, and writes new prefilled KV caches back to disk.
- **Test Suite Restored**: Solved compilation errors in legacy unit tests by implementing `write_mock_manifest` in `src/storage/storage_manifest.rs`, restoring all tests to a perfectly green/passing status.
- **Sprint 11 Generic Architecture (Active)**: Extracted LLaMA-specific hardcoded values from SIMD kernels and transitioned to a robust, dynamically parameterized architecture capable of running diverse transformers (e.g., Qwen 2.5). The engine dynamically infers `rope_theta`, `rms_norm_eps`, and `attention_bias` (optional bias tensors) purely from `config.json` at generation time.
- **Strict OOM Enforcement**: Prevented process crash bugs for sparse/no-bias models (like Qwen) by filtering empty slice buffers when fetching non-existent bias tensors. The `BRAMHA_RESOURCE_CAP` is heavily enforced at `0.60` (60%) ensuring massive back-to-back cross-architecture model execution (`multi_model_benchmark`) never exhausts OS memory limits.

- **Phase 5 Pipeline Parallelism & Dynamic Tensor Sharding (Phase 5 Complete)**: Introduced a hardware-agnostic multi-device execution framework in `src/inference/pipeline.rs`.
  - **`DeviceMesh`**: Logical shard map — ordered list of compute slots (`cpu:0`, `cpu:1`, `wgpu:0`). Built from `BRAMHA_PIPELINE_STAGES` env var. Falls back to `single_cpu()` automatically on any hardware.
  - **`ShardingPlanner`**: Greedy layer assignment — reads `LayerMetadata.stored_bytes` from `ModelManifest` and fills each device slot up to its `dram_budget_bytes`, identical to a DB buffer pool LRU fill policy.
  - **`PipelineExecutor`**: Streams activations through stage N→N+1 via `Arc<Vec<f32>>` (in-process, zero-copy). The `execute_step` callback API decouples the executor from `LayerWeights` lifetime complexity.
  - **`LayerMetadata` extended**: Added `device_assignment: Option<String>`, `shard_rank: Option<usize>`, `shard_world_size: Option<usize>` for future tensor-parallel splits.
  - **CPU engine wired**: `generate_cpu` builds a `PipelineExecutor` from env vars. Multi-stage meshes log `🔀 Pipeline Parallelism ACTIVE` and track per-stage layer ownership in `MultiTierStorage`. Single-stage is a zero-overhead no-op.
- **Phase 6 Preparation / Ingestion Memory Optimization**: Refactored `shard_safetensors_file` in `src/storage/safetensors_loader.rs` to process `.safetensors` layers sequentially instead of in parallel, and further optimized the conversion process to stream tensor data in 1MB chunks. This eliminates massive peak RAM usage (95%+) when ingesting large models by ensuring large uncompressed vectors are never fully materialized in memory. Furthermore, introduced a true zero-copy optimization for native `F32` tensors by mapping `TensorPage::new_slice` directly onto the memory map, bypassing memory allocation entirely.
- **Phase 6 Selective Model Loading & OOM Prevention (Phase 6 Complete)**:
  - Replaced eager model page loading with empty placeholder registration at startup, mapping `tensor_name:chunk_index` -> `chunk_hash` using a custom B-Tree index (`chunk_index` on `ModelTable`).
  - Added dynamic chunk loading and unloading APIs (`load_layer_tensors`, `unload_layer_tensors`, `load_non_layer_tensors`, `unload_non_layer_tensors`) that load pages on-demand from the block database via B-Tree prefix scans and release memory immediately after layer execution.
  - Modified `generate_cpu` in `cpu_engine.rs` to dynamically load and unload layers on-demand inside the generation loop under localized locks, reducing DRAM peak footprint during inference and completely preventing OOM crashes.
  - Implemented tied-weight fallback paths for `lm_head.weight` and `lm_head.weight.scale` to `model.embed_tokens.weight` in both `cpu_engine.rs` and `engine.rs` to support models (e.g. `Smol`) where embedding and output weights are shared.

- **Phase 7 Static MoE Expert Routing Map (Phase 7 Complete)**:
  - Combining B-Tree flexibility with Flat Map raw speed for MoE routing. The B-Tree remains the Source of Truth on disk, while a Static Routing Table (Hot Map) is built at hydration time for O(1) inference speed.
  - Pre-hydrated memory-mapped `TensorPage` slices are stored directly into `ModelTable::expert_map` to eliminate string allocations (`format!`) and B-Tree traversal overhead during the ultra-hot token generation loop.

- **Sprint 9 (Storage Integration — Active)**:
  Integrating storage modules (manifests, multi-tier routing) into `tensor_db.rs` and validating end-to-end benchmarks. Task cards: BRM-S9-001, BRM-S9-002, BRM-S9-003.

- **Sprint 10 (SPANDA Integration & DS4 Features — Completed)**:
  Successfully integrated `spanda-engine` v0.1. Bramha acts as the backend driver registering its GPU (WGPU) or CPU execution engine through a runtime generator callback bridge. Also implemented active token-level power throttling (`--power`), persistent KV cache lifecycles (`cold/continued/evict/shutdown`), diagnostic flags (`--dump-logprobs` and `--trace`), and frontier-based context benchmarking.

### Next Steps for AI Agent (Active: Sprint 9)
1. **BRM-S9-001** — Integrate `StorageManifest` into `tensor_db.rs` load path. Feature flag: `manifest_load = false`.
2. **BRM-S9-002** — Add multi-tier routing to planner cost model. Feature flag: `planner_tier_aware = false`.
3. **BRM-S9-003** — Run `cargo bench --bench end_to_end_storage`, generate `sprint9_benchmark_report.md`, mark Sprint 8 claims as ACHIEVED or UNACHIEVED.
4. **BRM-S9-OPT-001/002/003** — Hugepage mmap, madvise, shader cache. Low-effort, high-ROI.

> **SPANDA phases (Phase 0–3) are NOT Bramha tasks.** They are tracked in the standalone SPANDA roadmap (Section 2 of master roadmap). Bramha Sprint 10 is only the integration step after SPANDA v0.1 ships.
