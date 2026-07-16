# Sprint 2 Execution Cards: Fast Local Inference

This document tracks the execution cards for Sprint 2, which aims to build a high-performance local inference capability across CPU and wgpu backends.

---

### Task ID
BRM-S2-001

### Title
CPU backend baseline

### Objective
Implement a pure Rust, zero-GPU inference engine serving as the canonical fallback and baseline, targeting >20 tokens/second on standard hardware.

### Scope
- `src/inference/cpu_engine.rs`
- `src/inference/engine.rs`

### Non-Goals
- Do not add GPU or wgpu support.
- Do not implement quantized paths yet.

### Dependencies
- BRM-S1-004 (In-process tokenizer).

### Inputs
- Dense tensor weights from disk.

### Steps
1. Implement the standard transformer forward pass (Attention + FFN) natively.
2. Ensure naive KV caching is functional.
3. Validate output correctness against known baseline models (e.g., TinyLlama).

### Outputs
- `cpu_engine.rs` implementation.
- Basic correctness tests.

### Acceptance Criteria
- [x] Engine successfully generates coherent text on CPU only.
- [x] Test suite includes a speed enforcer ensuring >20 TPS on a mock/tiny model.

### Failure Modes
- If CPU inference encounters NaNs, fail loudly.

### Rollback
- N/A.

### Tests
- Benchmark: Inference speed enforcer test.

### Regression Risks
- Slow fallback if not carefully written.

### Notes
- "CPU is the canonical backend."

---

### Task ID
BRM-S2-002

### Title
SIMD Optimization

### Objective
Enhance the pure CPU backend with loop unrolling, chunking, and memory-layout improvements to maximize SIMD utilization on `target-cpu=native`.

### Scope
- `src/inference/cpu_engine.rs`

### Non-Goals
- Do not use inline assembly.
- Do not write manual AVX512 intrinsics (rely on Rust auto-vectorization).

### Dependencies
- BRM-S2-001 (CPU Backend).

### Inputs
- Existing naive GEMM/GEMV loops.

### Steps
1. Refactor matrix-vector multiplication to process data in chunks (e.g., blocks of 16 or 32).
2. Ensure arrays are laid out contiguously for the inner loop.
3. Add compile flags for `native` target.

### Outputs
- Optimized loops in `cpu_engine.rs`.

### Acceptance Criteria
- [x] Performance of CPU inference improves measurably over naive baseline.
- [x] Output remains bit-for-bit identical to unoptimized path.

### Failure Modes
- N/A (Standard Rust arrays).

### Rollback
- Revert loop changes to naive iteration.

### Tests
- Unit: Correctness comparison between naive and optimized GEMV.

### Regression Risks
- Cache thrashing if chunk size mismatches L1/L2.

---

### Task ID
BRM-S2-003

### Title
Speculative Decode Pipeline

### Objective
Implement a speculative decoding path using a draft/target model pair to increase tokens-per-second without sacrificing target model quality.

### Scope
- `src/inference/speculative/`
- `src/inference/engine.rs`

### Non-Goals
- Do not implement statistical proposal injection yet.

### Dependencies
- BRM-S2-001 (CPU Backend).

### Inputs
- Target model and smaller Draft model.

### Steps
1. Implement draft model generation loop (produce N tokens).
2. Implement target model verification pass (evaluate N tokens in parallel).
3. Implement rejection sampling to accept/reject drafted tokens.

### Outputs
- Speculative decode orchestrator.
- Rejection sampling logic.

### Acceptance Criteria
- [x] Output matches exactly the distribution of the target model alone.
- [x] Speedup is achieved on matching prompt/model pairs.

### Failure Modes
- If draft model fails, safely degrade to exact decode.

### Rollback
- Config flag `enable_speculative=false` forces exact decode.

### Tests
- Integration: Compare speculative output with exact decode output for identical seed.

### Regression Risks
- Latency regression if acceptance rate is too low.

---

### Task ID
BRM-S2-004

### Title
wgpu Compute Plane Setup

### Objective
Introduce the portable `wgpu` accelerator backend to support GPU inference across Windows (DX12), macOS (Metal), and Linux (Vulkan).

### Scope
- `Cargo.toml`
- `src/compute/wgpu_backend.rs`
- `src/compute/shaders/`

### Non-Goals
- Do not replace the CPU backend.
- Do not implement flash attention in wgpu yet.

### Dependencies
- BRM-S2-001 (CPU Backend).

### Inputs
- `wgpu` crate.

### Steps
1. Initialize wgpu Instance, Adapter, Device, and Queue.
2. Write basic WGSL compute shaders for matrix multiplication.
3. Implement buffer uploads and dispatches.
4. Integrate with the Inference Engine via dynamic dispatch or traits.

### Outputs
- `wgpu_backend.rs`.
- `gemm.wgsl` (basic).

### Acceptance Criteria
- [x] System automatically selects a wgpu adapter if available.
- [x] Simple forward pass computes correctly using wgpu.
- [x] Equivalent output to CPU backend.

### Failure Modes
- If adapter fails to initialize, immediately fallback to CPU backend.

### Rollback
- Pass `--disable-gpu` to force CPU only.

### Tests
- Unit: Matrix multiplication on GPU matches CPU exact result.

### Regression Risks
- Silent corruption due to shader floating-point discrepancies.

---

### Task ID
BRM-S2-005

### Title
Prefix KV Cache

### Objective
Implement a paged KV cache that identifies and reuses prompt prefixes to reduce Time-To-First-Token (TTFT) for repeated workflows.

### Scope
- `src/inference/paged_kv/`
- `src/storage/cache_db.rs`

### Non-Goals
- Do not implement long-term activation views (Sprint 7).

### Dependencies
- Paged KV foundational structure.

### Inputs
- Incoming prompts and their token hashes.

### Steps
1. Implement a rolling hash tree for token prefixes.
2. Separate KV blocks by prefix tree node.
3. On incoming prompt, match longest existing prefix in memory/disk.
4. Resume generation from the matched block instead of layer 0, token 0.

### Outputs
- Prefix matching logic.
- KV Cache allocator updates.

### Acceptance Criteria
- [x] Repeated exact prompts skip prefill compute entirely.
- [x] Partially matching prompts skip compute up to the divergence point.

### Failure Modes
- If cache mismatch occurs, fallback to full recomputation without crashing.

### Rollback
- Set config `use_prefix_cache=false`.

### Tests
- Integration: Send prompt A, then prompt A + B. Verify second prompt has near-zero TTFT for part A.

### Regression Risks
- Stale or corrupted KV state bleeding into new requests.

---

### Task ID
BRM-S2-006

### Title
Flash Attention

### Objective
Implement flash attention for the CPU (and eventually wgpu) backend to heavily optimize attention bottleneck on long contexts.

### Scope
- `src/inference/flash_attn_cpu.rs`
- `src/inference/engine.rs`

### Non-Goals
- Do not support arbitrary attention masks initially (causal only).

### Dependencies
- SIMD optimization structure.

### Inputs
- Tiling logic principles.

### Steps
1. Implement block-wise QK^T calculation.
2. Maintain online softmax scaling (row max and denominator).
3. Integrate into the attention forward pass.

### Outputs
- Optimized flash attention kernel.

### Acceptance Criteria
- [x] Memory footprint during attention is O(N) instead of O(N^2).
- [x] Output matches naive attention exactly (within floating point epsilon).

### Failure Modes
- N/A

### Rollback
- Revert to naive attention via compile flag or config.

### Tests
- Benchmark: 8K context memory profiling (must show flat memory curve).

### Regression Risks
- Floating point drift at very long contexts.

---

### Task ID
BRM-S2-007

### Title
INT4/INT8 Support

### Objective
Add support for quantized weights (INT4/INT8) and dequantization kernels to fit larger models into limited RAM/VRAM.

### Scope
- `src/models/quantization.rs`
- `src/compute/wgpu_backend.rs` (shader dequant)
- `src/inference/cpu_engine.rs`

### Non-Goals
- Do not implement KV cache quantization (separate task).

### Dependencies
- BRM-S2-004 (wgpu).

### Inputs
- Standard GGUF or Safetensors quantized formats.

### Steps
1. Parse quantization metadata from models.
2. Implement CPU dequant-and-multiply kernel.
3. Implement WGSL dequant-and-multiply kernel.

### Outputs
- Quantized format loaders.
- Dequantize kernels.

### Acceptance Criteria
- [x] Models loaded as INT8/INT4 consume roughly 1/2 or 1/4 the RAM.
- [x] Inference succeeds without crashing.


### Failure Modes
- Unsupported quantization format fails explicitly on load.

### Rollback
- Use FP16 models only.

### Tests
- Unit: Dequantization array match.

### Regression Risks
- Severe latency penalty if dequantization is not fused with GEMM.

---

### Task ID
BRM-S2-008

### Title
Persistent GPU Buffers

### Objective
Maintain model weights persistently in wgpu buffers across requests to eliminate PCIe transfer overhead on every query.

### Scope
- `src/compute/wgpu_backend.rs`
- `src/models/tensor_db.rs`

### Non-Goals
- Do not implement cluster-level placement.

### Dependencies
- BRM-S2-004 (wgpu).

### Inputs
- Model loading lifecycle.

### Steps
1. Allocate wgpu buffers during model `load` phase.
2. Keep buffers alive in the `WgpuBackend` state.
3. Reference these buffers during `forward` passes instead of re-uploading.

### Outputs
- Stateful wgpu backend.

### Acceptance Criteria
- [x] VRAM utilization shows model resident after first load.
- [x] TTFT drops significantly on second query due to zero upload.

### Failure Modes
- If OOM occurs during load, gracefully fail the load request.

### Rollback
- Revert to streaming buffers if persistent causes memory leaks.

### Tests
- Benchmark: TTFT comparison between first and second query.

### Regression Risks
- VRAM leaks across sessions.

---

### Task ID
BRM-S2-009

### Title
Criterion Benchmarks

### Objective
Establish a formal, automated benchmark suite using the `criterion` crate to track latency and throughput regressions on all hardware paths.

### Scope
- `benches/inference_bench.rs`
- `Cargo.toml`

### Non-Goals
- Do not test model accuracy (MMLU, etc.) here; only performance.

### Dependencies
- CPU and wgpu backends.

### Inputs
- Fixed standard prompt.

### Steps
1. Add `criterion` to `dev-dependencies`.
2. Create benchmark harnessing exact decode, speculative decode, CPU, and wgpu paths.
3. Output standard TPS metrics.

### Outputs
- `benches/` directory.

### Acceptance Criteria
- [x] `cargo bench` runs successfully and outputs statistical variance.
- [x] Benchmarks isolate prefill from decode phases.

### Failure Modes
- N/A

### Rollback
- N/A

### Tests
- N/A (this is the test).

### Regression Risks
- CI timeouts if benchmarks are too large.

---

### Task ID
BRM-S2-010

### Title
Heterogeneous Scheduler v1

### Objective
Implement the logic to dynamically route inference work to the CPU or GPU based on availability and tensor sizes.

### Scope
- `src/planner/scheduler.rs`
- `src/inference/engine.rs`

### Non-Goals
- Do not route across multiple nodes.

### Dependencies
- BRM-S2-001 (CPU)
- BRM-S2-004 (wgpu)

### Inputs
- Current hardware availability map.

### Steps
1. Identify if a valid wgpu adapter exists.
2. If yes, route large GEMM operations to wgpu.
3. Keep small operations (like single-token sampling or tiny embeddings) on CPU to avoid transfer latency.
4. Fall back to CPU entirely if wgpu encounters an error.

### Outputs
- `scheduler.rs`

### Acceptance Criteria
- [x] Engine automatically utilizes wgpu if available.
- [x] Disconnecting/simulating GPU failure causes exact fallback to CPU.

### Failure Modes
- If GPU queue blocks forever, abort request and log scheduler failure.

### Rollback
- `--disable-gpu` config overrides scheduler.

### Tests
- Integration: Simulate GPU failure midway through generation, verify CPU takes over.

### Regression Risks
- Over-scheduling CPU while GPU is idle.
