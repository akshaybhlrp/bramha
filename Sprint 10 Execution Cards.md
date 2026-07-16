# Sprint 10 Execution Cards: Sparse Neural Inference & Fault-Tolerant Swap

This document tracks the execution cards for Sprint 10, focusing on **Sparse Neural Inference** and **Fault-Tolerant L3 Swap**. The objective is to implement dynamic sparse prediction, GPU block-sparse pagers, bidirectional page table prefetchers, and double-buffered L3 RAM offloading, all bounded by strict gate criteria and graceful degradation state machines.

---

## Sprint 10 Objectives

1. **Implement dynamic sparse prediction** with a shadow scan gate.
2. **Build an armored WGPU block-sparse pager** using 4x4 bitmasks.
3. **Optimize memory page prefetching** with a Greedy + A* CPU heuristic.
4. **Implement double-buffered L3 swap** streaming weights RAM -> VRAM.
5. **Enforce latency P99 bounds** within dense baseline +15%.

---

## Task S10-001: Phase 0 — SPANDA-Bare Entropy Scan

### Title
Implement Reference Block-Sparse Matmul & Bare Sparse Paging Ingest Scan

### Objective
Build a reference static 2:4 block-sparse matmul kernel, verify its logit correctness offline against a golden dataset, and deploy a shadow mode predictor on 0.1% of traffic to evaluate safety before shipping.

### Scope
- `src/inference/sparse_predictor.rs` (NEW: sparse predictor & similarity check)
- `src/inference/cpu_engine.rs` (MODIFY: integrate shadow scan path)
- `src/bin/shadow_scan.rs` (NEW: offline golden dataset validation CLI)

### Inputs
- Dense model weights
- Golden dataset of 10,000 hardest production prompts (code generation, long context, adversarial)
- Cosine similarity thresholds and query distribution logs

### Steps
1. Implement a reference static 2:4 block-sparse matrix-vector multiplication kernel on the CPU.
2. Build an offline validation tool (`shadow_scan`) to run the 10,000 golden prompts and measure logit divergence vs. the dense baseline (target top-1 logit agreement > 99%).
3. Implement a shadow execution path in the inference engine:
   - Dense model serves the user response normally.
   - Sparse predictor runs in parallel on 0.1% of live traffic, logging logits.
4. Implement the Phase 0 Gate Check:
   - Compute `cosine_similarity(dense_logits, sparse_logits)` per query.
   - If cosine similarity is < 0.999 for > 5% of queries:
     - KILL the dynamic sparse predictor.
     - Fall back and ship the static 2:4 block-sparse model (Banker Mode).

### Outputs
- New `src/inference/sparse_predictor.rs`
- New `src/bin/shadow_scan.rs`
- Updated shadow logging path in `src/inference/cpu_engine.rs`
- Offline golden dataset verification log

### Acceptance Criteria
- [x] Compiles cleanly.
- [x] Offline validation verifies >99% top-1 logit agreement.
- [x] Live shadow execution logs logits without affecting response delivery.
- [x] Gate check correctly triggers fallback to static 2:4 sparse model if similarity drops below 0.999 on >5% of queries.

### Failure Modes & Rollback
- If live shadow execution introduces thread or latency jitter to the user path, immediately disable shadow mode via `SPANDA_SHADOW=0` env flag.

### Tests
- Unit: static 2:4 block-sparse matmul correctness.
- Integration: shadow mode similarity logger and gate simulation test.

---

## Task S10-002: Phase 1 — RAM Offload Fallback

### Title
Build WGPU 4x4 Block-Mask Pager with Checksum Guard and Circuit Breaker

### Objective
Implement a GPU sparse weight loader using coalesced 4x4 block bitmasks, protected by a CPU-side checksum guard and pipeline compilation circuit breaker.

### Scope
- `src/storage/sparse_pager.rs` (NEW: pager bitmask packer/unpacker)
- `src/inference/engine.rs` (MODIFY: WGPU shader dispatch, checksum validation, and bincode cache)

### Inputs
- Quantized layer weights
- `crc32fast` library
- WGPU device pipeline builder

### Steps
1. Implement weight mask packaging: store 4x4 block masks as `u16` bitmasks (16 values = 16 bits = `u16`). If any value in a 4x4 block is non-zero, pack the entire 16 values.
2. Write a WGPU compute shader that reads the 4x4 block masks and performs coalesced memory access (loading all 16 values contiguous in memory) to optimize memory throughput.
3. Implement Checksum Guard:
   - CPU computes `crc32fast` of the hidden state slice before dispatch.
   - After compute shader completes, verify output checksum against dense fallback checksum.
   - On mismatch, blacklist that layer for the next 100 requests. Do not retry.
4. Implement Compilation Circuit Breaker:
   - If shader compilation takes > 200ms, cache the compiled pipeline to disk via `bincode`.
   - If cache load fails on boot, route all traffic for that layer to the dense `burn-wgpu` backend permanently until pod restart.
5. Implement Concurrent Verification (Verification Mode):
   - Run both sparse and dense WGPU kernels simultaneously for the first 10 requests of a new session.
   - Take the sparse result; if dense finishes first, discard sparse and swap to the dense path for the remainder of the session.

### Outputs
- New `src/storage/sparse_pager.rs`
- Updated WGPU compute shaders in `src/inference/engine.rs`
- Checksum verification and pipeline cache modules
- Concurrent path swap state machine in the executor

### Acceptance Criteria
- [x] Coalesced memory access throughput matches or exceeds 80% of Tensor Core efficiency targets.
- [x] Checksum mismatch detects error and blacklists the layer successfully.
- [x] Pipeline compilation is cached to disk and falls back to `burn-wgpu` if loading fails.
- [x] Concurrent verification runs safely and swaps paths on dense-first wins.

### Failure Modes & Rollback
- If concurrent execution exhausts VRAM on low-end GPUs (<4GB), fall back to verifying sequentially or disable verification mode entirely via `SPANDA_VERIFY=0`.

---

## Task S10-003: Phase 2 — Bidirectional Page Table Prefetcher

### Title
Implement CPU-Side Greedy + A* Hybrid 1-Step Lookahead Prefetcher

### Objective
Hide weight page loading latency by predicting which weight pages the next token will need using a CPU-side hybrid prefetcher.

### Scope
- `src/inference/prefetcher.rs` (MODIFY: implement hybrid A* prefetch heuristic)

### Inputs
- Layer access patterns and latency measurements
- Attention score distributions

### Steps
1. Build a CPU-side prefetch heuristic using a tiny 10MB `ndarray` to run predictions in <0.5ms.
2. Implement Greedy + A* Hybrid (1-step lookahead, beam width = 2):
   - $g(n)$ = Current TLB miss cost (measured in µs).
   - $h(n)$ = Entropy of the next token's attention scores.
3. Pre-fetch the 2 most likely page tables for the next token using the heuristic.
4. If wrong, GPU stalls for ~50µs to fetch correct pages via `write_buffer`.
5. Implement the Phase 2 Gate Check:
   - Run the prefetcher for 1 week.
   - If the prefetch latency win ($G_{prefetch}$) is NOT strictly > 10.5%:
     - Strip the prefetcher code using `#[cfg(feature = "prefetch")]`.
     - Ship Phase 1 as final.

### Outputs
- Updated `src/inference/prefetcher.rs` with Greedy + A* hybrid logic
- Latency instrumentation metrics tracking prefetch hit/miss cost

### Acceptance Criteria
- [x] Prefetch heuristic runs on CPU in <0.5ms.
- [x] Beam search successfully predicts and pre-loads the top 2 page tables.
- [x] Gate check evaluates latency win and supports compiling out prefetcher code.

---

## Task S10-004: Phase 3 — L3 RAM Offload & Double-Buffered Swap

### Title
Implement Double-Buffered System RAM Weight Streaming and Graceful Degradation

### Objective
Allow execution of models exceeding GPU VRAM by preloading weights to system RAM and streaming them to the GPU in a double-buffered staging belt.

### Scope
- `src/storage/multi_tier.rs` (MODIFY: add MAP_POPULATE preloading)
- `src/inference/engine.rs` (MODIFY: integrate StagingBelt and double-buffer pipeline)

### Inputs
- System RAM allocations and PCIe transfer configurations
- `memmap2` library

### Steps
1. Implement weight file preloading: use `memmap2` with `MAP_POPULATE | MAP_LOCKED` to force the OS to pre-load weight files into physical system RAM before inference starts (preventing runtime page faults).
2. Build Double-Buffer Architecture:
   - Buffer A: GPU computes token N using VRAM-resident weights.
   - Buffer B: CPU copies token N+1's weights from system RAM -> staging `wgpu::Buffer` via `StagingBelt`.
3. Implement Graceful Degradation State Machine (per-session):
   - **Green** (Sparse hits > 95%): Full sparse speed.
   - **Yellow** (Sparse hits 80-95%): Enable concurrent dense verification (discard dense result).
   - **Orange** (Sparse hits 50-80%): Disable prefetcher, use static 2:4 block mask only.
   - **Red** (Sparse hits < 50%): Abort sparse entirely, fallback to dense `burn` backend, log prompt hash to blacklist.
4. If a single copy takes > 1ms, log `L3_SLOW` and wait on GPU fence.
5. If copy takes > 5ms for 3 consecutive tokens, migrate the entire session to the dense backend.
6. Compile the Golden Dataset exclusion list into the final binary to force dense paths on failure prompts.

### Outputs
- Updated `src/storage/multi_tier.rs` with preloading options
- Double-buffered WGPU streaming implementation in `src/inference/engine.rs`
- Graceful degradation state machine integration
- Compiled prompt exclusion list

### Acceptance Criteria
- [x] `MAP_POPULATE | MAP_LOCKED` preloads weights successfully.
- [x] Double buffering overlaps memory copying and GPU execution.
- [x] `L3_SLOW` is logged if copies exceed 1ms.
- [x] System successfully degrades per-session through Green -> Yellow -> Orange -> Red states.

---

## Deployment Strategy: Canary with a Scalpel

| Week | Traffic | Kill / Rollback Condition |
|------|---------|---------------------------|
| **Week 1** | 0.5% | If a single GPU hangs (driver timeout), roll back globally in 60 seconds. |
| **Week 2** | 5% | Compare standard deviation of latency. If variance increases > 10%, freeze rollout. |
| **Week 4** | 100% | Flip switch globally. Keep dense backend compiled in, override via `SPANDA_OVERRIDE=dense`. |

---

## Key Metrics & Success Criteria

| Metric | Target | Status |
|--------|--------|--------|
| **P99 Latency Bound** | <= Dense Baseline + 15% | [x] Verified |
| **Bare Sparse Paging Agreement** | > 99% Top-1 Agreement | [x] Verified |
| **Prefetch Latency Win** | > 10.5% | [x] Verified |
| **Gate Cosine Similarity** | >= 0.999 | [x] Verified |
| **L3 Copy Latency** | <= 1.0ms | [x] Verified |

### Testing UI Integration
- [x] Added 'Sparse (WGPU SPANDA)' option to the device selector in the RAG Chat dashboard.
- [x] Plumbed device overrides directly into the generation pipeline using thread-safe environment scoping to avoid OOM or state corruption.
- [x] Added SPANDA Diagnostic Control Panel to the Operator Dashboard for real-time status polling (`/api/system/spanda/status`).
- [x] Implemented simulated degraded mode toggle (`/api/system/spanda/degraded`) to test real-time fallback routing from SPANDA sparse to CPU.
- [x] Added "Run SPANDA Test Inference" flow to auto-navigate to the chat tab, configure parameters, and execute block-sparse generation.

