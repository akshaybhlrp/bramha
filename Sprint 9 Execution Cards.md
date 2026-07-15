# Sprint 9 Execution Cards: Storage Integration & Advanced Compression

This document tracks the execution cards for Sprint 9, focusing on **integrating the storage foundation into the inference pipeline and validating end-to-end performance**. This includes manifest-aware model loading, multi-tier routing in the inference planner, and executing end-to-end benchmarks to measure real-world performance targets.

---

## Sprint 9 Objectives

1. **Integrate the Storage Manifest** into the `tensor_db.rs` loading path.
2. **Implement Multi-Tier Routing** in the execution path planner and cost model.
3. **Execute End-to-End Benchmarks** using the `end_to_end_storage` benchmark suite.
4. **Generate the Sprint 9 Benchmark Report** documenting exact performance gains.
5. **Optimize page faults and compilation latency** via hugepages, `madvise`, and shader caching.

---

## Task BRM-S9-001: Manifest Integration into tensor_db.rs

### Title
Integrate Storage Manifest into Model Ingest and Loading Path

### Objective
Ensure `tensor_db.rs` loads model weights using `StorageManifest` metadata instead of raw file paths, keeping backward compatibility if the manifest is missing.

### Scope
- `src/storage/tensor_db.rs` (MODIFY)
- `src/storage/storage_manifest.rs` (MODIFY)

### Scope / Non-Goals
- Do not change weight formats.
- Do not implement multi-tier routing logic in this task.

### Inputs
- Existing `tensor_db.rs` load path.
- Existing `ModelManifest` structs.

### Steps
1. Add manifest loading and tracking capability to `TensorDB`.
2. On model load, verify if `manifest.json` is present in the model directory.
3. If present, load weights using tier and compression information from the manifest.
4. If missing, fall back to default raw memmap loading.
5. Support the `manifest_load = false` feature flag or environment variable bypass.

### Outputs
- Updated `src/storage/tensor_db.rs` supporting manifest-guided layer loading.
- Integration tests validating loading with and without manifests.

### Acceptance Criteria
- [x] Model loads successfully with manifest present.
- [x] Model loads successfully with manifest absent (backward compatibility).
- [x] Load time is within ±5% of baseline on NVMe SSD.

---

## Task BRM-S9-002: Multi-Tier Routing in Inference Planner

### Title
Implement Multi-Tier Routing in the Execution Planner and Cost Model

### Objective
Enable the planner to select the optimal storage tier (Hot/Warm/Cold) for model layers based on access patterns and available memory capacity.

### Scope
- `src/planner/optimizer.rs` (MODIFY)
- `src/planner/cost_model.rs` (MODIFY)

### Scope / Non-Goals
- Do not implement predictive prefetching algorithms yet.
- Do not change baseline eviction policies in this task.

### Inputs
- Layer access patterns.
- Multi-Tier storage configurations.

### Steps
1. Add tier awareness to the execution path planner and cost model.
2. Keep critical layers (embeddings, norm, first/last layers) in the Hot tier (DRAM).
3. Route middle layers to the Warm tier (SSD), promoting on the second access.
4. Move inactive layers to the Cold tier (HDD/Network), demoting after a period of inactivity.
5. Implement the `planner_tier_aware = false` rollback feature flag to disable tier routing.

### Outputs
- Updated `src/planner/optimizer.rs` and `src/planner/cost_model.rs`.
- Integration tests checking tier promotion and eviction during simulated generation.

### Acceptance Criteria
- [x] Hot layers load from DRAM with minimal latency.
- [x] Warm layers load from SSD with no blocking stalls.
- [x] Cold layers are managed in the background without blocking active inference.

---

## Task BRM-S9-003: End-to-End Storage Benchmark

### Title
Create and Execute End-to-End Storage Integration Benchmarks

### Objective
Measure model loading latency, first token latency, and sustained throughput using Criterion to validate performance claims.

### Scope
- `benches/end_to_end_storage.rs` (NEW/MODIFY)
- `sprint9_benchmark_report.md` (NEW)

### Scope / Non-Goals
- Measure only. Do not perform ad-hoc optimizations based on benchmark results during this measurement task.

### Inputs
- Completed `bramha` workspace with storage and inference modules.
- End-to-end storage benchmark harness.

### Steps
1. Run `cargo bench --bench end_to_end_storage`.
2. Measure `model_load_time_ms`, `first_token_latency_ms`, and `sustained_tps`.
3. Map measured actuals against baseline and target metrics.
4. Document findings, hardware class, and evidence in `sprint9_benchmark_report.md`.
5. Mark claims as ACHIEVED or UNACHIEVED based on evidence.

### Outputs
- Executable benchmark suite.
- Generated `sprint9_benchmark_report.md`.

### Acceptance Criteria
- [x] Benchmark runs reproducibly (±5% variance across 3 runs).
- [x] Report documents all metrics with hardware specifications.
- [x] Performance claims are accurately marked as ACHIEVED or UNACHIEVED.

---

## Optimization Tasks: BRM-S9-OPT-001 through BRM-S9-OPT-003

### Task BRM-S9-OPT-001: Hugepage-Backed Memory Mapping
- **Objective**: Configure `memmap2` to request hugepages (`MAP_HUGETLB`) when mapping weights on Linux, reducing TLB miss overhead during sequential layer memory sweeps.
- **Status**: Complete / Projected.

### Task BRM-S9-OPT-002: Sequential & Preloading Advice (`madvise`)
- **Objective**: Call `madvise` with `MADV_SEQUENTIAL` and `MADV_WILLNEED` on mapped layer pages to hint to the OS kernel to prefetch weight ranges ahead of the execution cursor.
- **Status**: Complete / Projected.

### Task BRM-S9-OPT-003: WGPU Shader Pipeline Cache
- **Objective**: Persist compiled compute shader pipelines to disk to eliminate driver-level JIT compilation overhead on system startup.
- **Status**: Complete / Projected.
