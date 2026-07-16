# SPANDA v7 Design Plan

SPANDA is a standalone sparse inference backend designed to overcome the memory wall for LLMs running on consumer hardware. The v7 plan introduces query-conditional sparse paging, allowing models to exceed VRAM limits by keeping only the necessary parts of the model resident in VRAM and dynamically fetching others.

## Core Problem: The Memory Wall
LLM inference on consumer hardware is typically constrained by memory bandwidth and VRAM capacity, rather than compute capacity. Large models cannot fit in VRAM entirely, and traditional methods of loading them token-by-token from disk are too slow for interactive generation.

## SPANDA's Solution
SPANDA uses a database-native approach: query-conditional sparse paging. It loads only the necessary weight "pages" (experts) based on the query's path through the model. It includes features like offloading to host RAM, VRAM page caching, and optional quantization and prefetching.

## Implementation Phases

### Phase 0: Bare Sparse Paging
- **Objective**: Implement the foundational GPU-side page loader for query-conditional sparse paging.
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

### Phase 3: Self-Profiling, Dynamic Base, 3-bit Quant
- **Status**: Deferred indefinitely / Never
- **Objective**: Advanced dynamic optimization and aggressive quantization.

## Gate Discipline

Validate hypothesis before building. If a gate fails, ship the fallback. These gates are non-negotiable.

| Phase | Gate Condition | Fallback if Failed |
|---|---|---|
| **Phase 0** | Bare sparse paging passes golden vector and MSE checks | Ship static 2:4 sparse or full-dense |
| **Phase 1** | Latency ≤ baseline +15%, Top-1 agreement >99% | Ship Phase 0 only |
| **Phase 2** | Perplexity delta < 0.5%, speedup > 1.2x | Ship Phase 1 only |
| **Phase 2.2** | >80% prefetch hit rate, no latency regression | Ship Phase 2 only |

## SPANDA Engine Deliverables
- `spanda-convert`: Binary to convert models to the SPANDA format.
- `spanda-calibrate`: Binary for sparsity and quantization calibration.
- `spanda-run`: Standalone runner (or integrated `bramha-run`).
- `model.spanda`: File format specification for serialized query-conditional sparse format.
