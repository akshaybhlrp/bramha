# Bramha Storage Efficiency Optimization Roadmap

> **Thesis**: Modern LLM inference bottlenecks are increasingly storage-bound, not compute-bound. Model storage efficiency is more impactful than inference optimization for real-world performance.

---

## Executive Summary

**Current Problem**: 
- TinyLLaMA (201 layers, full-precision): ~500MB on disk, ~2.5GB in memory
- Standard safetensors + memmap loads entire layer at OS page granularity
- No deduplication across models or layers
- No semantic awareness of weight importance
- Inefficient for multi-model scenarios

**Opportunity**: 
Bramha can achieve 10-50x storage reduction through database-native storage strategies that view models as queryable data structures, not opaque binary files.

---

## 1. Core Storage Efficiency Strategies

### 1.1 Strategy: Columnar Tensor Storage
**Concept**: Store weight matrices in columnar format instead of row-major, enabling:
- Better compression (columns with similar patterns group together)
- Efficient partial loading (load only needed columns for inference)
- Cache-friendly access patterns for matrix-vector products

**Implementation**:
```
Standard: W[n_out, n_in] stored row-by-row
Columnar: W stored as [col_0, col_1, ..., col_n_in]
Benefit: In transformer, we compute y = W @ x where x is [n_in]
         Can compute y[i] = col_i · x without loading full W
```

**Storage Saving**: 15-30% through better compression + reduced I/O

---

### 1.2 Strategy: Differential Compression Across Layers
**Concept**: Many transformer layers are near-identical. Store deltas instead of full copies.

**Implementation**:
```
Layer N:   Full weights (reference)
Layer N+1: Delta from Layer N (store only differences)
Layer N+2: Delta from Layer N+1 (or reference Layer N)
...
Layer N+K: Delta from Layer N (multi-level delta trees)
```

**Use Case**:
- LLaMA: 32 layers often share 70-80% of parameter structure
- TinyLLaMA: 201 layers show even more redundancy
- Multi-model: Different quantizations of same base model

**Storage Saving**: 40-60% on layers 2+ (delta representation ~0.4-0.6x of full layer)

---

### 1.3 Strategy: Semantic Weight Importance Tiering & Block-Sparsity
**Concept**: Not all weights matter equally. Store strategically. Introduce block-sparse paging for memory efficiency without NVIDIA's hardware restrictions.

**Weight Sensitivity Hierarchy**:
1. **Critical** (attention heads, output projections): Full precision, always loaded.
2. **Important** (feed-forward gates): INT8 or INT4, precompiled lookup tables, or **4×4 block masks** stored as `u16` bitmasks.
   - *Coalesced Memory Access:* Store 4×4 block masks as `u16` bitmasks (16 values = 16 bits = `u16`). If any value in a 4×4 block is non-zero, load the entire 16 values to maintain cache alignment.
3. **Robust** (intermediate layers): INT4 quantized, 2-4 bit hashing, or static 2:4 sparse layout.
4. **Redundant** (layer normalization scales): Extreme compression, on-demand recompute.

**Implementation**:
```
model_manifest.json:
{
  "layer_0": {
    "attn.q_proj": { "tier": "critical", "format": "f32", "size": 8.2MB },
    "attn.k_proj": { "tier": "critical", "format": "f32", "size": 8.2MB },
    "mlp.gate": { "tier": "important", "format": "sparse_4x4_block", "size": 0.35MB },
    "mlp.down": { "tier": "important", "format": "sparse_4x4_block", "size": 0.35MB },
    "norm.scale": { "tier": "redundant", "format": "f32_computed", "size": 0.004MB }
  }
}
```

**Storage Saving**: 60-75% with no quality loss (via coalesced 4×4 block-sparse masks and redundant tier recomputation).

---

### 1.4 Strategy: Content-Addressed Deduplication
**Concept**: Multiple models often share identical weight blocks. Detect and deduplicate.

**Implementation**:
```
1. Hash every 256-element chunk: blake3(chunk) -> 32-byte hash
2. Store hash -> chunk mapping in dedupe index
3. Models reference by hash, not copy entire weights
4. Garbage collection: delete unreferenced chunks

Example:
  tinyllama base model: 500MB
  tinyllama-q4 quantized: normally 125MB
  With deduplication: 125MB - (shared weights hash refs) = ~40-60MB
  
  Multiple models: 500MB + 125MB + 200MB = 825MB (without dedup)
  With deduplication: 500MB + 60MB + 80MB = 640MB (22% saving)
```

**Storage Saving**: 15-30% with 2-3 models, 40-60% with 10+ models

---

### 1.5 Strategy: Adaptive Quantization by Weight Matrix
**Concept**: Different weight matrices have different quantization tolerance.

**Implementation**:
```
Sensitivity Analysis (done at ingest):
  - Attention Q/K/V projections: INT8 max (critical for output quality)
  - Attention O projection: INT8
  - FFN gates: INT4 (robust to quantization noise)
  - FFN down projections: INT4 safe
  - Embedding: INT4 safe (lookup table cached)
  
Calibration: Use 100 random samples from corpus to find optimal bit-width
             that maintains <0.1% accuracy loss per layer

Result:
  Standard INT4 for all: 125MB, but noisy
  Adaptive quantization: 160MB but negligible quality loss
  Speedup vs quality: 1.4x better quality/size ratio
```

**Storage Saving**: Negligible space (slightly larger) but 20-40% quality improvement

---

### 1.6 Strategy: Incremental/Lazy Layer Loading & Bidirectional Prefetching
**Concept**: Don't load all 201 layers at startup. Load on-demand per inference step and hide memory latency by predicting which weight pages the next token will need using a bidirectional prefetcher.

**Implementation**:
```
Design:
  - Model manifest lists all layers with their storage offsets/sizes.
  - At inference time: only load layers 0-N needed for early exit detection.
  - Bidirectional Prefetcher (Greedy + A* Hybrid, 1-Step Lookahead):
    - Beam width = 2.
    - Heuristics:
      - g(n) = Current TLB miss cost (measured in µs).
      - h(n) = Entropy of the next token's attention scores (predictor of complexity).
    - Heuristic runs on CPU only using a tiny 10MB `ndarray` (<0.5ms overhead).
    - Action: Pre-fetch the 2 most likely page tables for the next token.
    - If wrong: GPU stalls for ~50µs to fetch correct page via `write_buffer`.
  - Cache: keep 3-4 hot layers in memory, spill older ones.

Memory Profile:
  Standard: load 201 layers at startup = 2.5GB DRAM
  Lazy: load 5-10 active layers at once = 65-130MB DRAM
  Prefetch overhead: 2 prefetch pages being loaded = +32MB
  Total: 100-180MB DRAM vs 2.5GB (20-30x reduction)
```

**DRAM Saving**: 85-90% for first token, 70-80% sustained with prefetch win target > 10.5% latency reduction.

**DS4-Validated Refinement** *(Reference: antirez/ds4)*:
DS4 implements a production-grade version of this strategy for routed MoE expert weights:
- **Automatic Cache Budget**: Uses 80% of GPU recommended working set, subtracts non-routed weights, allocates remainder for expert cache
- **Expert Granularity**: Cache is sized in units of complete experts, not generic bytes
- **Hot Expert Preloading**: Pre-populates cache with frequently-used experts on startup (cold start streaming is too slow)
- **mlock Pinning**: Expert cache buffers are `mlock`'d into physical RAM to prevent OS paging
- **Graceful Degradation**: If `mlock` fails, releases a locked-cache margin and continues with measured lockable size
- **Key Insight**: Bramha's A* prefetcher proposal may be overengineered — DS4's simpler approach (automatic budget + LRU eviction) is proven in production

---

### 1.7 Strategy: Multi-Tier Storage & L3 RAM Offload ("Fault-Tolerant" Swap)
**Concept**: Mimic database buffer pools. Keep hot weights in VRAM/DRAM, but offload larger model weights to system RAM (L3) and stream them to the GPU asynchronously via double-buffering.

**Implementation**:
```
Tier 0 (Hot): VRAM/DRAM cache, 4-8 layers, subms access latency
Tier 1 (Warm / L3 offload): System RAM (L3 swap), mmap'ed weights, <1ms access
Tier 2 (Cold): HDD/SSD or network storage, older layers, 10-100ms access

L3 RAM Offload Design:
  - Never fault per-layer (PCIe spikes are unpredictable).
  - Use `memmap2` with `MAP_POPULATE | MAP_LOCKED` to force OS to load the entire offloaded
    weight file into physical RAM before inference starts (requires CAP_IPC_LOCK or root).
  - Double-Buffer Architecture:
    - Buffer A: GPU computes token N using VRAM-resident weights.
    - Buffer B: CPU memcopies token N+1's weights from RAM -> staging `wgpu::Buffer` via `StagingBelt`.
  - Graceful Degradation:
    - If copy takes > 1ms: Log `L3_SLOW` warning, GPU waits on fence (do not crash).
    - If copy takes > 5ms for 3 consecutive tokens: Migrate session to dense `burn` backend.

Example throughput impact:
  All DRAM (2.5GB): 100 tps
  Lazy load with A* prefetch: 80-90 tps (prefetch overlaps I/O)
```

**Effective DRAM/VRAM**: 100-200MB for 1-2 concurrent inferences, allowing large models to execute within tight hardware bounds.

**DS4-Validated Refinements** *(Reference: antirez/ds4)*:
- **read/write I/O over mmap for KV cache files**: DS4 intentionally uses ordinary read/write I/O, NOT mmap, for KV cache persistence. This avoids adding VM mappings to a process that already maps the model via mmap. Bramha should adopt this for KV/session persistence.
- **KV Cache Save Lifecycle**: DS4 saves KV checkpoints at four moments — `cold` (after first long prompt), `continued` (at aligned context frontiers), `evict` (before replacing live session), `shutdown` (clean exit). Bramha should adopt this taxonomy for its multi-tier swap system.
- **Boundary Alignment**: KV saves trim a configurable number of tail tokens and align to prefill chunk boundaries, preventing BPE retokenization drift on resume.

---

### 1.8 Strategy: Dictionary/Pattern Compression
**Concept**: Weight matrices often have repetitive patterns. Use dictionary encoding.

**Implementation**:
```
Example: Layer normalization scales often have value patterns like:
  [1.002, 1.001, 1.003, 1.001, 1.002, ...]
  
Dictionary compression:
  1. Identify top 256 unique values in tensor
  2. Assign 8-bit codes to each value
  3. Store: [codes array] + [value dictionary]
  4. Savings: 8x reduction for tensors with <256 unique values
  
For LLMs:
  Embedding layers: 5-8x (embeddings cluster around mean)
  Layer norm: 8-16x (scales near 1.0)
  Attention masks: 100-1000x (mostly 0, some -inf)
```

**Storage Saving**: 20-50% on specific layer types (3-10% overall)

---

### 1.9 Strategy: Model Factorization at Ingest
**Concept**: Pre-decompose weights using SVD/QR at ingest time (aligns with roadmap Invention 1).

**Implementation**:
```
At ingest:
  1. For each weight matrix W[n_out, n_in]:
     - Compute SVD: W ≈ U[n_out, r] @ S[r, r] @ Vt[r, n_in]
     - Choose r adaptively based on singular value spectrum
     - Store only U, S, Vt instead of W
  
  2. Example: W[2048, 8192] normally 64MB
     - SVD with r=256: U(32MB) + S(0.2MB) + Vt(8MB) = 40MB
     - Or r=128: U(16MB) + S(0.1MB) + Vt(4MB) = 20MB
  
  3. Inference: compute (U @ (S @ (Vt @ x))) instead of W @ x
     - 3 small matmuls instead of 1 large one
     - Highly cache-friendly
     - 35-50% storage reduction, 10-20% compute overhead

When to use:
  - Always for MLP feed-forward layers (high redundancy)
  - Optionally for attention projections (lower rank redundancy)
  - Skip embedding layers (already optimal rank)
```

**Storage Saving**: 35-50% on FFN layers, 50-70% total model size with r=128-256

---

### 1.10 Strategy: Bloom Filter + Locality-Sensitive Hashing for Deduplication
**Concept**: Efficiently identify weight chunks that might be duplicated before full comparison.

**Implementation**:
```
At ingest (first model):
  1. For each weight chunk (256-element blocks):
     - Compute Bloom filter hash(chunk) 
     - Compute LSH signature (dimension-reduced sketch)
     - Store: chunk_id -> (bloom, lsh_sig, location)
  
At ingest (subsequent models):
  2. For each weight chunk in new model:
     - Check Bloom filter (fast negative filter)
     - If hit, compute full hash (blake3)
     - Compare with existing chunks (false positive rate ~1%)
     - If match: dedupe (store reference), else: store new

Speedup:
  Without Bloom+LSH: 100% content hash on all chunks (slow)
  With Bloom+LSH: 99% filtered by Bloom (very fast), 1% full hash (slow but rare)
```

**Ingest Time Saving**: 50-80% faster model ingest (dedup detection overhead minimized)

---

### 1.11 Strategy: Semantic Weight Routing
**Concept**: Route queries to different storage backends based on weight semantics.

**Implementation**:
```
Weight Classification:
  - Always-needed (embeddings, final projection): SSD tier 1, preload
  - Frequent (early layers): SSD tier 1
  - Occasional (middle layers): SSD tier 2
  - Rare (deep layers past early exit): HDD/network tier 3

Query Router:
  - Inference planner knows which layers will be used
  - Routes load requests to appropriate tier
  - Minimal latency for critical path, spills others

Example:
  Query: "simple Q&A" (early exit at layer 8)
  Router loads: layers 0-8 from tier 1 (fast)
  Skips: layers 9-201 (not needed)
  Total load time: 50ms instead of 500ms
```

**End-to-End Speedup**: 20-40% reduction in model loading latency

---

### 1.12 Strategy: Attention Weight Caching
**Concept**: Pre-compute and cache attention weights for common prompt prefixes.

**Implementation**:
```
Observation:
  - System prompts are identical across many queries
  - RAG contexts are retrieved repeatedly
  - Attention patterns over fixed prefixes are deterministic

Storage:
  - Cache attention outputs for first K tokens (prefix cache)
  - Example: 40-token system prompt attention = 201 layers × 32 heads × [40,40] scores
  - Stored as: {prefix_hash: [attention_outputs]}
  - Deduplicated by Bloom filter on prefix content
  
Benefit:
  - Skip 40 token forward passes for system prompt
  - Equivalent to ~200MB model × 40 forward passes = 8GB I/O saved
  - Stored locally: 200MB × 32 heads = 6.4MB (50x reduction)
```

**Inference Speedup**: 2-5x for prompts with large fixed prefixes

---

## 2. Orchestration: Storage Efficiency Pipeline

### 2.1 Multi-Tier Storage Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Query / Inference                     │
│                  (Planner decides path)                  │
└──────────────────────┬──────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
   ┌────▼─────┐   ┌───▼────┐   ┌────▼─────┐
   │ Early Exit│   │Standard│   │Specialized
   │  (1-8ms) │   │ (10ms) │   │ (50ms)   │
   │ Tier 0-1 │   │Tier 1-2│   │ Tier 2-3 │
   └────┬─────┘   └───┬────┘   └────┬─────┘
        │             │             │
    ┌───▼──────────────▼─────────────▼───┐
    │    Dedupe Index + Route Table      │
    │  (hash -> [models, offset, size])  │
    └───┬──────────────────────────────┬─┘
        │                              │
   ┌────▼──────────┐            ┌─────▼───────┐
   │ DRAM Cache    │            │  SSD Cache  │
   │ (100-200MB)   │            │ (2-5GB)     │
   │ Hot layers    │            │ Warm layers │
   └─────┬─────────┘            └──────┬──────┘
         │                             │
         │              ┌──────────────┤
         │              │              │
    ┌────▼──────┐  ┌───▼────┐  ┌─────▼──────┐
    │ /dev/shm  │  │ NVMe   │  │ HDD/Network│
    │ (fastest) │  │ (fast) │  │ (slow)     │
    └───────────┘  └────────┘  └────────────┘
```

### 2.2 Ingest Pipeline

```
1. User ingests model (tinyllama.safetensors)
   ↓
2. Content-addressing: hash each block
   ↓
3. Deduplication check: lookup in dedupe index
   ↓
4. Factorization: SVD/QR decompose weight matrices
   ↓
5. Adaptive quantization: per-layer optimal bit-width
   ↓
6. Columnar conversion: convert to column-major storage
   ↓
7. Differential encoding: compute deltas vs similar models
   ↓
8. Tiering decision: classify weights by importance
   ↓
9. Write to storage: Tier 0 (DRAM) / Tier 1 (SSD) / Tier 2 (HDD)
   ↓
10. Update metadata: model manifest, dedupe index, routing table
    ↓
11. Success: model ready for inference
```

### 2.3 Inference-Time Access Pattern

```
Inference query arrives
  ↓
Planner examines query (length, complexity, task type)
  ↓
Determines execution path (early exit likely? multi-model? speculative?)
  ↓
Routes to storage tier fetcher:
  
  If early exit likely:
    - Fetch layers 0-12 from Tier 0/1 (blocking, <100ms)
    - Prefetch layers 13-20 async from Tier 1
    - Skip layers 21+ (probably not needed)
    
  If full model needed:
    - Fetch layers 0-10 from Tier 0/1 (blocking, <200ms)
    - Prefetch layers 11-50 from Tier 1 (non-blocking, overlaps compute)
    - Stream layers 51+ as-needed during decode
    
  If multi-model:
    - Fetch shared weights once (dedupe hit)
    - Fetch model-specific deltas (small, <20ms each)
  ↓
Decode begins while prefetcher works in background
  ↓
Inference completes, profiling updates storage stats
  ↓
Promote hot layers, demote cold layers in tiering system
```

---

## 3. Implementation Roadmap

### Phase 1: Foundation (Week 1-2) (COMPLETE ✅)
- [x] Implement `StorageManifest` struct with layer metadata
- [x] Build content-addressed storage layer (hash -> data mapping)
- [x] Add deduplication index (Bloom filter + LSH)
- [x] Implement multi-tier storage planner
- [x] Profile current model loading performance

### Phase 2: Compression (Week 3-4)
- [ ] Columnar tensor format codec
- [ ] Differential compression across layers
- [ ] Adaptive quantization calibration
- [ ] Dictionary encoding for special layers
- [ ] Benchmark compression ratios

### Phase 3: Factorization (Week 5-6)
- [ ] SVD factorization at ingest
- [ ] Fused inference kernel: (U @ (S @ (Vt @ x)))
- [ ] Rank adaptation based on singular values
- [ ] Ablation: rank vs accuracy/speed tradeoff

### Phase 4: Runtime Optimization (Week 7-8)
- [ ] Lazy layer loading with prefetcher
- [ ] Storage tier promotion/demotion
- [ ] Attention weight caching
- [ ] Routing planner integration

### Phase 5: Multi-Model Efficiency (Week 9-10)
- [ ] Delta storage for model variants
- [ ] Semantic weight routing
- [ ] Cross-model deduplication statistics
- [ ] Bundle multiple models efficiently

---

## 4. Sparse Neural Inference Integration (Active Sprint 10)

This track runs in parallel/sequence with Strategy 1.6 and 1.7 to integrate dynamic sparse weight prediction, optimized pagers, prefetching heuristics, and system RAM offloading.

### Phase 0: Ingest & Entropy Scan (Bare Sparse Paging)
- [ ] Build static 2:4 block-sparse reference matmul
- [ ] Run golden dataset offline (verify top-1 agreement > 99% against dense baseline)
- [ ] Shadow mode execution on 0.1% traffic for 24h
- [ ] GATE: If cosine_similarity(dense_logits, sparse_logits) < 0.999 for > 5% of queries:
  - KILL dynamic sparse predictor, fallback to static 2:4 sparse model (Banker Mode).

### Phase 1: RAM Offload Fallback
- [ ] Build 4x4 block-mask pager in Rust + wgpu
- [ ] Implement `crc32fast` checksum guard for hidden state validation
- [ ] Implement circuit breaker + `bincode` compiler pipeline disk caching
- [ ] Implement concurrent dense/sparse verification for first 10 requests of a session

### Phase 2: The "Bidirectional" Prefetcher
- [ ] Build CPU-side prefetch heuristic using a 10MB `ndarray`
- [ ] Implement Greedy + A* Hybrid 1-step lookahead (beam width = 2)
- [ ] GATE: 1 week latency win check (must be strictly > 10.5%, else strip prefetch and run Phase 1 only)

### Phase 3: L3 RAM Offload & Double-Buffer Swap
- [ ] Implement `memmap2` with `MAP_POPULATE | MAP_LOCKED` system RAM offload
- [ ] Build double-buffer staging belt (`StagingBelt` RAM -> VRAM copy)
- [ ] Implement `L3_SLOW` monitoring and 5ms consecutive token fallback to dense backend
- [ ] Compile final Golden Dataset exclusion list into binary


---

## 4. Expected Outcomes

### Storage Efficiency Targets

| Strategy | Saving | Implementation Effort |
|----------|--------|----------------------|
| Columnar + Dedup | 15-30% | Low |
| Differential Compression | 40-60% (layers 2+) | Medium |
| Semantic Tiering | 60-75% | Medium |
| SVD Factorization | 35-50% | High |
| Adaptive Quantization | Quality (+20-40%) | Medium |
| Lazy Loading | 85-90% DRAM reduction | High |
| Combined Stack | **50-80% total** | **High** |

### Performance Impact

| Metric | Before | After | Target |
|--------|--------|-------|--------|
| Model load time | 500ms | 50-100ms | <100ms |
| Startup DRAM | 2.5GB | 100-200MB | <200MB |
| Inference TTFT | 1.2s | 0.3-0.5s | <500ms |
| Sustained throughput | 0.42 tps | 5-15 tps | 50+ tps |
| Multi-model overhead | +500MB each | +40-60MB | +30-50MB |

---

## 5. Risk Mitigation

### Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Factorization reduces accuracy | Rank ablation study, quality thresholds, exact fallback |
| Complex prefetching bugs | Extensive testing, fallback to sync load if prefetch fails |
| Deduplication false positives | Bloom filter FP rate <1%, full hash validation on hits |
| Tiering invalidation | Version tracking, explicit invalidation on model update |
| Differential corruption | Validation checksums, atomic writes with WAL |

---

## 6. Next Steps

1. **Week 1**: Audit current model storage, measure baseline (size, load time, DRAM)
2. **Week 2**: Implement storage manifest + content-addressing
3. **Week 3**: Add deduplication + Bloom filter
4. **Week 4**: Implement columnar codec + differential compression
5. **Week 5+**: Factorization, lazy loading, tiering

**First Milestone**: 40-50% storage reduction + <100ms model load time

---

## Appendix: Code Structure

```
src/storage/
├── storage_manifest.rs         (NEW: layer metadata, versioning)
├── content_addressing.rs       (NEW: hash-based deduplication)
├── columnar_codec.rs           (NEW: column-major tensor format)
├── factorization.rs            (NEW: SVD/QR decomposition)
├── differential_compression.rs (NEW: delta storage)
├── adaptive_quantizer.rs       (NEW: per-layer quantization)
├── multi_tier.rs               (NEW: DRAM/SSD/HDD routing)
├── attention_cache.rs          (EXISTING: expanded for prefix caching)
├── tensor_db.rs                (MODIFY: integrate new layers)
└── inference_prefetcher.rs     (NEW: background layer loading)
```

---

## References

- Roadmap Section 3.1: Pre-Decomposed Tensor Storage (SVD factorization)
- Roadmap Section 3.2: Incremental and Bounded KV State (multi-tier design)
- Roadmap Section 5.4: Model DB (storage architecture)
- Database wisdom: LSM trees, B-trees, buffer pools, tiering, compression

## DS4 (DwarfStar) Cross-References

> [antirez/ds4](https://github.com/antirez/ds4) — DeepSeek V4 Flash/PRO local inference engine

| DS4 Feature | Bramha Strategy | Validation Status |
|---|---|---|
| SSD streaming for routed MoE experts | Strategy 1.6 (Lazy Loading) + 1.7 (Multi-Tier) | **Production-proven** on MacBook SSDs |
| Automatic cache budget (80% working set) | Multi-Tier hot tier sizing | **Proven** — simpler than Bramha's A* prefetcher |
| mlock for expert cache buffers | L3 RAM Offload (MAP_LOCKED) | **Proven** — with graceful fallback on lock failure |
| read/write I/O for KV files (not mmap) | Multi-Tier swap I/O path | **Proven** — avoids VM mapping bloat |
| KV save lifecycle (cold/continued/evict/shutdown) | KV persistence taxonomy | **Proven** — adopted into Bramha Invention 1 |
| Boundary-aligned KV trimming | BPE safety for prefix reuse | **Proven** — critical for reliable session resume |
| Rendered-text SHA key for cache lookup | Content-addressed session indexing | **Proven** — simpler than embedding similarity |

**Key DS4 Insight for Bramha Storage**:
DS4 demonstrates that the distinction between "model weights in GPU memory" and "KV cache on disk" is the correct architectural split. Bramha's multi-tier storage should similarly separate:
- **Weight tiers**: DRAM (hot) → SSD (warm) → HDD (cold) for model parameters
- **Session tiers**: In-memory (live) → SSD (saved) → archived (stripped) for KV state

These are different optimization problems with different access patterns and should not share eviction policies.
