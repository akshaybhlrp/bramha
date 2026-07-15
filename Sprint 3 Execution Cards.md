# Sprint 3 Execution Cards: Retrieval and Evidence

This document tracks the retroactive execution cards for Sprint 3, aligned with the Strict AI Execution Specifications (v8.0).

---

### Task ID
BRM-S3-001

### Title
IVF/HNSW/BM25 Vector and Text Search Indexes

### Objective
Implement high-performance vector search indexes (IVF-Flat, HNSW) and lexical keyword matching (BM25) to enable multi-faceted lookup capabilities.

### Scope
- `src/index/bm25.rs`
- `src/index/hnsw.rs`
- `src/index/ivf_flat.rs`
- `src/index/kmeans.rs`
- `src/index/mod.rs`
- `src/index/strategies.rs`

### Non-Goals
- Do not implement raw vector ingestion APIs inside this task.
- Do not connect to external third-party vector databases.

### Dependencies
- BRM-S1-001 (Stable Core setup).

### Inputs
- Multi-dimensional vector float arrays.
- Text document corpora.

### Steps
1. Create `bm25.rs` implementing standard BM25 tokenization, term-frequency calculation, and scoring logic.
2. Create `kmeans.rs` with standard clustering behaviors for IVF partition updates.
3. Create `ivf_flat.rs` implementing IVF partitioning, centroids lookup, and vector matching.
4. Create `hnsw.rs` implementing hierarchical proximity search graphs.
5. Create strategies helper for query index selection.

### Outputs
- IVF, HNSW, and BM25 index implementations.
- Associated index-creation correctness unit tests.

### Acceptance Criteria
- [x] Index implementations pass basic correctness queries.
- [x] BM25 indexing correctly scores lexical similarities.
- [x] HNSW/IVF-Flat builds search indices and resolves nearest-neighbor queries cleanly.

### Failure Modes
- If vector dimensionalities mismatch on query matching, return clear runtime errors instead of panic.

### Rollback
- Fall back to standard flat scan or direct SQL retrieval if advanced index structures fail.

### Tests
- Unit: `index::bm25::tests::test_bm25_search_indexing`
- Unit: `index::bm25::tests::test_bm25_tokenize`
- Unit: `index::kmeans::tests::test_kmeans_simple`
- Unit: `index::hnsw::tests::test_hnsw_correctness`
- Unit: `index::ivf_flat::tests::test_ivf_diagnostics_and_profiles`

### Regression Risks
- High indexing compile times or high VRAM/RAM overhead for massive vector datasets.

---

### Task ID
BRM-S3-002

### Title
Hybrid Retrieval Integration

### Objective
Provide combined vector and lexical retrieval (hybrid search) that merges results from semantic similarity scans and BM25 text keyword matching.

### Scope
- `src/storage/collection.rs` or `src/lib.rs` (Collection search integration)

### Non-Goals
- Do not add complex rerankers inside this specific baseline step.

### Dependencies
- BRM-S3-001 (Vector and lexical search indexes).

### Inputs
- Text queries.
- Pre-built index instances.

### Steps
1. Integrate BM25 keyword scans and IVF/HNSW vector semantic searches together.
2. Normalize distance and keyword scores into standard relative probabilities.
3. Apply Reciprocal Rank Fusion (RRF) or similar weighted combining to produce a unified document ranking.

### Outputs
- Collection query pipeline returning aggregated hybrid search outcomes.

### Acceptance Criteria
- [x] Query searches return balanced results merging keyword matches and semantic matches.
- [x] Score normalization scales all values consistently between 0.0 and 1.0.

### Failure Modes
- If either lexical or vector index is empty, gracefully degrade to execute only the populated index.

### Rollback
- Disable hybrid mode and execute flat lexical search alone.

### Tests
- Unit: Verification inside overall collection retrieval tests.

### Regression Risks
- Increased query latency due to double index searches.

---

### Task ID
BRM-S3-003

### Title
Evidence Sentence Overlap Mapping

### Objective
Examine generated completion drafts and dynamically compute overlap maps comparing generated sentences against raw retrieved text context.

### Scope
- `src/cognitive/evidence.rs`

### Non-Goals
- Do not edit standard LLM prompt logic inside this analysis helper.

### Dependencies
- BRM-S1-001 (Stable Core setup).

### Inputs
- Generated text completions.
- Array of retrieved text source chunks.

### Steps
1. Parse generated strings into discrete sentences.
2. Tokenize and clean sentences to compare word overlaps with retrieved chunks.
3. Generate evidence overlap map details specifying the exact context chunks validating each generated sentence.

### Outputs
- `src/cognitive/evidence.rs` implementing overlap matching logic and structural maps.

### Acceptance Criteria
- [x] Accurate mapping identifies validating source chunks for overlapping text blocks.
- [x] Unvalidated sentences are cleanly identified with empty overlaps.

### Failure Modes
- Empty retrieved arrays return empty overlap maps safely without crashes.

### Rollback
- Return ungrounded/unknown verification state in case of mapping processing faults.

### Tests
- Unit: `cognitive::evidence::tests::test_evidence_overlap_mapping_generation`

### Regression Risks
- String processing overheads for very long generated context windows.

---

### Task ID
BRM-S3-004

### Title
Citation Grounding Evaluation

### Objective
Verify that assertions in generated answers are grounded in reference context by computing citation overlap percentages.

### Scope
- `src/cognitive/dashboard_ops.rs`

### Non-Goals
- Do not render dashboards or user interfaces in this data aggregation task.

### Dependencies
- BRM-S3-003 (Evidence mapping).

### Inputs
- Evidence overlap maps.
- Target citation rules.

### Steps
1. Loop over parsed completions.
2. Measure the grounding ratio (grounded sentences vs. total sentences).
3. Compute precise citation metrics and telemetry profiles.

### Outputs
- Grounding ratios and citation evaluations integrated into telemetry.

### Acceptance Criteria
- [x] Computes exact, repeatable citation evaluation numbers between 0.0 and 1.0.
- [x] Highlights citation grounding status correctly.

### Failure Modes
- Return 0.0 grounding score if no reference context is loaded.

### Rollback
- Revert evaluation to pass/fail boolean default check.

### Tests
- Unit: `cognitive::dashboard_ops::tests::test_evidence_citation_grounding_eval`

### Regression Risks
- Slight timing overhead on query responses due to evaluation steps.

---

### Task ID
BRM-S3-005

### Title
Multi-Hop Retrieval and Goal Graph Pre-Filtering

### Objective
Create a goal graph system capable of routing complex queries by dynamically executing multi-hop context retrieval and pre-filtering based on target predicates.

### Scope
- `src/cognitive/goal_graph.rs`
- `src/cognitive/research.rs`

### Non-Goals
- Do not build user-visible visualization graphs in this backend scheduler layer.

### Dependencies
- BRM-S3-002 (Hybrid retrieval integration).

### Inputs
- Structured query goals.
- Predicate filtering rules.

### Steps
1. Parse search requests into hierarchical sub-goal nodes.
2. Query vector caches, resolving dependencies sequentially (multi-hop execution).
3. Implement metadata pre-filtering to bound the lookup scope before matching.

### Outputs
- Goal graph pipeline with multi-hop retrieval and pre-filtering logic.

### Acceptance Criteria
- [x] Resolves multi-hop query routes within the configured limits.
- [x] Graph pre-filtering accurately filters items according to metadata filters.

### Failure Modes
- If maximum hop limits are reached, abort search loops safely and return current best matches.

### Rollback
- Degrade dynamically to standard single-step retrieval.

### Tests
- Unit: `cognitive::goal_graph::tests::test_goal_graph_decomposition_and_execution`
- Unit: `cognitive::goal_graph::tests::test_goal_graph_max_hops_bound`
- Unit: `cognitive::research::tests::test_temporal_graph_filtering`
- Unit: `cognitive::research::tests::test_wave_guided_resonance_search`

### Regression Risks
- Execution timeouts if graph queries trigger endless cyclic hops.
