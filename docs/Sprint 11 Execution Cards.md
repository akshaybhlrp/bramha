# Sprint 11 Execution Cards: Adaptive Learning

This document tracks the execution cards for Sprint 11, focusing on adaptive learning features: Adapter learning pipeline, memory confidence updates, and contradiction resolution.

## BRM-S11-001: Adapter Learning Pipeline
### Objective
Implement the adapter learning pipeline to manage loading, applying, and saving LoRA adapters dynamically for model specialization.
### Scope
- `src/cognitive/adapter.rs` (new module)
- `src/inference/engine.rs` (integration)
### Inputs
- Existing model loading flow.
### Steps
1. Define `AdapterMetadata` struct to track adapter configurations (rank, alpha, target modules).
2. Implement an `AdapterManager` to handle loading LoRA weights.
3. Integrate adapter loading into the `InferenceEngine` initialization.
4. Add unit tests for adapter creation and metadata parsing.
### Acceptance Criteria
- [x] AdapterManager can parse adapter config and load weight metadata.
- [x] InferenceEngine can optionally initialize with an adapter path.
- [x] Unit tests pass for adapter config parsing and metadata extraction.

## BRM-S11-002: Memory Confidence Updates
### Objective
Automatically adjust memory node confidence scores based on retrieval frequency and feedback.
### Scope
- `src/cognitive/memory.rs`
### Inputs
- Existing memory node schema.
### Steps
1. Add a `confidence_score` (f32) field to memory entities if missing.
2. Implement a `reinforce_memory` function that increases confidence on successful retrieval.
3. Implement a `decay_memory` function that lowers confidence over time or unused sessions.
4. Add unit tests for the reinforcement and decay logic.
### Acceptance Criteria
- [x] Repeated retrieval of a memory node increases its confidence score.
- [x] Memory decay function lowers confidence score based on time delta.
- [x] Unit tests pass.

## BRM-S11-003: Contradiction Resolution
### Objective
Detect contradictions in semantic memory and flag them for human review or automatic resolution based on confidence scores.
### Scope
- `src/cognitive/memory.rs`
- `src/cognitive/research.rs`
### Inputs
- Existing semantic memory retrieval.
### Steps
1. Implement a `detect_contradiction` function that compares a new fact against highly retrieved semantic memories.
2. If a contradiction is detected, flag the new node and emit an event.
3. Add unit tests for contradiction detection logic.
### Acceptance Criteria
- [x] Contradictory facts trigger a flag and event.
- [x] Unit tests pass.

## BRM-S11-004: Engine & Tokenizer Stabilization
### Objective
Stabilize the inference engine by resolving infinite generation loops, fixing chat template rendering via minijinja, and optimizing Cargo compilation profiles.
### Scope
- `src/inference/engine.rs`
- `src/inference/cpu_engine.rs`
- `src/inference/tokenizer.rs`
- `Cargo.toml`
### Inputs
- WGPU testing suite and models demonstrating "garbage" generation loops.
### Steps
1. Fix tokenizer configuration loading to inject `added_tokens` into the active vocabulary dynamically.
2. Extend hardcoded EOS checks (e.g. ID `2`) to include multi-model architecture stops (`151645`, `151643`).
3. Add `minijinja` and refactor the `BramhaTokenizer` to dynamically render `chat_template` variables from `tokenizer_config.json`.
4. Replace hardcoded prompt building across `engine.rs` and `cpu_engine.rs` with `apply_chat_template`.
5. Optimize `Cargo.toml` dev profile by disabling main-crate optimizations and enabling max codegen units.
### Acceptance Criteria
- [x] Qwen and Llama models terminate properly without rambling indefinitely into unrelated dialog paths.
- [x] The `minijinja` executor properly maps roles (`system`, `user`) into native tokens without hardcoded logic.
- [x] Compilation time significantly reduced for iterative testing cycles.
- [x] WGPU test panics completely eliminated.
