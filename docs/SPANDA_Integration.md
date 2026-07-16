# SPANDA Integration Contract

This document outlines how the Bramha Neural Engine integrates with the `spanda-engine` crate.

## The Contract
The integration between Bramha and SPANDA is a **Rust-only contract**. SPANDA acts as a standalone inference backend, while Bramha functions as the database-native intelligence orchestrator.

### Dependency Management
Bramha consumes `spanda-engine` as a versioned dependency in its `Cargo.toml`. Bramha locks to specific SPANDA releases rather than relying on a floating branch to guarantee stability and reproduction.

```toml
# bramha-engine/Cargo.toml
[dependencies]
spanda-engine = { version = "0.7.0", path = "../spanda-engine" }
```

### Public API
The primary interaction point is the `spanda::InferenceSession` struct. Bramha initializes this session and invokes its public methods to execute generation steps.

#### Initializing the Engine
```rust
use spanda_engine::{EngineConfig, InferenceSession};

fn setup_spanda_backend(model_path: &str) -> InferenceSession {
    let config = EngineConfig {
        model_path: model_path.to_string(),
        max_vram_budget_mb: 4096, // E.g. limit to 4GB
        enable_l3_offload: true,  // Fallback Phase 1
        enable_prefetch: true,    // Phase 2.2 prefetch
    };
    
    InferenceSession::new(config)
        .expect("Failed to initialize SPANDA engine")
}
```

#### Running Generation
Bramha's `InferenceOrchestrator` delegates the generation loop to the SPANDA session:

```rust
use spanda_engine::{GenerationParams, Token};

fn generate_response(session: &mut InferenceSession, prompt_tokens: &[Token]) -> Vec<Token> {
    let params = GenerationParams {
        temperature: 0.7,
        top_p: 0.9,
        max_new_tokens: 256,
    };
    
    let mut generated = Vec::new();
    
    // The generate method handles query-conditional sparse paging internally
    for result in session.generate(prompt_tokens, &params) {
        match result {
            Ok(token) => {
                generated.push(token);
                // Can stream token to client here
            },
            Err(e) => {
                eprintln!("Inference error: {:?}", e);
                break;
            }
        }
    }
    
    generated
}
```

## Conversion Requirement
Bramha expects models in the `.spanda` format. End users or orchestration scripts must run the `spanda-convert` binary on their models before ingesting them into Bramha.

```bash
spanda-convert model.safetensors -o model.spanda
```
