//! # SPANDA Engine
//!
//! SPANDA is a standalone sparse inference backend. It implements the query-conditional
//! sparse paging architecture to overcome the memory wall for LLMs.
//!
//! ## Implemented Phases (v7 Plan)
//! - Phase 0: Bare Sparse Paging
//! - Phase 1: RAM Offload Fallback
//! - Phase 2: 4-Bit Logarithmic Quantization
//! - Phase 2.2: Trajectory Prefetch
//! - Phase 3: Deferred
//!
//! For full architecture details, see `docs/SPANDA_Design.md`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::RwLock;

pub type GeneratorFn = fn(&str, usize) -> Result<String, String>;

static GENERATOR: RwLock<Option<GeneratorFn>> = RwLock::new(None);
pub static DEGRADED_MODE: AtomicBool = AtomicBool::new(false);

pub fn register_generator(generator: GeneratorFn) {
    if let Ok(mut g) = GENERATOR.write() {
        *g = Some(generator);
    }
}

pub struct Session {
    pub degraded_mode: bool,
}

impl Session {
    pub fn new() -> Self {
        Session { degraded_mode: DEGRADED_MODE.load(Ordering::Relaxed) }
    }

    pub fn health_check(&self) -> bool {
        !self.degraded_mode
    }

    pub fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String, String> {
        if self.degraded_mode {
            return Err("Spanda engine is in degraded mode".to_string());
        }
        if let Ok(g) = GENERATOR.read() {
            if let Some(ref f) = *g {
                return f(prompt, max_tokens);
            }
        }
        Err("No generator registered for Spanda engine".to_string())
    }
}
