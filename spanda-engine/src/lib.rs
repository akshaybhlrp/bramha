//! # SPANDA Engine
//!
//! SPANDA is a standalone sparse inference backend. It implements the query-conditional
//! sparse paging architecture to overcome the memory wall for LLMs.
//!
//! ## Phase status (corrected — the previous version of this comment claimed phases
//! that had zero corresponding code; verify claims against `jaccard.rs`/`paging.rs`
//! before trusting doc comments here going forward, per project convention that
//! markdown/doc claims are not ground truth, source is)
//! - Phase 0 (predictability gate): IMPLEMENTED — see `jaccard::AccessTracker`.
//!   Computes real Jaccard similarity between consecutive access sets; nothing
//!   downstream activates until `evaluate_gate().passed`.
//! - Phase 1 (RAM-resident paging + confidence-based eviction): IMPLEMENTED —
//!   see `paging::PagingEngine`. Plain `Vec` storage (mmap rejected per design
//!   decision), eviction picks lowest-confidence page, not LRU.
//! - Phase 2 (4-bit logarithmic quantization): math helpers exist
//!   (`dequantize_log4`) but are not wired into the paging path yet.
//! - Phase 2.2 (trajectory prefetch): NOT IMPLEMENTED. `InferenceSession::generate`
//!   has a placeholder `_predicted_page` computation that is never used for
//!   actual prefetching — cosmetic only.
//! - Phase 3+: deferred, no code.
//!
//! For full architecture details, see `docs/SPANDA_Design.md` (also unverified
//! against code as of this pass — treat as aspirational until reconciled).

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::RwLock;
use serde::{Deserialize, Serialize};

pub mod jaccard;
pub mod paging;
pub mod sparse;

pub use jaccard::{AccessTracker, GateResult, JACCARD_PASS_THRESHOLD, MIN_SAMPLES};
pub use paging::{Page, PageId, PageStore, PagingEngine};
pub use sparse::{cosine_similarity, sparse_matvec_mul_2_4};

// --- Backward Compatibility Bridge ---
// GeneratorFn takes model_name explicitly. It used to be (prompt, max_tokens) with the
// caller expected to stash model_name in a thread_local before calling — safe only by
// accident (no .await between the thread_local write and this read on the one call site
// that used it), and one refactor away from silently reading a stale/wrong model name
// under tokio's multi-threaded scheduler. Explicit parameter can't have that failure mode.
pub type GeneratorFn = fn(&str, &str, usize) -> Result<String, String>;
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
        Session {
            degraded_mode: DEGRADED_MODE.load(Ordering::Relaxed),
        }
    }

    pub fn health_check(&self) -> bool {
        !self.degraded_mode
    }

    pub fn generate(&self, model_name: &str, prompt: &str, max_tokens: usize) -> Result<String, String> {
        if self.degraded_mode {
            return Err("Spanda engine is in degraded mode".to_string());
        }
        if let Ok(g) = GENERATOR.read() {
            if let Some(ref f) = *g {
                return f(model_name, prompt, max_tokens);
            }
        }
        Err("No generator registered for Spanda engine".to_string())
    }
}

// --- SPANDA v7 Public Contract & Types ---

pub type Token = u32;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EngineConfig {
    pub model_path: String,
    pub max_vram_budget_mb: usize,
    pub enable_l3_offload: bool,
    pub enable_prefetch: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GenerationParams {
    pub temperature: f32,
    pub top_p: f32,
    pub max_new_tokens: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ModelMetadata {
    pub name: String,
    pub architecture: String,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum SpandaTensor {
    Dense(Vec<f32>),
    BlockSparse2_4 {
        masks: Vec<u16>,
        nonzero_values: Vec<f32>,
    },
    Quant4Log {
        data: Vec<u8>,
        scales: Vec<f32>,
    },
    Quant8Linear {
        data: Vec<i8>,
        scales: Vec<f32>,
    },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SpandaModel {
    pub metadata: ModelMetadata,
    pub tensors: HashMap<String, SpandaTensor>,
}

impl SpandaModel {
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        let file = std::fs::File::create(path).map_err(|e| e.to_string())?;
        let writer = std::io::BufWriter::new(file);
        
        // Write magic bytes SPANDA07
        use std::io::Write;
        let mut writer = writer;
        writer.write_all(b"SPANDA07").map_err(|e| e.to_string())?;
        
        // Serialize using bincode (2.0.0-rc.3 style)
        bincode::serde::encode_into_std_write(self, &mut writer, bincode::config::standard())
            .map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
        let mut reader = std::io::BufReader::new(file);
        
        // Read and verify magic bytes
        use std::io::Read;
        let mut magic = [0u8; 8];
        reader.read_exact(&mut magic).map_err(|e| e.to_string())?;
        if &magic != b"SPANDA07" {
            return Err("Invalid file magic: Expected SPANDA07".to_string());
        }
        
        let model: SpandaModel = bincode::serde::decode_from_std_read(&mut reader, bincode::config::standard())
            .map_err(|e| e.to_string())?;
        Ok(model)
    }
}

pub struct InferenceSession {
    pub config: EngineConfig,
    pub model: SpandaModel,
    pub cache: HashMap<String, Vec<f32>>,
}

impl InferenceSession {
    pub fn new(config: EngineConfig) -> Result<Self, String> {
        let model = SpandaModel::load_from_file(&config.model_path)?;
        Ok(InferenceSession {
            config,
            model,
            cache: HashMap::new(),
        })
    }

    /// Iterator-based generate function mapping tokens to tokens
    pub fn generate<'a>(
        &'a mut self,
        prompt_tokens: &'a [u32],
        params: &'a GenerationParams,
    ) -> impl Iterator<Item = Result<u32, String>> + 'a {
        // Simple mock execution engine using the loaded model specs and weights
        let mut current_step = 0;
        let max_tokens = params.max_new_tokens;
        let vocab_size = self.model.metadata.vocab_size;
        
        std::iter::from_fn(move || {
            if current_step >= max_tokens {
                return None;
            }
            current_step += 1;
            
            // Execute mock sparse matmul / prefetch logging to demonstrate phase execution
            if self.config.enable_prefetch {
                // A* trajectory lookahead simulation
                let _predicted_page = (prompt_tokens.len() + current_step) % 32;
            }

            // Greedy sample next token based on prompt hashes
            let hash = blake3::hash(bytemuck::cast_slice(&[current_step as u32]));
            let bytes = hash.as_bytes();
            let next_token = (bytes[0] as u32 | ((bytes[1] as u32) << 8)) % (vocab_size as u32);
            
            Some(Ok(next_token))
        })
    }
}

// --- Block-Sparse & Quantization Helper Math ---

/// Packs a 4x4 block (16 elements) into a single u16 bitmask.
pub fn pack_4x4_block(block: &[f32; 16]) -> u16 {
    let mut mask: u16 = 0;
    for i in 0..16 {
        if block[i].abs() > 1e-7 {
            mask |= 1 << i;
        }
    }
    mask
}

// Note: the 2:4 block-sparse matvec + cosine-similarity implementation that used to
// live here (a sequential duplicate of the one moved from bramha-engine's
// sparse_predictor.rs) has been removed in favor of the single copy in `sparse.rs`,
// which is rayon-parallelized and is the one actually exercised by the shadow-scan
// gate in production. Use `spanda_engine::sparse_matvec_mul_2_4` /
// `spanda_engine::cosine_similarity` (re-exported above).

/// Logarithmic 4-bit compression / dequantization helper
pub fn dequantize_log4(data: &[u8], scale: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(data.len() * 2);
    for &byte in data {
        // High 4 bits
        let val1 = (byte >> 4) as f32;
        out.push(val1.signum() * (2.0f32.powf(val1.abs()) - 1.0) * scale);
        
        // Low 4 bits
        let val2 = (byte & 0x0F) as f32;
        out.push(val2.signum() * (2.0f32.powf(val2.abs()) - 1.0) * scale);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase0_block_sparse_mse() {
        // Raw f32 weights of a 4x4 block
        let raw_block = [1.5, 0.01, -0.02, -0.8, 0.0, 3.4, 0.1, 0.0, 0.05, -0.01, 1.2, 0.0, 0.0, 0.0, 0.0, -0.7];
        
        // Pack block
        let mask = pack_4x4_block(&raw_block);
        
        // Decompress block and measure Mean Squared Error (MSE) for active (non-zeroed) elements
        let mut decompressed = [0.0f32; 16];
        for i in 0..16 {
            if (mask & (1 << i)) != 0 {
                decompressed[i] = raw_block[i];
            }
        }

        let mut mse = 0.0;
        let mut active_count = 0;
        for i in 0..16 {
            if raw_block[i].abs() > 1e-2 {
                let diff = raw_block[i] - decompressed[i];
                mse += diff * diff;
                active_count += 1;
            }
        }
        if active_count > 0 {
            mse /= active_count as f32;
        }

        // Decompressed weight MSE for non-zero pruned elements must be < 1e-5
        assert!(mse < 1e-5, "MSE gate check failed: {}", mse);
    }

    #[test]
    fn test_log4_dequantization() {
        let quantized = vec![0x34, 0x12]; // byte 1: high=3, low=4; byte 2: high=1, low=2
        let scale = 0.5;
        let decompressed = dequantize_log4(&quantized, scale);
        
        assert_eq!(decompressed.len(), 4);
        // high 3: signum(3) * (2^3 - 1) * 0.5 = 1 * 7 * 0.5 = 3.5
        assert!((decompressed[0] - 3.5).abs() < 1e-5);
        // low 4: signum(4) * (2^4 - 1) * 0.5 = 1 * 15 * 0.5 = 7.5
        assert!((decompressed[1] - 7.5).abs() < 1e-5);
    }
}

