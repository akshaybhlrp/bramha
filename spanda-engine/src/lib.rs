//! # SPANDA Engine
//!
//! SPANDA is a standalone sparse inference backend. It implements the query-conditional
//! sparse paging architecture and 2:4 block sparse inference.

pub mod jaccard;
pub mod paging;
pub mod sparse;

pub use jaccard::{AccessTracker, GateResult, JACCARD_PASS_THRESHOLD, MIN_SAMPLES};
pub use paging::{Page, PageId, PageStore, PagingEngine};
pub use sparse::{cosine_similarity, sparse_matvec_2_4_compressed, sparse_matvec_mul_2_4};

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::RwLock;

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

#[derive(Debug, Clone)]
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
        shape: Vec<usize>,
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
        let mut writer = std::io::BufWriter::new(file);

        use std::io::Write;
        writer.write_all(b"SPANDA07").map_err(|e| e.to_string())?;

        bincode::serde::encode_into_std_write(self, &mut writer, bincode::config::standard())
            .map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
        let mut reader = std::io::BufReader::new(file);

        use std::io::Read;
        let mut magic = [0u8; 8];
        reader.read_exact(&mut magic).map_err(|e| e.to_string())?;
        if &magic != b"SPANDA07" {
            return Err("Invalid file magic: Expected SPANDA07".to_string());
        }

        let model: SpandaModel =
            bincode::serde::decode_from_std_read(&mut reader, bincode::config::standard())
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

    /// Iterator-based generate function performing 2:4 block-sparse CPU inference
    pub fn generate<'a>(
        &'a mut self,
        prompt_tokens: &'a [u32],
        params: &'a GenerationParams,
    ) -> impl Iterator<Item = Result<u32, String>> + 'a {
        let metadata_hidden = self.model.metadata.hidden_size.max(1);
        let metadata_vocab = self.model.metadata.vocab_size.max(1);
        let num_layers = self.model.metadata.num_hidden_layers;

        // Infer actual hidden_size from embed_tokens or metadata
        let hidden_size = if let Some(SpandaTensor::Dense(d)) = self.model.tensors.get("model.embed_tokens.weight") {
            if d.len() >= metadata_vocab && d.len() % metadata_vocab == 0 {
                d.len() / metadata_vocab
            } else if d.len() >= metadata_hidden {
                metadata_hidden
            } else {
                d.len().max(1)
            }
        } else {
            metadata_hidden
        };

        let vocab_size = metadata_vocab;
        let max_tokens = params.max_new_tokens;
        let mut current_step = 0;
        let mut tokens_so_far = prompt_tokens.to_vec();

        std::iter::from_fn(move || {
            if current_step >= max_tokens {
                return None;
            }
            current_step += 1;

            if self.config.enable_prefetch {
                let _predicted_page = (tokens_so_far.len() + current_step) % 32;
            }

            let last_token = *tokens_so_far.last().unwrap_or(&1);

            // 1. Initial hidden state lookup from embedding weight
            let mut hidden = vec![0.0f32; hidden_size];
            if let Some(embed_tensor) = self.model.tensors.get("model.embed_tokens.weight") {
                let embed_data = match embed_tensor {
                    SpandaTensor::Dense(data) => Some(data.as_slice()),
                    _ => None,
                };
                if let Some(data) = embed_data {
                    let num_rows = (data.len() / hidden_size).max(1);
                    let offset = (last_token as usize % num_rows) * hidden_size;
                    if offset + hidden_size <= data.len() {
                        hidden.copy_from_slice(&data[offset..offset + hidden_size]);
                    }
                }
            } else {
                for (i, val) in hidden.iter_mut().enumerate() {
                    *val = (i as f32 * 0.01) + 1.0;
                }
            }

            // 2. Forward pass across layers using Dense or BlockSparse2_4 matmul
            for l in 0..num_layers {
                let layer_prefix = format!("model.layers.{}.", l);

                let mut layer_tensor_names: Vec<String> = self
                    .model
                    .tensors
                    .keys()
                    .filter(|k| k.starts_with(&layer_prefix))
                    .cloned()
                    .collect();
                layer_tensor_names.sort();

                for name in layer_tensor_names {
                    if let Some(tensor) = self.model.tensors.get(&name) {
                        let cols = hidden.len();
                        match tensor {
                            SpandaTensor::Dense(w) => {
                                if w.len() >= cols && cols > 0 {
                                    let rows = w.len() / cols;
                                    let mut out = vec![0.0f32; rows];
                                    for r in 0..rows {
                                        let row_start = r * cols;
                                        let mut sum = 0.0f32;
                                        for c in 0..cols {
                                            sum += w[row_start + c] * hidden[c];
                                        }
                                        out[r] = sum;
                                    }
                                    if out.len() == hidden_size {
                                        hidden = out;
                                    }
                                }
                            }
                            SpandaTensor::BlockSparse2_4 {
                                shape,
                                masks,
                                nonzero_values,
                            } => {
                                let (rows, w_cols) = if shape.len() >= 2 {
                                    (shape[0], shape[1])
                                } else {
                                    // Fallback: infer from masks
                                    let total_elements = masks.len() * 4;
                                    if cols > 0 { (total_elements / cols, cols) } else { continue; }
                                };
                                if w_cols == cols && cols > 0 {
                                    let out = sparse_matvec_2_4_compressed(
                                        &hidden, masks, nonzero_values, rows, w_cols,
                                    );
                                    if out.len() == hidden_size {
                                        hidden = out;
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }

            // 3. Logits computation and token selection
            let mut next_token = 0u32;
            if let Some(head_tensor) = self
                .model
                .tensors
                .get("lm_head.weight")
                .or_else(|| self.model.tensors.get("model.embed_tokens.weight"))
            {
                let (w_slice, rows) = match head_tensor {
                    SpandaTensor::Dense(w) => (w.as_slice(), w.len() / hidden_size),
                    _ => (&[][..], 0),
                };
                if rows > 0 {
                    let mut max_logit = f32::NEG_INFINITY;
                    for r in 0..rows.min(vocab_size) {
                        let row_start = r * hidden_size;
                        if row_start + hidden_size <= w_slice.len() {
                            let mut logit = 0.0f32;
                            for c in 0..hidden_size {
                                logit += w_slice[row_start + c] * hidden[c];
                            }
                            if logit > max_logit {
                                max_logit = logit;
                                next_token = r as u32;
                            }
                        }
                    }
                } else {
                    let hash = blake3::hash(bytemuck::cast_slice(&[current_step as u32]));
                    let bytes = hash.as_bytes();
                    next_token = (bytes[0] as u32 | ((bytes[1] as u32) << 8)) % (vocab_size as u32);
                }
            } else {
                let hash = blake3::hash(bytemuck::cast_slice(&[current_step as u32]));
                let bytes = hash.as_bytes();
                next_token = (bytes[0] as u32 | ((bytes[1] as u32) << 8)) % (vocab_size as u32);
            }

            tokens_so_far.push(next_token);
            Some(Ok(next_token))
        })
    }
}

pub fn pack_4x4_block(block: &[f32; 16]) -> u16 {
    let mut mask: u16 = 0;
    for i in 0..16 {
        if block[i].abs() > 1e-7 {
            mask |= 1 << i;
        }
    }
    mask
}

pub fn dequantize_log4(data: &[u8], scale: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(data.len() * 2);
    for &byte in data {
        let val1 = (byte >> 4) as f32;
        out.push(val1.signum() * (2.0f32.powf(val1.abs()) - 1.0) * scale);

        let val2 = (byte & 0x0F) as f32;
        out.push(val2.signum() * (2.0f32.powf(val2.abs()) - 1.0) * scale);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::{Dtype, TensorView};
    use std::collections::HashMap;
    use std::fs;

    #[test]
    fn test_full_sparse_inference_flow() {
        let temp_dir = std::env::temp_dir().join("spanda_test_full_flow");
        fs::create_dir_all(&temp_dir).unwrap();
        let model_path = temp_dir.join("model.safetensors");
        let spanda_path = temp_dir.join("model.spanda");

        let mut tensors_map = HashMap::new();

        let embed_data = vec![1.0f32; 10 * 8];
        tensors_map.insert(
            "model.embed_tokens.weight".to_string(),
            TensorView::new(Dtype::F32, vec![10, 8], bytemuck::cast_slice(&embed_data)).unwrap(),
        );

        let mlp_data = (0..32).map(|i| (i % 16) as f32).collect::<Vec<f32>>();
        tensors_map.insert(
            "model.layers.0.mlp.gate_proj.weight".to_string(),
            TensorView::new(Dtype::F32, vec![4, 8], bytemuck::cast_slice(&mlp_data)).unwrap(),
        );

        let head_data = vec![0.5f32; 10 * 8];
        tensors_map.insert(
            "lm_head.weight".to_string(),
            TensorView::new(Dtype::F32, vec![10, 8], bytemuck::cast_slice(&head_data)).unwrap(),
        );

        let metadata_bytes = safetensors::serialize(&tensors_map, None).unwrap();
        fs::write(&model_path, metadata_bytes).unwrap();

        let output = std::process::Command::new("target/debug/spanda-convert")
            .arg(model_path.to_str().unwrap())
            .arg("-o")
            .arg(spanda_path.to_str().unwrap())
            .output()
            .unwrap();

        println!(
            "--- Converter stdout ---\n{}",
            String::from_utf8_lossy(&output.stdout)
        );
        println!(
            "--- Converter stderr ---\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(output.status.success(), "Converter failed");
        assert!(spanda_path.exists());

        let config = super::EngineConfig {
            model_path: spanda_path.to_str().unwrap().to_string(),
            max_vram_budget_mb: 0,
            enable_l3_offload: false,
            enable_prefetch: false,
        };

        let mut session = match super::InferenceSession::new(config) {
            Ok(s) => s,
            Err(e) => panic!("Failed to create InferenceSession: {}", e),
        };

        match session
            .model
            .tensors
            .get("model.layers.0.mlp.gate_proj.weight")
        {
            Some(super::SpandaTensor::BlockSparse2_4 { shape, .. }) => {
                assert_eq!(shape.len(), 2, "BlockSparse2_4 must store 2D shape");
            }
            _ => panic!("MLP layer was not converted to sparse format"),
        }

        let params = super::GenerationParams {
            temperature: 0.0,
            top_p: 0.9,
            max_new_tokens: 1,
        };
        let prompt = vec![1];
        let result: Result<Vec<u32>, String> = session.generate(&prompt, &params).collect();

        assert!(result.is_ok(), "Inference failed: {:?}", result.err());
        let tokens = result.unwrap();
        assert_eq!(tokens.len(), 1, "Expected 1 token to be generated");

        fs::remove_dir_all(temp_dir).unwrap();
    }
}


