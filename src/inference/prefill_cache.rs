use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use sha2::{Sha256, Digest};
use burn::tensor::{Tensor, Shape, Data, activation::softmax};
use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayDevice;

use crate::inference::tokenizer::BramhaTokenizer;
use crate::inference::engine::{
    rms_norm, precompute_rope_freqs, apply_rope, repeat_kv, causal_mask,
};
use crate::storage::Database;
use crate::storage::cache_db::KvCacheEntry;

type B = NdArray;

pub struct PrefillCacheManager;

impl PrefillCacheManager {
    /// Returns the subdirectory where prefill caches are stored for a given model
    pub fn get_cache_dir(base_path: &Path) -> PathBuf {
        base_path.join("prefill_cache")
    }

    /// Computes the SHA-256 hash of a prompt
    pub fn compute_prompt_hash(prompt: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(prompt.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Returns the absolute path of a cache entry on disk
    pub fn get_cache_file_path(base_path: &Path, prompt: &str) -> PathBuf {
        let cache_dir = Self::get_cache_dir(base_path);
        let hash = Self::compute_prompt_hash(prompt);
        cache_dir.join(format!("{}.bin", hash))
    }

    /// Checks if a prefill cache exists for the given prompt
    pub fn exists(base_path: &Path, prompt: &str) -> bool {
        let path = Self::get_cache_file_path(base_path, prompt);
        path.exists()
    }

    /// Loads a prefill cache entry from disk
    pub fn load(base_path: &Path, prompt: &str) -> Result<KvCacheEntry, String> {
        let path = Self::get_cache_file_path(base_path, prompt);
        if !path.exists() {
            return Err(format!("Cache file not found for prompt at {:?}", path));
        }

        let mut file = File::open(&path).map_err(|e| format!("Failed to open prefill cache file: {}", e))?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).map_err(|e| format!("Failed to read prefill cache file: {}", e))?;

        let config = bincode::config::standard();
        let entry: KvCacheEntry = bincode::serde::decode_from_slice(&buffer, config)
            .map_err(|e| format!("Failed to deserialize prefill cache: {}", e))?.0;

        Ok(entry)
    }

    /// Saves a prefill cache entry to disk
    pub fn save(base_path: &Path, prompt: &str, entry: KvCacheEntry) -> Result<(), String> {
        let cache_dir = Self::get_cache_dir(base_path);
        fs::create_dir_all(&cache_dir).map_err(|e| format!("Failed to create prefill cache directory: {}", e))?;

        let path = Self::get_cache_file_path(base_path, prompt);
        let config = bincode::config::standard();
        let encoded = bincode::serde::encode_to_vec(&entry, config)
            .map_err(|e| format!("Failed to serialize prefill cache: {}", e))?;

        let mut file = File::create(&path).map_err(|e| format!("Failed to create prefill cache file: {}", e))?;
        file.write_all(&encoded).map_err(|e| format!("Failed to write prefill cache file: {}", e))?;

        Ok(())
    }

    /// Precomputes the KV cache of the prompt using pure CPU NdArray backend and stores it
    pub async fn prefill_and_cache(
        db: Arc<Database>,
        model_name: &str,
        prompt: &str,
    ) -> Result<KvCacheEntry, String> {
        let device = NdArrayDevice::Cpu;

        // 1. Fetch Model Table
        let tensor_db_guard = db.tensor_db.read().await;
        let model = tensor_db_guard.models.get(model_name)
            .ok_or_else(|| format!("Model '{}' not found in database. Ingest model first.", model_name))?;

        // 2. Tokenize prompt
        let bramha_tokenizer = BramhaTokenizer::load(model_name, &model.base_path)?;
        let tokens = bramha_tokenizer.encode(prompt, true)?;

        if tokens.is_empty() {
            return Err("Cannot prefill cache for an empty prompt".to_string());
        }

        println!("⚡ Prefilling prompt tokens (len: {}): {:?}", tokens.len(), tokens);

        let seq_len = tokens.len();
        let is_mistral = model_name.to_lowercase().contains("mistral");
        let (num_layers, head_dim, num_q_heads, num_kv_heads, hidden_size) = if is_mistral {
            (32, 128, 32, 8, 4096)
        } else {
            (22, 64, 32, 4, 2048)
        };

        // Helper load closure
        let load_tensor_1d = |name: &str| -> Result<Tensor<B, 1>, String> {
            let page = model.layers.get(name)
                .ok_or_else(|| format!("Weight not found: {}", name))?;
            let f32_data: &[f32] = bytemuck::cast_slice(page.as_bytes());
            let expected_len = page.shape[0];
            if f32_data.len() != expected_len {
                return Err(format!("Quantized or mismatch 1D weight size for {}: got {}, expected {}", name, f32_data.len(), expected_len));
            }
            let data = Data::new(f32_data.to_vec(), Shape::from([page.shape[0]])).convert();
            Ok(Tensor::<B, 1>::from_data(data, &device))
        };

        let load_tensor_2d = |name: &str| -> Result<Tensor<B, 2>, String> {
            let page = model.layers.get(name)
                .ok_or_else(|| format!("Weight not found: {}", name))?;
            let f32_data: &[f32] = bytemuck::cast_slice(page.as_bytes());
            let expected_len = page.shape[0] * page.shape[1];
            if f32_data.len() != expected_len {
                return Err(format!("Quantized or mismatch 2D weight size for {}: got {}, expected {}", name, f32_data.len(), expected_len));
            }
            let data = Data::new(f32_data.to_vec(), Shape::from([page.shape[0], page.shape[1]])).convert();
            Ok(Tensor::<B, 2>::from_data(data, &device))
        };

        // Precompute RoPE
        let (cos, sin) = precompute_rope_freqs::<B>(seq_len, head_dim, 10000.0, &device);

        // Embedding lookup
        let embed_w = load_tensor_2d("model.embed_tokens.weight")?;
        let tokens_data = Data::new(
            tokens.iter().map(|&t| t as i32).collect::<Vec<i32>>(),
            Shape::from([seq_len]),
        ).convert();
        let tokens_tensor = Tensor::<B, 1, burn::tensor::Int>::from_data(tokens_data, &device);
        let mut x = embed_w.select(0, tokens_tensor);

        let mut cached_keys = Vec::new();
        let mut cached_values = Vec::new();

        // Decoder Layer Loop
        for layer_idx in 0..num_layers {
            // RMSNorm 1
            let norm1_w = load_tensor_1d(&format!("model.layers.{}.input_layernorm.weight", layer_idx))?;
            let h = rms_norm(x.clone(), norm1_w, 1e-5);

            // Self Attention Projections
            let (q_res, (k_res, v_res)) = tokio::task::block_in_place(|| {
                rayon::join(
                    || load_tensor_2d(&format!("model.layers.{}.self_attn.q_proj.weight", layer_idx)),
                    || rayon::join(
                        || load_tensor_2d(&format!("model.layers.{}.self_attn.k_proj.weight", layer_idx)),
                        || load_tensor_2d(&format!("model.layers.{}.self_attn.v_proj.weight", layer_idx)),
                    ),
                )
            });
            let q_proj_w = q_res?;
            let k_proj_w = k_res?;
            let v_proj_w = v_res?;

            let q = h.clone().matmul(q_proj_w.transpose());
            let k = h.clone().matmul(k_proj_w.transpose());
            let v = h.matmul(v_proj_w.transpose());

            // Reshape for attention heads
            let q = q.reshape(Shape::from([seq_len, num_q_heads, head_dim]));
            let k = k.reshape(Shape::from([seq_len, num_kv_heads, head_dim]));
            let v = v.reshape(Shape::from([seq_len, num_kv_heads, head_dim]));

            // Apply RoPE
            let q = apply_rope(q, cos.clone(), sin.clone());
            let k = apply_rope(k, cos.clone(), sin.clone());

            // Flatten K and V and add to cache
            let k_flat = k.clone().into_data().value;
            let v_flat = v.clone().into_data().value;
            cached_keys.push(k_flat);
            cached_values.push(v_flat);

            // GQA head repeating
            let k = repeat_kv(k, num_q_heads / num_kv_heads);
            let v = repeat_kv(v, num_q_heads / num_kv_heads);

            // Permute for batch attention
            let q = q.swap_dims(0, 1);
            let k = k.swap_dims(0, 1);
            let v = v.swap_dims(0, 1);

            // Attention score calculation
            let scale = 1.0 / (head_dim as f32).sqrt();
            let scores = q.matmul(k.swap_dims(1, 2)).mul_scalar(scale);
            let mask = causal_mask::<B>(seq_len, &device).unsqueeze_dim::<3>(0);
            let probs = softmax(scores.add(mask), 2);
            let context = probs.matmul(v);

            let context = context.swap_dims(0, 1).reshape(Shape::from([seq_len, hidden_size]));

            // Output projection
            let o_proj_w = load_tensor_2d(&format!("model.layers.{}.self_attn.o_proj.weight", layer_idx))?;
            let attn_out = context.matmul(o_proj_w.transpose());

            let x_attn = x.add(attn_out);

            // RMSNorm 2
            let norm2_w = load_tensor_1d(&format!("model.layers.{}.post_attention_layernorm.weight", layer_idx))?;
            let h2 = rms_norm(x_attn.clone(), norm2_w, 1e-5);

            // SwiGLU MLP
            let (gate_res, (up_res, down_res)) = tokio::task::block_in_place(|| {
                rayon::join(
                    || load_tensor_2d(&format!("model.layers.{}.mlp.gate_proj.weight", layer_idx)),
                    || rayon::join(
                        || load_tensor_2d(&format!("model.layers.{}.mlp.up_proj.weight", layer_idx)),
                        || load_tensor_2d(&format!("model.layers.{}.mlp.down_proj.weight", layer_idx)),
                    ),
                )
            });
            let gate_proj_w = gate_res?;
            let up_proj_w = up_res?;
            let down_proj_w = down_res?;

            let gate = h2.clone().matmul(gate_proj_w.transpose());
            let silu_gate = gate.clone().mul(burn::tensor::activation::sigmoid(gate));
            let up = h2.matmul(up_proj_w.transpose());
            let mlp_h = silu_gate.mul(up);
            let mlp_out = mlp_h.matmul(down_proj_w.transpose());

            x = x_attn.add(mlp_out);
        }

        // Save entry
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let entry = KvCacheEntry {
            session_id: Self::compute_prompt_hash(prompt),
            tokens,
            keys: cached_keys,
            values: cached_values,
            last_accessed: now,
            ttl_expiry: now + 365 * 24 * 3600 * 1000, // Prefill caches don't expire for a year
        };

        Self::save(&model.base_path, prompt, entry.clone())?;

        Ok(entry)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prompt_hash_consistency() {
        let prompt = "<|system|>\nYou are a helpful AI assistant.</s>\n";
        let hash1 = PrefillCacheManager::compute_prompt_hash(prompt);
        let hash2 = PrefillCacheManager::compute_prompt_hash(prompt);
        assert_eq!(hash1, hash2);

        let diff_prompt = "<|system|>\nYou are a funny pirate assistant.</s>\n";
        let hash3 = PrefillCacheManager::compute_prompt_hash(diff_prompt);
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_prefill_cache_persistence_roundtrip() {
        let temp_dir = std::env::temp_dir().join("bramha_test_prefill_cache");
        fs::create_dir_all(&temp_dir).unwrap();

        let prompt = "Test prompt for roundtrip persistence";
        let mock_keys = vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]];
        let mock_values = vec![vec![1.1, 1.2, 1.3], vec![1.4, 1.5, 1.6]];
        let entry = KvCacheEntry {
            session_id: PrefillCacheManager::compute_prompt_hash(prompt),
            tokens: vec![42, 1337],
            keys: mock_keys.clone(),
            values: mock_values.clone(),
            last_accessed: 1000,
            ttl_expiry: 2000,
        };

        // Initially not exists
        assert!(!PrefillCacheManager::exists(&temp_dir, prompt));

        // Save
        PrefillCacheManager::save(&temp_dir, prompt, entry).unwrap();

        // Should exist now
        assert!(PrefillCacheManager::exists(&temp_dir, prompt));

        // Load and verify
        let loaded = PrefillCacheManager::load(&temp_dir, prompt).unwrap();
        assert_eq!(loaded.tokens, vec![42, 1337]);
        assert_eq!(loaded.keys, mock_keys);
        assert_eq!(loaded.values, mock_values);

        // Clean up
        let _ = fs::remove_dir_all(&temp_dir);
    }
}
