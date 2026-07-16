use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub trait ModelAdapter {
    fn name(&self) -> &str;
    fn num_layers(&self) -> usize;
    fn hidden_dim(&self) -> usize;
    fn rope_theta(&self) -> f32;
    fn max_context_length(&self) -> usize;
    fn get_early_exit_bounds(&self) -> (usize, usize);
    fn supported_tasks(&self) -> Vec<&'static str>;
    fn native_precision(&self) -> &'static str;
}

pub struct LlamaAdapter {
    pub name: String,
    pub num_layers: usize,
    pub hidden_dim: usize,
    pub rope_theta: f32,
    pub max_context_length: usize,
}

impl Default for LlamaAdapter {
    fn default() -> Self {
        LlamaAdapter {
            name: "LLaMA-3-8B".to_string(),
            num_layers: 32,
            hidden_dim: 4096,
            rope_theta: 500000.0,
            max_context_length: 8192,
        }
    }
}

impl ModelAdapter for LlamaAdapter {
    fn name(&self) -> &str {
        &self.name
    }

    fn num_layers(&self) -> usize {
        self.num_layers
    }

    fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    fn rope_theta(&self) -> f32 {
        self.rope_theta
    }

    fn max_context_length(&self) -> usize {
        self.max_context_length
    }

    fn get_early_exit_bounds(&self) -> (usize, usize) {
        (8, 24)
    }

    fn supported_tasks(&self) -> Vec<&'static str> {
        vec!["rag", "reasoning", "summarize", "code"]
    }

    fn native_precision(&self) -> &'static str {
        "F32"
    }
}

pub struct GemmaAdapter {
    pub name: String,
    pub num_layers: usize,
    pub hidden_dim: usize,
    pub rope_theta: f32,
    pub max_context_length: usize,
}

impl Default for GemmaAdapter {
    fn default() -> Self {
        GemmaAdapter {
            name: "Gemma-2-9B".to_string(),
            num_layers: 42,
            hidden_dim: 3584,
            rope_theta: 10000.0,
            max_context_length: 8192,
        }
    }
}

impl ModelAdapter for GemmaAdapter {
    fn name(&self) -> &str {
        &self.name
    }

    fn num_layers(&self) -> usize {
        self.num_layers
    }

    fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    fn rope_theta(&self) -> f32 {
        self.rope_theta
    }

    fn max_context_length(&self) -> usize {
        self.max_context_length
    }

    fn get_early_exit_bounds(&self) -> (usize, usize) {
        (10, 30)
    }

    fn supported_tasks(&self) -> Vec<&'static str> {
        vec!["rag", "fast-decode", "summarize"]
    }

    fn native_precision(&self) -> &'static str {
        "INT8"
    }
}

/// ModelCapabilityRegistry manages registered adapters and selects the best model for a task.
pub struct ModelCapabilityRegistry {
    pub adapters: HashMap<String, Box<dyn ModelAdapter + Send + Sync>>,
}

impl ModelCapabilityRegistry {
    pub fn new() -> Self {
        let mut registry = ModelCapabilityRegistry {
            adapters: HashMap::new(),
        };
        registry.register_adapter("llama", Box::new(LlamaAdapter::default()));
        registry.register_adapter("gemma", Box::new(GemmaAdapter::default()));
        registry
    }

    pub fn register_adapter(&mut self, key: &str, adapter: Box<dyn ModelAdapter + Send + Sync>) {
        self.adapters.insert(key.to_string(), adapter);
    }

    pub fn get_adapter(&self, key: &str) -> Option<&(dyn ModelAdapter + Send + Sync)> {
        self.adapters.get(key).map(|b| b.as_ref())
    }

    pub fn select_best_model_for_task(&self, task: &str) -> Option<String> {
        for (key, adapter) in &self.adapters {
            if adapter.supported_tasks().contains(&task) {
                return Some(key.clone());
            }
        }
        None
    }
}

/// BackendCapabilityProfile maps local hardware limits to model constraints.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BackendCapabilityProfile {
    pub device_name: String,
    pub max_vram_bytes: u64,
    pub max_storage_buffer_binding_size: u64,
    pub fp16_supported: bool,
}

impl BackendCapabilityProfile {
    pub fn new_cpu_profile() -> Self {
        BackendCapabilityProfile {
            device_name: "CPU_SIMD".to_string(),
            max_vram_bytes: 32 * 1024 * 1024 * 1024, // Assumed 32GB system memory fallback
            max_storage_buffer_binding_size: 4 * 1024 * 1024 * 1024, // 4GB max
            fp16_supported: false,
        }
    }

    pub fn new_gpu_profile(
        device_name: String,
        max_vram_bytes: u64,
        max_storage_buffer_binding_size: u64,
        fp16_supported: bool,
    ) -> Self {
        BackendCapabilityProfile {
            device_name,
            max_vram_bytes,
            max_storage_buffer_binding_size,
            fp16_supported,
        }
    }

    /// Checks if a model fits within the backend's resource limits.
    pub fn supports_model(&self, adapter: &dyn ModelAdapter) -> bool {
        // Approximate model weight size: parameters ~= layers * hidden_dim * hidden_dim * 3
        let hidden_dim = adapter.hidden_dim();
        let layers = adapter.num_layers();
        let bytes_per_param = match adapter.native_precision() {
            "INT4" => 0.5,
            "INT8" => 1.0,
            _ => 4.0, // F32
        };
        let estimated_size_bytes = (layers * hidden_dim * hidden_dim * 6) as f64 * bytes_per_param;

        // Model fits if VRAM has sufficient headroom, and individual layer tensor weights don't exceed storage buffer limits
        let layer_weight_bytes = (hidden_dim * hidden_dim) as f64 * bytes_per_param;

        estimated_size_bytes < self.max_vram_bytes as f64
            && layer_weight_bytes < self.max_storage_buffer_binding_size as f64
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterMetadata {
    pub target_modules: Vec<String>,
    pub rank: u32,
    pub alpha: f32,
}

pub struct AdapterManager {
    pub metadata: AdapterMetadata,
    pub adapter_path: std::path::PathBuf,
    // Maps layer name to (lora_A, lora_B) weights
    pub lora_weights: HashMap<String, (Vec<f32>, Vec<f32>)>,
    pub hidden_dim: usize,
}

impl AdapterManager {
    pub fn load_from_config(path: &str) -> Result<Self, String> {
        let metadata = AdapterMetadata {
            target_modules: vec!["q_proj".into(), "v_proj".into()],
            rank: 16,
            alpha: 32.0,
        };
        let hidden_dim = 4096;
        let mut lora_weights = HashMap::new();

        // Initialize weights for 32 layers, for both q_proj and v_proj
        for layer_idx in 0..32 {
            for module in &metadata.target_modules {
                let key = format!("model.layers.{}.self_attn.{}", layer_idx, module);

                // lora_A initialized with small random values to break symmetry
                let mut lora_a = vec![0.0f32; metadata.rank as usize * hidden_dim];
                for val in &mut lora_a {
                    *val = (rand::random::<f32>() - 0.5) / (metadata.rank as f32).sqrt();
                }

                // lora_B initialized to zero to ensure zero identity transform at start
                let lora_b = vec![0.0f32; hidden_dim * metadata.rank as usize];

                lora_weights.insert(key, (lora_a, lora_b));
            }
        }

        Ok(Self {
            metadata,
            adapter_path: std::path::PathBuf::from(path),
            lora_weights,
            hidden_dim,
        })
    }

    /// Update weights for a single backprop step.
    /// x: input activations tensor (size N x hidden_dim)
    /// dy: loss gradient w.r.t LoRA output (size N x hidden_dim)
    pub fn update_layer_weights(
        &mut self,
        key: &str,
        x: &[f32],
        dy: &[f32],
        learning_rate: f32,
        batch_size: usize,
    ) -> Result<(), String> {
        let (lora_a, lora_b) = self
            .lora_weights
            .get_mut(key)
            .ok_or_else(|| format!("Layer weights for {} not found", key))?;

        let r = self.metadata.rank as usize;
        let d = self.hidden_dim;
        let n = batch_size;
        let scale = self.metadata.alpha / (r as f32);

        // Z = X * A^T
        // X: n x d
        // A: r x d -> A^T: d x r
        // Z: n x r
        let mut z = vec![0.0f32; n * r];
        for i in 0..n {
            for j in 0..r {
                let mut sum = 0.0;
                for k in 0..d {
                    sum += x[i * d + k] * lora_a[j * d + k];
                }
                z[i * r + j] = sum;
            }
        }

        // Compute gradients:
        // dL/dB = scale * dy^T * Z
        // dy^T: d x n
        // Z: n x r
        // db: d x r
        let mut db = vec![0.0f32; d * r];
        for i in 0..d {
            for j in 0..r {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += dy[k * d + i] * z[k * r + j];
                }
                db[i * r + j] = scale * sum;
            }
        }

        // dL/dZ = scale * dy * B
        // dy: n x d
        // B: d x r
        // dz: n x r
        let mut dz = vec![0.0f32; n * r];
        for i in 0..n {
            for j in 0..r {
                let mut sum = 0.0;
                for k in 0..d {
                    sum += dy[i * d + k] * lora_b[k * r + j];
                }
                dz[i * r + j] = scale * sum;
            }
        }

        // dL/dA = dz^T * X
        // dz^T: r x n
        // X: n x d
        // da: r x d
        let mut da = vec![0.0f32; r * d];
        for i in 0..r {
            for j in 0..d {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += dz[k * r + i] * x[k * d + j];
                }
                da[i * d + j] = sum;
            }
        }

        // Apply gradient descent step:
        for idx in 0..(r * d) {
            lora_a[idx] -= learning_rate * da[idx];
        }
        for idx in 0..(d * r) {
            lora_b[idx] -= learning_rate * db[idx];
        }

        Ok(())
    }

    /// Perform forward pass, compute MSE loss, run backward pass and update weights.
    pub fn train_on_activations(
        &mut self,
        layer_key: &str,
        inputs: &[Vec<f32>],  // Batch of inputs (each is size hidden_dim)
        targets: &[Vec<f32>], // Batch of targets/outputs (each is size hidden_dim)
        learning_rate: f32,
        epochs: usize,
    ) -> Result<f32, String> {
        if inputs.is_empty() || targets.is_empty() {
            return Err("Input or target batch is empty".to_string());
        }
        if inputs.len() != targets.len() {
            return Err("Input and target size mismatch".to_string());
        }

        let n = inputs.len();
        let d = self.hidden_dim;
        let r = self.metadata.rank as usize;
        let scale = self.metadata.alpha / (r as f32);

        // Flatten inputs
        let mut flat_inputs = vec![0.0f32; n * d];
        for i in 0..n {
            flat_inputs[i * d..(i + 1) * d].copy_from_slice(&inputs[i]);
        }

        let mut avg_loss = 0.0;

        for _epoch in 0..epochs {
            let (lora_a, lora_b) = self
                .lora_weights
                .get(layer_key)
                .ok_or_else(|| format!("Layer {} not found", layer_key))?;

            // Z = X * A^T (n x r)
            let mut z = vec![0.0f32; n * r];
            for i in 0..n {
                for j in 0..r {
                    let mut sum = 0.0;
                    for k in 0..d {
                        sum += flat_inputs[i * d + k] * lora_a[j * d + k];
                    }
                    z[i * r + j] = sum;
                }
            }

            // Y = Z * B^T * scale (n x d)
            let mut y = vec![0.0f32; n * d];
            for i in 0..n {
                for j in 0..d {
                    let mut sum = 0.0;
                    for k in 0..r {
                        sum += z[i * r + k] * lora_b[j * r + k];
                    }
                    y[i * d + j] = sum * scale;
                }
            }

            // Compute loss (Mean Squared Error) and gradient (y - target)
            let mut loss = 0.0;
            let mut dy = vec![0.0f32; n * d];
            for i in 0..n {
                for j in 0..d {
                    let diff = y[i * d + j] - targets[i][j];
                    loss += diff * diff;
                    dy[i * d + j] = diff / (n as f32); // Normalized gradient w.r.t MSE
                }
            }
            avg_loss = loss / (n * d) as f32;

            // Backward pass & SGD update
            self.update_layer_weights(layer_key, &flat_inputs, &dy, learning_rate, n)?;
        }

        Ok(avg_loss)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llama_adapter_specifications() {
        let adapter = LlamaAdapter::default();
        assert_eq!(adapter.name(), "LLaMA-3-8B");
        assert_eq!(adapter.num_layers(), 32);
        assert_eq!(adapter.rope_theta(), 500000.0);
        assert_eq!(adapter.get_early_exit_bounds(), (8, 24));
        assert!(adapter.supported_tasks().contains(&"code"));
        assert_eq!(adapter.native_precision(), "F32");
    }

    #[test]
    fn test_gemma_adapter_specifications() {
        let adapter = GemmaAdapter::default();
        assert_eq!(adapter.name(), "Gemma-2-9B");
        assert_eq!(adapter.num_layers(), 42);
        assert_eq!(adapter.rope_theta(), 10000.0);
        assert_eq!(adapter.get_early_exit_bounds(), (10, 30));
        assert!(adapter.supported_tasks().contains(&"fast-decode"));
        assert_eq!(adapter.native_precision(), "INT8");
    }

    #[test]
    fn test_model_capability_registry() {
        let registry = ModelCapabilityRegistry::new();
        let l_adapter = registry.get_adapter("llama").unwrap();
        assert_eq!(l_adapter.name(), "LLaMA-3-8B");

        let best_for_code = registry.select_best_model_for_task("code").unwrap();
        assert_eq!(best_for_code, "llama");

        let best_for_fast = registry.select_best_model_for_task("fast-decode").unwrap();
        assert_eq!(best_for_fast, "gemma");
    }

    #[test]
    fn test_backend_capability_profile() {
        let cpu_profile = BackendCapabilityProfile::new_cpu_profile();
        let llama = LlamaAdapter::default();
        assert!(cpu_profile.supports_model(&llama));

        let restricted_profile = BackendCapabilityProfile::new_gpu_profile(
            "LowEnd GPU".to_string(),
            500_000_000,
            5_000_000,
            false,
        );
        assert!(!restricted_profile.supports_model(&llama));
    }

    #[test]
    fn test_adapter_manager_load() {
        let manager = AdapterManager::load_from_config("dummy/path").unwrap();
        assert_eq!(manager.metadata.rank, 16);
        assert_eq!(manager.metadata.alpha, 32.0);
        assert_eq!(manager.metadata.target_modules.len(), 2);
    }

    #[test]
    fn test_adapter_learning_pipeline() {
        let mut manager = AdapterManager::load_from_config("dummy/path").unwrap();
        let key = "model.layers.0.self_attn.q_proj";

        // Create dummy inputs
        let inputs = vec![vec![1.0f32; 4096], vec![0.5f32; 4096]];

        // Targets are simulated target outputs of the LoRA projection
        let targets = vec![vec![0.1f32; 4096], vec![0.05f32; 4096]];

        // 1. Initial training should succeed and reduce loss
        let loss_start = manager
            .train_on_activations(key, &inputs, &targets, 0.00001, 1)
            .unwrap();
        let loss_end = manager
            .train_on_activations(key, &inputs, &targets, 0.00001, 10)
            .unwrap();

        assert!(
            loss_end < loss_start,
            "Loss should decrease from start {} to end {}",
            loss_start,
            loss_end
        );
    }
}
