use burn::backend::Wgpu;
use burn::backend::wgpu::WgpuDevice;
use burn::tensor::backend::Backend;
use burn::tensor::{Data, Shape, Tensor, activation::softmax};
use memmap2::Mmap;
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use tokenizers::Tokenizer;

/// Helper function to load named tensor from safetensors into Burn
fn get_bert_tensor<B: Backend, const D: usize>(
    st: &safetensors::SafeTensors,
    name: &str,
    device: &<B as Backend>::Device,
) -> Result<Tensor<B, D>, String> {
    let view = st
        .tensor(name)
        .map_err(|e| format!("Tensor '{}' not found in BERT safetensors: {:?}", name, e))?;
    let float_data: &[f32] = bytemuck::cast_slice(view.data());

    let mut shape_arr = [0; D];
    for (i, &dim) in view.shape().iter().enumerate().take(D) {
        shape_arr[i] = dim;
    }

    let data = Data::new(float_data.to_vec(), Shape::from(shape_arr)).convert();
    Ok(Tensor::<B, D>::from_data(data, device))
}

/// Helper for standard LayerNorm in Burn
fn layer_norm<B: Backend, const D: usize>(
    x: Tensor<B, D>,
    weight: Tensor<B, 1>,
    bias: Tensor<B, 1>,
    eps: f32,
) -> Tensor<B, D> {
    let mean = x.clone().mean_dim(D - 1);
    let variance = x.clone().sub(mean.clone()).powf_scalar(2.0).mean_dim(D - 1);
    let norm = x.sub(mean).div(variance.add_scalar(eps).sqrt());
    norm.mul(weight.unsqueeze_dim(0)).add(bias.unsqueeze_dim(0))
}

struct RerankCache {
    map: HashMap<String, f32>,
    order: VecDeque<String>,
    max_size: usize,
}

impl RerankCache {
    fn new(max_size: usize) -> Self {
        Self {
            map: HashMap::new(),
            order: VecDeque::new(),
            max_size,
        }
    }

    fn get(&self, key: &str) -> Option<f32> {
        self.map.get(key).copied()
    }

    fn insert(&mut self, key: String, score: f32) {
        if !self.map.contains_key(&key) {
            if self.map.len() >= self.max_size
                && let Some(old_key) = self.order.pop_front()
            {
                self.map.remove(&old_key);
            }
            self.map.insert(key.clone(), score);
            self.order.push_back(key);
        }
    }
}

static NATIVE_RERANKER_CPU: OnceLock<Reranker> = OnceLock::new();
static NATIVE_RERANKER_GPU: OnceLock<Reranker> = OnceLock::new();

/// Thread-safe, WGPU-accelerated Cross-Encoder Reranker
pub struct Reranker {
    tokenizer: Tokenizer,
    mmap: Mmap,
    device: WgpuDevice,
    cache: Mutex<RerankCache>,
}

impl Reranker {
    /// Thread-safe accessor for the globally initialized native reranker
    pub async fn get_global() -> Result<&'static Self, String> {
        let is_cpu = crate::inference::is_cpu_only();
        let lock = if is_cpu {
            &NATIVE_RERANKER_CPU
        } else {
            &NATIVE_RERANKER_GPU
        };

        if let Some(reranker) = lock.get() {
            return Ok(reranker);
        }

        // WgpuDevice::Cpu panics — no adapter reports DeviceType::Cpu on this system.
        // BestAvailable falls through to lavapipe/software renderer when no GPU is present.
        let device = if is_cpu {
            WgpuDevice::BestAvailable
        } else {
            WgpuDevice::default()
        };

        let reranker = Self::new(device).await?;
        let _ = lock.set(reranker);
        Ok(lock.get().unwrap())
    }

    /// Creates and downloads (if missing) the ms-marco-MiniLM-L-6-v2 Cross-Encoder model
    async fn new(device: WgpuDevice) -> Result<Self, String> {
        let model_dir = PathBuf::from("models/ms-marco-MiniLM-L-6-v2");
        std::fs::create_dir_all(&model_dir).map_err(|e| e.to_string())?;

        let config_path = model_dir.join("config.json");
        let tokenizer_path = model_dir.join("tokenizer.json");
        let model_path = model_dir.join("model.safetensors");

        let client = reqwest::Client::new();
        for (filename, path) in &[
            ("config.json", &config_path),
            ("tokenizer.json", &tokenizer_path),
            ("model.safetensors", &model_path),
        ] {
            if !path.exists() {
                println!("📥 Downloading native Cross-Encoder asset: {}...", filename);
                let url = format!(
                    "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/main/{}",
                    filename
                );
                let bytes = client
                    .get(&url)
                    .send()
                    .await
                    .map_err(|e| format!("Failed to download {}: {}", filename, e))?
                    .bytes()
                    .await
                    .map_err(|e| format!("Failed to parse bytes for {}: {}", filename, e))?;
                std::fs::write(path, bytes).map_err(|e| e.to_string())?;
                println!("✅ Downloaded: {}", filename);
            }
        }

        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| e.to_string())?;

        // Load weights via safe memory-mapping
        let file = std::fs::File::open(&model_path).map_err(|e| e.to_string())?;
        // SAFETY: Manual invariants verified for performance/FFI
        let mmap = unsafe { Mmap::map(&file).map_err(|e| e.to_string())? };

        println!(
            "⚡ Native WGPU Cross-Encoder Reranker initialized successfully on device: {:?}!",
            device
        );
        Ok(Reranker {
            tokenizer,
            mmap,
            device,
            cache: Mutex::new(RerankCache::new(2000)),
        })
    }

    /// Computes high-fidelity relevance score for a given query and document pair
    pub fn compute_score(&self, query: &str, document: &str) -> Result<f32, String> {
        let cache_key = format!("{}|||{}", query, document);
        if let Ok(cache) = self.cache.lock()
            && let Some(score) = cache.get(&cache_key)
        {
            return Ok(score);
        }

        type B = Wgpu;
        let device = &self.device;

        // Tokenize [CLS] Query [SEP] Document [SEP] automatically as a pair
        let tokens = self
            .tokenizer
            .encode((query.to_string(), document.to_string()), true)
            .map_err(|e| e.to_string())?;

        let token_ids = tokens.get_ids();
        let type_ids = tokens.get_type_ids();
        let seq_len = token_ids.len();

        if seq_len == 0 {
            return Ok(0.0f32);
        }

        let st = safetensors::SafeTensors::deserialize(&self.mmap).map_err(|e| e.to_string())?;

        // 1. Embeddings Lookups
        let word_emb_w = get_bert_tensor::<B, 2>(&st, "embeddings.word_embeddings.weight", device)?;
        let pos_emb_w =
            get_bert_tensor::<B, 2>(&st, "embeddings.position_embeddings.weight", device)?;
        let type_emb_w =
            get_bert_tensor::<B, 2>(&st, "embeddings.token_type_embeddings.weight", device)?;

        let tokens_data = Data::new(
            token_ids.iter().map(|&t| t as i32).collect::<Vec<i32>>(),
            Shape::from([seq_len]),
        )
        .convert();
        let tokens_t = Tensor::<B, 1, burn::tensor::Int>::from_data(tokens_data, device);
        let mut x = word_emb_w.select(0, tokens_t);

        // Sum Position Embeddings
        let positions: Vec<i32> = (0..seq_len).map(|i| i as i32).collect();
        let pos_data = Data::new(positions, Shape::from([seq_len])).convert();
        let pos_t = Tensor::<B, 1, burn::tensor::Int>::from_data(pos_data, device);
        x = x.add(pos_emb_w.select(0, pos_t));

        // Sum Segment Token Type Embeddings
        let type_data = Data::new(
            type_ids.iter().map(|&t| t as i32).collect::<Vec<i32>>(),
            Shape::from([seq_len]),
        )
        .convert();
        let type_t = Tensor::<B, 1, burn::tensor::Int>::from_data(type_data, device);
        x = x.add(type_emb_w.select(0, type_t));

        // Embedding LayerNorm
        let emb_ln_w = get_bert_tensor::<B, 1>(&st, "embeddings.LayerNorm.weight", device)?;
        let emb_ln_b = get_bert_tensor::<B, 1>(&st, "embeddings.LayerNorm.bias", device)?;
        x = layer_norm(x, emb_ln_w, emb_ln_b, 1e-12);

        // 2. Transformer layers (6 Layers for MiniLM)
        let num_layers = 6;
        let num_heads = 12;
        let head_dim = 32;
        let hidden_size = 384;

        for layer_idx in 0..num_layers {
            let prefix = format!("bert.encoder.layer.{}", layer_idx);

            // Self-Attention QKV Projections
            let q_w = get_bert_tensor::<B, 2>(
                &st,
                &format!("{}.attention.self.query.weight", prefix),
                device,
            )?;
            let q_b = get_bert_tensor::<B, 1>(
                &st,
                &format!("{}.attention.self.query.bias", prefix),
                device,
            )?;
            let k_w = get_bert_tensor::<B, 2>(
                &st,
                &format!("{}.attention.self.key.weight", prefix),
                device,
            )?;
            let k_b = get_bert_tensor::<B, 1>(
                &st,
                &format!("{}.attention.self.key.bias", prefix),
                device,
            )?;
            let v_w = get_bert_tensor::<B, 2>(
                &st,
                &format!("{}.attention.self.value.weight", prefix),
                device,
            )?;
            let v_b = get_bert_tensor::<B, 1>(
                &st,
                &format!("{}.attention.self.value.bias", prefix),
                device,
            )?;

            let q = x.clone().matmul(q_w.transpose()).add(q_b.unsqueeze_dim(0));
            let k = x.clone().matmul(k_w.transpose()).add(k_b.unsqueeze_dim(0));
            let v = x.clone().matmul(v_w.transpose()).add(v_b.unsqueeze_dim(0));

            // Reshape and Transpose for Multi-Head Attention
            let q = q
                .reshape(Shape::from([seq_len, num_heads, head_dim]))
                .swap_dims(0, 1);
            let k = k
                .reshape(Shape::from([seq_len, num_heads, head_dim]))
                .swap_dims(0, 1);
            let v = v
                .reshape(Shape::from([seq_len, num_heads, head_dim]))
                .swap_dims(0, 1);

            // Attention Score Calculation
            let scale = 1.0 / (head_dim as f32).sqrt();
            let scores = q.matmul(k.transpose()).mul_scalar(scale);
            let probs = softmax(scores, 2);
            let context = probs.matmul(v);

            // Reshape Context Back
            let context = context
                .swap_dims(0, 1)
                .reshape(Shape::from([seq_len, hidden_size]));

            // Attention Output Projection
            let o_w = get_bert_tensor::<B, 2>(
                &st,
                &format!("{}.attention.output.dense.weight", prefix),
                device,
            )?;
            let o_b = get_bert_tensor::<B, 1>(
                &st,
                &format!("{}.attention.output.dense.bias", prefix),
                device,
            )?;
            let attn_out = context.matmul(o_w.transpose()).add(o_b.unsqueeze_dim(0));

            // Residual + LayerNorm
            let attn_ln_w = get_bert_tensor::<B, 1>(
                &st,
                &format!("{}.attention.output.LayerNorm.weight", prefix),
                device,
            )?;
            let attn_ln_b = get_bert_tensor::<B, 1>(
                &st,
                &format!("{}.attention.output.LayerNorm.bias", prefix),
                device,
            )?;
            x = layer_norm(x.add(attn_out), attn_ln_w, attn_ln_b, 1e-12);

            // Feed Forward MLP
            let inter_w = get_bert_tensor::<B, 2>(
                &st,
                &format!("{}.intermediate.dense.weight", prefix),
                device,
            )?;
            let inter_b = get_bert_tensor::<B, 1>(
                &st,
                &format!("{}.intermediate.dense.bias", prefix),
                device,
            )?;
            let out_w =
                get_bert_tensor::<B, 2>(&st, &format!("{}.output.dense.weight", prefix), device)?;
            let out_b =
                get_bert_tensor::<B, 1>(&st, &format!("{}.output.dense.bias", prefix), device)?;

            let h_inter = x
                .clone()
                .matmul(inter_w.transpose())
                .add(inter_b.unsqueeze_dim(0));
            // GELU activation (approximate via tanh)
            let gelu_h = h_inter.clone().mul(
                h_inter
                    .mul_scalar(0.044715)
                    .powf_scalar(3.0)
                    .add_scalar(1.0)
                    .mul_scalar(0.797884)
                    .tanh()
                    .add_scalar(1.0)
                    .mul_scalar(0.5),
            );
            let ffn_out = gelu_h.matmul(out_w.transpose()).add(out_b.unsqueeze_dim(0));

            // Residual + LayerNorm
            let ffn_ln_w = get_bert_tensor::<B, 1>(
                &st,
                &format!("{}.output.LayerNorm.weight", prefix),
                device,
            )?;
            let ffn_ln_b =
                get_bert_tensor::<B, 1>(&st, &format!("{}.output.LayerNorm.bias", prefix), device)?;
            x = layer_norm(x.add(ffn_out), ffn_ln_w, ffn_ln_b, 1e-12);
        }

        // Extract [CLS] representation (index 0 of seq_len)
        let cls_rep = x
            .slice([0..1, 0..hidden_size])
            .reshape(Shape::from([1, hidden_size]));

        // Classification Head
        let classifier_w = get_bert_tensor::<B, 2>(&st, "classifier.weight", device)?;
        let classifier_b = get_bert_tensor::<B, 1>(&st, "classifier.bias", device)?;

        let logits = cls_rep
            .matmul(classifier_w.transpose())
            .add(classifier_b.unsqueeze_dim(0));
        let logit_val = logits.into_data().value[0];

        // Apply Sigmoid to produce probabilistic relevance score
        let sigmoid_score = 1.0 / (1.0 + (-logit_val).exp());

        if let Ok(mut cache) = self.cache.lock() {
            cache.insert(cache_key, sigmoid_score);
        }

        Ok(sigmoid_score)
    }
}
