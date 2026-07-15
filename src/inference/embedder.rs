use burn::backend::Wgpu;
use burn::backend::wgpu::WgpuDevice;
use burn::tensor::backend::Backend;
use burn::tensor::{Data, Shape, Tensor, activation::softmax};
use memmap2::Mmap;
use std::path::PathBuf;
use std::sync::OnceLock;
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

/// High-performance in-process sentence-transformer embedder using Burn WGPU
pub struct Embedder {
    tokenizer: Tokenizer,
    mmap: Mmap,
    device: WgpuDevice,
}

static NATIVE_EMBEDDER_CPU: OnceLock<Embedder> = OnceLock::new();
static NATIVE_EMBEDDER_GPU: OnceLock<Embedder> = OnceLock::new();

impl Embedder {
    /// Lazy thread-safe getter for the global native embedder
    pub async fn get_global() -> Result<&'static Self, String> {
        let is_cpu = crate::inference::is_cpu_only();
        let lock = if is_cpu {
            &NATIVE_EMBEDDER_CPU
        } else {
            &NATIVE_EMBEDDER_GPU
        };

        if let Some(embedder) = lock.get() {
            return Ok(embedder);
        }

        // WgpuDevice::Cpu panics — no adapter reports DeviceType::Cpu on this system.
        // BestAvailable falls through to lavapipe/software renderer when no GPU is present.
        let device = if is_cpu {
            WgpuDevice::BestAvailable
        } else {
            WgpuDevice::default()
        };

        let embedder = Self::new(device).await?;
        let _ = lock.set(embedder);
        Ok(lock.get().unwrap())
    }

    /// Creates and downloads (if missing) the all-MiniLM-L6-v2 sentence-transformer
    async fn new(device: WgpuDevice) -> Result<Self, String> {
        let model_dir = PathBuf::from("models/all-MiniLM-L6-v2");
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
                println!("📥 Downloading {} for native Rust embedder...", filename);
                let url = format!(
                    "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/{}",
                    filename
                );

                let res = client
                    .get(&url)
                    .send()
                    .await
                    .map_err(|e| format!("Failed to download {}: {}", filename, e))?;

                if !res.status().is_success() {
                    return Err(format!("Failed download HTTP status: {}", res.status()));
                }

                let bytes = res.bytes().await.map_err(|e| e.to_string())?;
                std::fs::write(path, bytes).map_err(|e| e.to_string())?;
            }
        }

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| e.to_string())?;

        // Load weights via safe memory-mapping
        let file = std::fs::File::open(&model_path).map_err(|e| e.to_string())?;
        let mmap = unsafe { Mmap::map(&file).map_err(|e| e.to_string())? };

        println!(
            "⚡ Native WGPU Bramha Embedder initialized successfully on device: {:?}!",
            device
        );
        Ok(Embedder {
            tokenizer,
            mmap,
            device,
        })
    }

    /// Computes normalized 384-dimensional text embeddings in microseconds
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, String> {
        type B = Wgpu;
        let device = &self.device;

        let tokens = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| e.to_string())?;
        let token_ids = tokens.get_ids();
        let seq_len = token_ids.len();

        if seq_len == 0 {
            return Ok(vec![0.0f32; 384]);
        }

        // Open safetensors
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

        // Word + Position + Token type sums
        let mut x = word_emb_w.select(0, tokens_t);

        let positions: Vec<i32> = (0..seq_len).map(|i| i as i32).collect();
        let pos_data = Data::new(positions, Shape::from([seq_len])).convert();
        let pos_t = Tensor::<B, 1, burn::tensor::Int>::from_data(pos_data, device);
        let pos_embeddings = pos_emb_w.select(0, pos_t);
        x = x.add(pos_embeddings);

        let types = vec![0i32; seq_len];
        let type_data = Data::new(types, Shape::from([seq_len])).convert();
        let type_t = Tensor::<B, 1, burn::tensor::Int>::from_data(type_data, device);
        let type_embeddings = type_emb_w.select(0, type_t);
        x = x.add(type_embeddings);

        // Embedding LayerNorm
        let emb_ln_w = get_bert_tensor::<B, 1>(&st, "embeddings.LayerNorm.weight", device)?;
        let emb_ln_b = get_bert_tensor::<B, 1>(&st, "embeddings.LayerNorm.bias", device)?;
        x = layer_norm(x, emb_ln_w, emb_ln_b, 1e-12);

        // 2. Transformer layers (6 Layers for all-MiniLM-L6-v2)
        let num_layers = 6;
        let num_heads = 12;
        let head_dim = 32;
        let hidden_size = 384;

        for layer_idx in 0..num_layers {
            let prefix = format!("encoder.layer.{}", layer_idx);

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

            // Scores
            let scale = 1.0 / (head_dim as f32).sqrt();
            let scores = q.matmul(k.transpose()).mul_scalar(scale);
            let probs = softmax(scores, 2);
            let context = probs.matmul(v);

            // Reshape context back
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

        // 3. Mean Pooling: Average token embeddings across sequence length
        let mean_embedding = x.mean_dim(0).reshape(Shape::from([hidden_size]));

        // L2 Normalization
        let sum_squares = mean_embedding.clone().powf_scalar(2.0).sum();
        let norm = sum_squares.sqrt();
        let normalized = mean_embedding.div(norm);

        // Download result back to CPU
        let embedding_data = normalized.into_data();
        Ok(embedding_data.value)
    }
}
