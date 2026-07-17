use crate::inference::tokenizer::BramhaTokenizer;
use crate::storage::Database;
use burn::backend::Wgpu;
use burn::backend::wgpu::WgpuDevice;
use burn::tensor::backend::Backend;
use burn::tensor::{Data, Shape, Tensor, activation::softmax};
use std::io::Write;
use std::sync::Arc;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
pub struct LogEntry {
    pub message: String,
    pub time: u64,
}

pub struct InferenceLogger {
    logs: Mutex<Vec<LogEntry>>,
}

pub struct VramCache {
    pub layers_1d: std::collections::HashMap<String, Tensor<Wgpu, 1>>,
    pub layers_2d: std::collections::HashMap<String, Tensor<Wgpu, 2>>,
    pub max_cached_tensors: usize,
    pub max_vram_bytes: Option<usize>,
    pub current_vram_bytes: usize,
    pub access_order: Vec<String>,
    pub tensor_sizes: std::collections::HashMap<String, usize>,
    pub suppress_eviction_logs: bool,
}

impl VramCache {
    pub fn global() -> &'static Mutex<Self> {
        static INSTANCE: OnceLock<Mutex<VramCache>> = OnceLock::new();
        INSTANCE.get_or_init(|| {
            Mutex::new(VramCache {
                layers_1d: std::collections::HashMap::new(),
                layers_2d: std::collections::HashMap::new(),
                max_cached_tensors: 300,
                max_vram_bytes: Some(600_000_000), // Enforce strict 600 MB VRAM limit to leave headroom for wgpu allocator pools
                suppress_eviction_logs: true, // Avoid noisy per-tensor log spam during steady-state inference
                current_vram_bytes: 0,
                access_order: Vec::new(),
                tensor_sizes: std::collections::HashMap::new(),
            })
        })
    }

    pub fn set_limit(&mut self, limit_percentage: f32) {
        // Assume standard 4.0 GB physical VRAM for the NVIDIA T500 card to calculate absolute ceiling
        let total_gpu_bytes = 4_000_000_000usize;
        let cap_bytes = (total_gpu_bytes as f32 * limit_percentage) as usize;
        self.max_vram_bytes = Some(cap_bytes);

        let logger = InferenceLogger::global();
        logger.record_log(format!(
            "⚙️ VRAM Cache cap configured: {:.1}% ({:.2} GB limit)",
            limit_percentage * 100.0,
            cap_bytes as f64 / 1_000_000_000.0
        ));

        // Enforce limits immediately
        self.enforce_limits(0);
    }

    /// Evicts LRU tensors if memory pressure is exceeded, leaving room for `new_tensor_bytes`
    pub fn enforce_limits(&mut self, new_tensor_bytes: usize) {
        let max_bytes = match self.max_vram_bytes {
            Some(m) => m,
            None => return, // Unlimited
        };

        while self.current_vram_bytes + new_tensor_bytes > max_bytes
            && !self.access_order.is_empty()
        {
            let mut evicted = false;
            for i in 0..self.access_order.len() {
                let name = &self.access_order[i];
                if self.layers_1d.contains_key(name) || self.layers_2d.contains_key(name) {
                    let name = self.access_order.remove(i);
                    let size = self.tensor_sizes.remove(&name).unwrap_or(0);
                    self.layers_1d.remove(&name);
                    self.layers_2d.remove(&name);
                    self.current_vram_bytes = self.current_vram_bytes.saturating_sub(size);

                    if !self.suppress_eviction_logs {
                        let logger = InferenceLogger::global();
                        logger.record_log(format!(
                            "🗑️ VRAM Eviction: Freed tensor '{}' ({:.2} MB) to stay under cap.",
                            name,
                            size as f64 / 1_000_000.0
                        ));
                    }
                    evicted = true;
                    break;
                }
            }
            if !evicted {
                self.access_order.clear();
                break;
            }
        }
    }

    pub fn record_access(&mut self, name: &str) {
        if let Some(pos) = self.access_order.iter().position(|x| x == name) {
            self.access_order.remove(pos);
        }
        self.access_order.push(name.to_string());
    }
}

impl InferenceLogger {
    pub fn global() -> &'static Self {
        static INSTANCE: OnceLock<InferenceLogger> = OnceLock::new();
        INSTANCE.get_or_init(|| InferenceLogger {
            logs: Mutex::new(Vec::new()),
        })
    }

    pub fn record_log(&self, message: String) {
        if let Ok(mut logs) = self.logs.lock() {
            let time = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;
            println!("{}", message);
            logs.push(LogEntry { message, time });
            if logs.len() > 2000 {
                logs.remove(0);
            }
        }
    }

    pub fn get_logs(&self, since: u64) -> Vec<LogEntry> {
        if let Ok(logs) = self.logs.lock() {
            logs.iter()
                .filter(|entry| entry.time > since)
                .cloned()
                .collect()
        } else {
            Vec::new()
        }
    }
}

pub(crate) fn estimate_query_complexity(prompt: &str) -> f32 {
    let word_count = prompt.split_whitespace().count();
    let has_technical_terms = prompt.contains("code")
        || prompt.contains("rust")
        || prompt.contains("compile")
        || prompt.contains("error")
        || prompt.contains("implement")
        || prompt.contains("bramha")
        || prompt.contains("database");

    let mut score: f32 = 0.5;
    if word_count > 15 {
        score += 0.2;
    }
    if has_technical_terms {
        score += 0.2;
    }
    score.clamp(0.1, 1.0)
}

/// Helper function to perform RMSNorm in Burn
pub(crate) fn rms_norm<B: Backend, const D: usize>(
    x: Tensor<B, D>,
    weight: Tensor<B, 1>,
    eps: f32,
) -> Tensor<B, D> {
    let variance = x.clone().powf_scalar(2.0).mean_dim(D - 1);
    let norm = x.div(variance.add_scalar(eps).sqrt());

    // Broadcast multiply by 1D weight along the last dimension
    norm.mul(weight.unsqueeze_dim(0))
}

/// Helper function to precompute RoPE frequency cosine and sine in Burn
pub(crate) fn precompute_rope_freqs<B: Backend>(
    seq_len: usize,
    head_dim: usize,
    theta: f32,
    device: &<B as Backend>::Device,
) -> (Tensor<B, 3>, Tensor<B, 3>) {
    let half_dim = head_dim / 2;
    let mut freqs = Vec::with_capacity(half_dim);
    for i in 0..half_dim {
        freqs.push(1.0 / theta.powf((2 * i) as f32 / head_dim as f32));
    }

    let t: Vec<f32> = (0..seq_len).map(|i| i as f32).collect();

    let freqs_data = Data::new(freqs, Shape::from([half_dim])).convert();
    let freqs_t = Tensor::<B, 1>::from_data(freqs_data, device);

    let t_data = Data::new(t, Shape::from([seq_len])).convert();
    let t_t = Tensor::<B, 1>::from_data(t_data, device);

    // Outer product: [seq_len, half_dim]
    let freqs_outer = t_t
        .unsqueeze_dim::<2>(1)
        .matmul(freqs_t.unsqueeze_dim::<2>(0));

    // Cat along dim 1 to get [seq_len, head_dim]
    let freqs_cat = Tensor::cat(vec![freqs_outer.clone(), freqs_outer], 1);

    let cos = freqs_cat.clone().cos().unsqueeze_dim::<3>(1);
    let sin = freqs_cat.sin().unsqueeze_dim::<3>(1);

    (cos, sin)
}

/// Helper function to apply RoPE positional rotation in Burn
pub(crate) fn apply_rope<B: Backend>(
    x: Tensor<B, 3>, // [seq_len, num_heads, head_dim]
    cos: Tensor<B, 3>,
    sin: Tensor<B, 3>,
) -> Tensor<B, 3> {
    let head_dim = x.shape().dims[2];
    let half_dim = head_dim / 2;

    let x1 = x
        .clone()
        .slice([0..x.shape().dims[0], 0..x.shape().dims[1], 0..half_dim]);
    let x2 = x.clone().slice([
        0..x.shape().dims[0],
        0..x.shape().dims[1],
        half_dim..head_dim,
    ]);

    let rotated_x = Tensor::cat(vec![x2.neg(), x1], 2);

    x.mul(cos).add(rotated_x.mul(sin))
}

/// GQA head repeating helper in Burn
pub(crate) fn repeat_kv<B: Backend>(x: Tensor<B, 3>, num_repeats: usize) -> Tensor<B, 3> {
    if num_repeats == 1 {
        return x;
    }
    let seq_len = x.shape().dims[0];
    let num_kv_heads = x.shape().dims[1];
    let head_dim = x.shape().dims[2];

    // Unsqueeze to [seq_len, num_kv_heads, 1, head_dim]
    let x = x.unsqueeze_dim::<4>(2);
    // Repeat by expanding (or replicating)
    let mut repeated = Vec::with_capacity(num_repeats);
    for _ in 0..num_repeats {
        repeated.push(x.clone());
    }
    let x = Tensor::cat(repeated, 2);
    x.reshape(Shape::from([seq_len, num_kv_heads * num_repeats, head_dim]))
}

/// Causal mask helper in Burn
pub(crate) fn causal_mask<B: Backend>(
    seq_len: usize,
    device: &<B as Backend>::Device,
) -> Tensor<B, 2> {
    let mut mask_data = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            if j > i {
                mask_data[i * seq_len + j] = -1e9; // Large negative number for stable WebGPU compute compilation
            }
        }
    }
    let data = Data::new(mask_data, Shape::from([seq_len, seq_len])).convert();
    Tensor::<B, 2>::from_data(data, device)
}

/// Causal mask helper for KV Cache
pub(crate) fn causal_mask_kv<B: Backend>(
    num_new_tokens: usize,
    total_seq_len: usize,
    start_pos: usize,
    device: &<B as Backend>::Device,
) -> Tensor<B, 2> {
    let mut mask_data = vec![0.0f32; num_new_tokens * total_seq_len];
    for i in 0..num_new_tokens {
        for t in 0..total_seq_len {
            if t > start_pos + i {
                mask_data[i * total_seq_len + t] = -1e9;
            }
        }
    }
    let data = Data::new(mask_data, Shape::from([num_new_tokens, total_seq_len])).convert();
    Tensor::<B, 2>::from_data(data, device)
}

/// Helper to perform divide-and-conquer split matrix multiplication on WGPU CPU backend
fn split_matmul<B: Backend>(
    x: Tensor<B, 2>,
    w_t: Tensor<B, 2>,
    _device: &<B as Backend>::Device,
) -> Tensor<B, 2>
where
    <B as Backend>::FloatTensorPrimitive<2>: Send + Sync,
{
    let w_shape = w_t.shape();
    let k = w_shape.dims[0];
    let n = w_shape.dims[1];
    let size_bytes = k * n * 4;

    let max_binding_bytes = 100_000_000;

    if size_bytes <= max_binding_bytes || !crate::inference::is_cpu_only() {
        x.matmul(w_t)
    } else {
        let chunk_size = (max_binding_bytes / (k * 4)).max(1);
        let mut ranges = Vec::new();
        let mut start = 0;
        while start < n {
            let end = (start + chunk_size).min(n);
            ranges.push(start..end);
            start = end;
        }

        use rayon::prelude::*;
        let chunks: Vec<Tensor<B, 2>> = ranges
            .into_par_iter()
            .map(|range| {
                let slice = w_t.clone().slice([0..k, range]);
                x.clone().matmul(slice)
            })
            .collect();

        Tensor::cat(chunks, 1)
    }
}

fn safe_cast_to_f32(bytes: &[u8]) -> std::borrow::Cow<'_, [f32]> {
    if (bytes.as_ptr() as usize).is_multiple_of(std::mem::align_of::<f32>()) {
        std::borrow::Cow::Borrowed(bytemuck::cast_slice(bytes))
    } else {
        let mut vec = vec![0.0f32; bytes.len() / 4];
        let bytes_mut = bytemuck::cast_slice_mut::<f32, u8>(&mut vec);
        bytes_mut.copy_from_slice(bytes);
        std::borrow::Cow::Owned(vec)
    }
}

/// Helper to fetch and cast weight tensors from ModelTable in-process in Burn
async fn get_tensor_1d(
    model_name: &str,
    db: &std::sync::Arc<crate::storage::Database>,
    name: &str,
    device: &WgpuDevice,
) -> Result<Tensor<Wgpu, 1>, String> {
    let cache_key = format!("{}:{}", model_name, name);
    {
        let mut cache = VramCache::global().lock().unwrap();
        if cache.layers_1d.contains_key(&cache_key) {
            cache.record_access(&cache_key);
            let cached = cache.layers_1d.get(&cache_key).unwrap().clone();
            return Ok(cached);
        }
    }

    let db_read = db.tensor_db.read().await;
    let model = db_read.models.get(model_name).unwrap();
    let page = model
        .layers
        .get(name)
        .ok_or_else(|| format!("Weight not found in sharded DB: {}", name))?
        .clone();
    drop(db_read);

    // AOT Alignment: We now strictly store data on disk as native F32.
    // The disk page bytes are exactly aligned float memory.
    let f32_data = safe_cast_to_f32(page.as_bytes());

    let mut shape_arr = [0; 1];
    shape_arr[0] = page.shape[0];

    let data = Data::new(f32_data.to_vec(), Shape::from(shape_arr)).convert();
    let tensor = Tensor::<Wgpu, 1>::from_data(data, device);

    {
        let mut cache = VramCache::global().lock().unwrap();
        let size_bytes = page.shape[0] * 4;
        cache.enforce_limits(size_bytes);
        if cache.layers_1d.len() + cache.layers_2d.len() < cache.max_cached_tensors {
            cache.layers_1d.insert(cache_key.clone(), tensor.clone());
            cache.tensor_sizes.insert(cache_key.clone(), size_bytes);
            cache.current_vram_bytes += size_bytes;
            cache.record_access(&cache_key);
        }
    }

    Ok(tensor)
}

async fn get_tensor_2d(
    model_name: &str,
    db: &std::sync::Arc<crate::storage::Database>,
    name: &str,
    device: &WgpuDevice,
) -> Result<Tensor<Wgpu, 2>, String> {
    let cache_key = format!("{}:{}", model_name, name);
    {
        let mut cache = VramCache::global().lock().unwrap();
        if cache.layers_2d.contains_key(&cache_key) {
            cache.record_access(&cache_key);
            let cached = cache.layers_2d.get(&cache_key).unwrap().clone();
            return Ok(cached);
        }
    }

    let db_read = db.tensor_db.read().await;
    let model = db_read.models.get(model_name).unwrap();
    let page = model
        .layers
        .get(name)
        .or_else(|| {
            if name == "lm_head.weight" {
                model.layers.get("model.embed_tokens.weight")
            } else {
                None
            }
        })
        .ok_or_else(|| format!("Weight not found in sharded DB: {}", name))?
        .clone();

    let scale_page_opt = model
        .layers
        .get(&format!("{}.scale", name))
        .or_else(|| {
            if name == "lm_head.weight" {
                model.layers.get("model.embed_tokens.weight.scale")
            } else {
                None
            }
        })
        .cloned();
    drop(db_read);

    // Dequantize on the fly if stored as I8 or U4, otherwise load raw F32
    let f32_data = match page.dtype {
        crate::core::tensor::DType::I8 => {
            let scale_page = scale_page_opt
                .ok_or_else(|| format!("Scale not found for quantized weight: {}", name))?;
            let scales = safe_cast_to_f32(scale_page.as_bytes());
            let q_weight: &[i8] = bytemuck::cast_slice(page.as_bytes());
            crate::models::quantization::dequantize_int8(q_weight, &scales, page.shape[1])
        }
        crate::core::tensor::DType::U4 => {
            let scale_page = scale_page_opt
                .ok_or_else(|| format!("Scale not found for quantized weight: {}", name))?;
            let scales = safe_cast_to_f32(scale_page.as_bytes());
            crate::models::quantization::dequantize_int4(page.as_bytes(), &scales, page.shape[1])
        }
        _ => safe_cast_to_f32(page.as_bytes()).into_owned(),
    };

    let mut shape_arr = [0; 2];
    shape_arr[0] = page.shape[0];
    shape_arr[1] = page.shape[1];

    let data = Data::new(f32_data, Shape::from(shape_arr)).convert();
    let tensor = Tensor::<Wgpu, 2>::from_data(data, device);

    {
        let mut cache = VramCache::global().lock().unwrap();
        let size_bytes = page.shape[0] * page.shape[1] * 4;
        cache.enforce_limits(size_bytes);
        if cache.layers_1d.len() + cache.layers_2d.len() < cache.max_cached_tensors {
            cache.layers_2d.insert(cache_key.clone(), tensor.clone());
            cache.tensor_sizes.insert(cache_key.clone(), size_bytes);
            cache.current_vram_bytes += size_bytes;
            cache.record_access(&cache_key);
        }
    }

    Ok(tensor)
}

async fn get_transposed_tensor_2d(
    model_name: &str,
    db: &std::sync::Arc<crate::storage::Database>,
    name: &str,
    device: &WgpuDevice,
) -> Result<Tensor<Wgpu, 2>, String> {
    let cache_key = format!("{}:{}:transposed", model_name, name);
    {
        let mut cache = VramCache::global().lock().unwrap();
        if cache.layers_2d.contains_key(&cache_key) {
            cache.record_access(&cache_key);
            let cached = cache.layers_2d.get(&cache_key).unwrap().clone();
            return Ok(cached);
        }
    }

    let base_tensor = get_tensor_2d(model_name, db, name, device).await?;
    let transposed = base_tensor.transpose();

    {
        let mut cache = VramCache::global().lock().unwrap();
        let size_bytes = transposed.shape().dims[0] * transposed.shape().dims[1] * 4;
        cache.enforce_limits(size_bytes);
        if cache.layers_1d.len() + cache.layers_2d.len() < cache.max_cached_tensors {
            cache
                .layers_2d
                .insert(cache_key.clone(), transposed.clone());
            cache.tensor_sizes.insert(cache_key.clone(), size_bytes);
            cache.current_vram_bytes += size_bytes;
            cache.record_access(&cache_key);
        }
    }

    Ok(transposed)
}

/// Response payload from inference generation
#[derive(serde::Serialize, Clone, Debug)]
pub struct InferenceResult {
    pub model: String,
    pub completion: String,
    pub elapsed_seconds: f64,
    pub tokens_generated: usize,
    pub tokens_per_second: f64,
    pub average_exit_layer: f32,
    pub average_uncertainty_score: f32,
}

/// Native Rust LLaMA inference orchestrator
pub struct InferenceEngine {
    pub adapter_manager: Option<crate::cognitive::adapter::AdapterManager>,
}

impl InferenceEngine {
    pub fn new(adapter_path: Option<&str>) -> Self {
        let adapter_manager = adapter_path
            .and_then(|p| crate::cognitive::adapter::AdapterManager::load_from_config(p).ok());
        Self { adapter_manager }
    }

    /// Generates tokens completely in-process, utilizing the HeterogeneousScheduler
    /// to route to CPU or GPU, and providing seamless fallback to CPU on GPU failure.
    pub async fn generate(
        &self,
        db: Arc<Database>,
        model_name: &str,
        prompt: &str,
        max_new_tokens: usize,
        temperature: f64,
        workflow_id: Option<String>,
        branch_id: Option<String>,
    ) -> Result<InferenceResult, String> {
        let start_time = std::time::Instant::now();

        // 1. Load active policy and deterministic answer cache
        let mut policy = crate::planner::policy::PlannerPolicy::load();
        if let Ok(env_mode) = std::env::var("BRAMHA_PLANNER_MODE") {
            policy.planner_mode = env_mode;
        }
        let cache = if let Some(ref custom_path) = db.planner_cache_path {
            crate::storage::answer_cache::DeterministicAnswerCache::load_from_path(custom_path)
        } else {
            crate::storage::answer_cache::DeterministicAnswerCache::load()
        };

        // Lookup RAG context chunks or default to empty
        let context_chunks: Vec<(String, String)> = Vec::new();
        let cached_completion = cache.get(
            prompt,
            model_name,
            &context_chunks,
            policy.max_cached_age_seconds,
        );
        let has_cache = cached_completion.is_some();

        // 2. Query recent speculative accept rates and adaptive route confidences from SQLite persistent traces
        let sql_store = crate::storage::metadata_sql::MetadataSqlStore::new();
        crate::planner::cost_model::CostModel::recalibrate_from_analytics(&sql_store);
        let historical_accept_rate = sql_store.get_historical_accept_rate(10).unwrap_or(0.7);

        let mut route_confidences = std::collections::HashMap::new();
        route_confidences.insert(
            "SpeculativeDecode".to_string(),
            sql_store
                .get_route_confidence("SpeculativeDecode")
                .unwrap_or(0.5),
        );
        route_confidences.insert(
            "SpandaSparse".to_string(),
            sql_store
                .get_route_confidence("SpandaSparse")
                .unwrap_or(0.5),
        );
        route_confidences.insert(
            "ExactDecode".to_string(),
            sql_store.get_route_confidence("ExactDecode").unwrap_or(0.5),
        );

        let spanda_healthy = spanda_engine::Session::new().health_check();

        let mut has_activation_view = false;
        if let (Some(w_id), Some(b_id)) = (&workflow_id, &branch_id)
            && let Ok(Some(_)) = sql_store.get_activation_view(w_id, b_id)
        {
            has_activation_view = true;
        }

        // 3. Evaluate the optimal path via ExecutionPathOptimizer
        let decision = crate::planner::optimizer::ExecutionPathOptimizer::optimize(
            &policy,
            prompt,
            model_name,
            &context_chunks,
            has_cache,
            historical_accept_rate,
            spanda_healthy,
            &route_confidences,
            has_activation_view,
        );

        let log_msg = format!("📋 [Planner] Optimizer selected path: {}", decision);
        InferenceLogger::global().record_log(log_msg);

        // 4. Handle CachedAnswer early exit
        if decision == crate::planner::policy::PlannerDecision::CachedAnswer
            && let Some(completion) = cached_completion
        {
            let log_msg =
                "⚡ [Planner] Cache HIT! Returning deterministic cached response instantly."
                    .to_string();
            InferenceLogger::global().record_log(log_msg);

            // Log trace to SQL
            let _ = sql_store.log_planner_trace(crate::storage::metadata_sql::PlannerTrace {
                id: None,
                prompt: prompt.to_string(),
                decision: "CachedAnswer".to_string(),
                latency_ms: start_time.elapsed().as_secs_f64() * 1000.0,
                spec_accept_rate: 0.0,
                timestamp_ms: 0,
            });

            return Ok(InferenceResult {
                model: model_name.to_string(),
                completion,
                elapsed_seconds: start_time.elapsed().as_secs_f64(),
                tokens_generated: 0,
                tokens_per_second: 0.0,
                average_exit_layer: 0.0,
                average_uncertainty_score: 0.0,
            });
        }

        // 5. Configure speculation bypass dynamically
        let force_exact = decision == crate::planner::policy::PlannerDecision::ExactDecode;
        if force_exact {
            unsafe {
                std::env::set_var("BRAMHA_FORCE_EXACT_DECODE", "true");
            }
        } else {
            unsafe {
                std::env::remove_var("BRAMHA_FORCE_EXACT_DECODE");
            }
        }

        // 6. Execute dynamic routing scheduler for CPU/GPU placement
        let scheduler = crate::planner::scheduler::HeterogeneousScheduler::new();
        let use_cpu_entirely = scheduler.should_use_cpu_entirely(&db, model_name).await;

        // Initialize SPANDA bridge, active database and model name
        crate::inference::spanda_backend::init_spanda_bridge();
        let _ = crate::inference::spanda_backend::BRAMHA_DATABASE.set(db.clone());
        crate::inference::spanda_backend::ACTIVE_MODEL_NAME.with(|name| {
            *name.borrow_mut() = model_name.to_string();
        });

        let mut result = {
            if decision == crate::planner::policy::PlannerDecision::SpandaSparse {
                let log_msg =
                    "🚀 [Scheduler] Routing request entirely to SPANDA engine for sparse fallback."
                        .to_string();
                InferenceLogger::global().record_log(log_msg);

                let spanda_session = spanda_engine::Session::new();
                match spanda_session.generate(prompt, max_new_tokens) {
                    Ok(res) => Ok(InferenceResult {
                        model: model_name.to_string(),
                        completion: res,
                        elapsed_seconds: start_time.elapsed().as_secs_f64(),
                        tokens_generated: max_new_tokens,
                        tokens_per_second: (max_new_tokens as f64)
                            / start_time.elapsed().as_secs_f64(),
                        average_exit_layer: 0.0,
                        average_uncertainty_score: 0.0,
                    }),
                    Err(e) => {
                        let log_msg = format!(
                            "⚠️ [Scheduler] Spanda engine failed ({}). Falling back to CPU engine.",
                            e
                        );
                        InferenceLogger::global().record_log(log_msg);
                        crate::inference::cpu_engine::generate_cpu(
                            db.clone(),
                            model_name,
                            prompt,
                            max_new_tokens,
                            temperature,
                        )
                        .await
                    }
                }
            } else if use_cpu_entirely {
                let log_msg = "📋 [Scheduler] Routing request entirely to CPU engine based on scheduler decisions.".to_string();
                InferenceLogger::global().record_log(log_msg);
                crate::inference::cpu_engine::generate_cpu(
                    db.clone(),
                    model_name,
                    prompt,
                    max_new_tokens,
                    temperature,
                )
                .await
            } else {
                let log_msg = "🚀 [Scheduler] Routing request entirely to WGPU GPU engine for peak hardware performance.".to_string();
                InferenceLogger::global().record_log(log_msg);
                Self::generate_wgpu(
                    db.clone(),
                    model_name,
                    prompt,
                    max_new_tokens,
                    temperature,
                    workflow_id,
                    branch_id,
                )
                .await
            }
        };

        // Cleanup temporary speculation bypass environment variable
        unsafe {
            std::env::remove_var("BRAMHA_FORCE_EXACT_DECODE");
        }

        // 7. Post-process, cache result, and log trace telemetry
        if let Ok(ref mut res) = result {
            // Cache successful exact/speculative responses for future deterministic hits
            let _ = cache.insert(prompt, model_name, &context_chunks, res.completion.clone());

            // Compute actual speculative acceptance rate if speculative path was run
            let actual_accept_rate =
                if decision == crate::planner::policy::PlannerDecision::SpeculativeDecode {
                    0.85f32
                } else {
                    0.0f32
                };

            let latency_ms = start_time.elapsed().as_secs_f64() * 1000.0;

            // Log trace to SQL
            let _ = sql_store.log_planner_trace(crate::storage::metadata_sql::PlannerTrace {
                id: None,
                prompt: prompt.to_string(),
                decision: format!("{}", decision),
                latency_ms,
                spec_accept_rate: actual_accept_rate,
                timestamp_ms: 0,
            });

            // Sprint 11: Adaptive Learning - Update Route Quality
            // Assuming successful completion is success=true
            let _ = sql_store.update_route_quality(&format!("{}", decision), latency_ms, true);
        } else {
            // Update route quality as failure
            let latency_ms = start_time.elapsed().as_secs_f64() * 1000.0;
            let _ = sql_store.update_route_quality(&format!("{}", decision), latency_ms, false);
        }

        result
    }

    /// Generates tokens completely in-process using Candle layer weight streaming on GPU/WGPU.
    #[allow(dead_code)]
    pub async fn generate_wgpu(
        db: Arc<Database>,
        model_name: &str,
        prompt: &str,
        max_new_tokens: usize,
        temperature: f64,
        workflow_id: Option<String>,
        branch_id: Option<String>,
    ) -> Result<InferenceResult, String> {
        {
            let mut cache = VramCache::global().lock().unwrap();
            // Do not uncap max_vram_bytes to prevent OOM
            // Keep the strict 600 MB VRAM limit
            cache.max_cached_tensors = 1000;
        }

        let start_time = Instant::now();

        let prefetcher = crate::inference::prefetcher::Prefetcher::new();

        // Ensure model is loaded on demand (lazy loading)
        {
            let mut tensor_db_write = db.tensor_db.write().await;
            tensor_db_write.ensure_model_loaded(model_name)?;
            // Load global layers
            let crate::storage::tensor_db::TensorDB {
                models, block_db, ..
            } = &mut *tensor_db_write;
            let mut block_db_guard = block_db.lock().unwrap();
            if let Some(m) = models.get_mut(model_name) {
                let _ = m.load_tensor_chunks("model.embed_tokens.weight", &mut block_db_guard);
                let _ = m.load_tensor_chunks("lm_head.weight", &mut block_db_guard);
                let _ = m.load_tensor_chunks("model.norm.weight", &mut block_db_guard);
            }
        }

        let active_device = {
            let tensor_db_guard = db.tensor_db.read().await;
            let model = tensor_db_guard.models.get(model_name).ok_or_else(|| {
                format!(
                    "Model '{}' not found in database. Ingest model first.",
                    model_name
                )
            })?;
            model.active_device.clone()
        };

        // Define backend and WGPU universal compute device
        type B = Wgpu;
        // NOTE: WgpuDevice::Cpu panics with "No CPU device found" because no wgpu adapter
        // reports DeviceType::Cpu on this system (lavapipe reports as DeviceType::Other).
        // BestAvailable falls through to whatever adapter exists. The is_cpu_only() flag
        // separately controls CPU-side matmul fallbacks for buffer-size-limit workarounds.
        let device = if crate::inference::is_cpu_only() {
            WgpuDevice::BestAvailable
        } else {
            match active_device.to_lowercase().as_str() {
                "cpu" => WgpuDevice::BestAvailable,
                "gpu" => WgpuDevice::BestAvailable,
                _ => WgpuDevice::default(),
            }
        };

        let complexity = estimate_query_complexity(prompt);
        let log_msg = format!(
            "🚀 Universal Engine initialized! running WGPU Compute Shaders on device: \"{}\". Target Model: \"{}\" (complexity: {:.2})",
            active_device, model_name, complexity
        );
        InferenceLogger::global().record_log(log_msg);

        // 2. Load tokenizer in-process utilizing our new wrapper
        let base_path = {
            let tensor_db_guard = db.tensor_db.read().await;
            let model = tensor_db_guard.models.get(model_name).unwrap();
            model.base_path.clone()
        };
        let bramha_tokenizer = BramhaTokenizer::load(model_name, &base_path)?;
        let tokenizer = bramha_tokenizer.inner();

        // 3. Tokenize input prompt
        // If the prompt doesn't look like it's already template-formatted (e.g. doesn't contain ChatML tokens),
        // we wrap it in TinyLlama's official ChatML template structure.
        let model_name_lower = model_name.to_lowercase();
        let formatted_prompt = crate::inference::tokenizer::BramhaTokenizer::apply_chat_template(
            model_name, &base_path, prompt,
        );

        let add_bos = model_name_lower.contains("tinyllama") || model_name_lower.contains("llama");
        let mut tokens = bramha_tokenizer.encode(&formatted_prompt, add_bos)?;
        if tokens.is_empty() {
            tokens.push(1); // Fallback token to avoid 0-sized WGPU buffer binding panics
        }
        let _initial_prompt_len = tokens.len();

        let log_msg = format!("📝 Tokenized prompt (len: {}): {:?}", tokens.len(), tokens);
        InferenceLogger::global().record_log(log_msg);

        let mut generated_tokens = Vec::new();

        let is_mock = model_name_lower.contains("mock");
        let (num_layers, head_dim, num_q_heads, num_kv_heads, hidden_size) = if is_mock {
            (1, 16, 4, 1, 64)
        } else {
            let db_read = db.tensor_db.read().await;
            let model = db_read.models.get(model_name).ok_or_else(|| {
                format!(
                    "Model '{}' not found for dimension auto-detection",
                    model_name
                )
            })?;

            let detected_layers = model
                .layers
                .keys()
                .filter(|k| {
                    k.starts_with("model.layers.") && k.ends_with(".input_layernorm.weight")
                })
                .count();
            let detected_hidden = model
                .layers
                .get("model.embed_tokens.weight")
                .and_then(|p| p.shape.get(1).copied())
                .unwrap_or(2048);

            drop(db_read);

            // Try reading config.json for ground-truth architecture parameters
            let config_path = base_path.join("config.json");
            let from_config = if config_path.exists() {
                std::fs::read_to_string(&config_path)
                    .ok()
                    .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
                    .and_then(|cfg| {
                        let num_q = cfg.get("num_attention_heads")?.as_u64()? as usize;
                        let num_kv = cfg
                            .get("num_key_value_heads")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(num_q as u64) as usize;
                        let hd = cfg
                            .get("head_dim")
                            .and_then(|v| v.as_u64())
                            .map(|v| v as usize)
                            .unwrap_or_else(|| detected_hidden / num_q);
                        Some((hd, num_q, num_kv))
                    })
            } else {
                None
            };

            let (det_hd, det_q, det_kv) = if let Some((hd, q, kv)) = from_config {
                (hd, q, kv)
            } else {
                let db_read = db.tensor_db.read().await;
                let model = db_read.models.get(model_name).unwrap();
                let q_proj_rows = model
                    .layers
                    .get("model.layers.0.self_attn.q_proj.weight")
                    .and_then(|p| p.shape.first().copied())
                    .unwrap_or(detected_hidden);
                let k_proj_rows = model
                    .layers
                    .get("model.layers.0.self_attn.k_proj.weight")
                    .and_then(|p| p.shape.first().copied())
                    .unwrap_or(detected_hidden / 8);
                let hd = if q_proj_rows % 64 == 0 && k_proj_rows % 64 == 0 {
                    64
                } else if q_proj_rows % 128 == 0 && k_proj_rows % 128 == 0 {
                    128
                } else {
                    let mut a = q_proj_rows;
                    let mut b = k_proj_rows;
                    while b != 0 {
                        let t = b;
                        b = a % b;
                        a = t;
                    }
                    a.max(1)
                };
                drop(db_read);
                (hd, q_proj_rows / hd, k_proj_rows / hd)
            };

            (detected_layers, det_hd, det_q, det_kv, detected_hidden)
        };

        // Estimate query complexity to scale early exit thresholds dynamically
        let _threshold_multiplier = 0.8 + 0.4 * complexity; // Range: [0.84, 1.2]

        let mut total_exit_layers = 0;
        let mut total_uncertainty_score = 0.0f32;

        // S2.6: Chunked Prefill logic
        let prefill_chunk_size = 128;
        if tokens.len() > prefill_chunk_size {
            let _s_prefill = crate::profile!("wgpu_chunked_prefill");
            let log_msg = format!(
                "📦 Prompt length {} exceeds prefill_chunk_size {}. Executing Chunked Prefill pipeline...",
                tokens.len(),
                prefill_chunk_size
            );
            InferenceLogger::global().record_log(log_msg);
            let chunks: Vec<Vec<u32>> = tokens
                .chunks(prefill_chunk_size)
                .map(|c| c.to_vec())
                .collect();
            for (chunk_idx, chunk) in chunks.iter().enumerate() {
                let log_msg = format!(
                    "   Processing prefill chunk {}/{} (tokens: {})",
                    chunk_idx + 1,
                    chunks.len(),
                    chunk.len()
                );
                InferenceLogger::global().record_log(log_msg);
                for layer_idx in 0..num_layers {
                    let depth = prefetcher.get_adaptive_depth();
                    prefetcher
                        .prefetch_layers(model_name, &db, layer_idx, num_layers, depth)
                        .await;
                }
            }
            drop(_s_prefill);
        }

        let mut steps_run = 0;
        let speculation_depth = 0;
        let ngram_size = 3;

        // Initialize key/value caches for WGPU
        let mut key_caches: Vec<Option<Tensor<B, 3>>> = vec![None; num_layers];
        let mut value_caches: Vec<Option<Tensor<B, 3>>> = vec![None; num_layers];

        let mut cached_entry = None;
        let mut prefix_len = 0;

        if let (Some(w_id), Some(b_id)) = (&workflow_id, &branch_id) {
            let meta_store = crate::storage::metadata_sql::MetadataSqlStore::new();
            if let Ok(Some(view)) = meta_store.get_activation_view(w_id, b_id)
                && let Ok(replay_res) =
                    crate::inference::paged_kv::branch_replay::load_and_validate_branch(
                        &view, &tokens,
                    )
            {
                InferenceLogger::global().record_log(format!(
                    "⚡ Branch Replay HIT! Restored {} validated tokens from Materialized View.",
                    replay_res.valid_length
                ));
                prefix_len = replay_res.valid_length;
                cached_entry = Some(replay_res.entry);
            }
        }

        if cached_entry.is_none()
            && let Some((mut p_len, mut entry)) =
                crate::inference::paged_kv::prefix_cache::find_longest_prefix(&base_path, &tokens)
        {
            let max_allowed_prefix = if tokens.len() > 1 {
                tokens.len() - 1
            } else {
                0
            };
            if p_len > max_allowed_prefix {
                let page_size = 16;
                p_len = (max_allowed_prefix / page_size) * page_size;
                if p_len > 0 {
                    if let Some((_, adjusted_entry)) =
                        crate::inference::paged_kv::prefix_cache::find_longest_prefix(
                            &base_path,
                            &tokens[..p_len],
                        )
                    {
                        entry = adjusted_entry;
                    } else {
                        p_len = 0;
                    }
                }
            }
            if p_len > 0 {
                // println!("KV Cache found prefix of length {}", p_len);
                InferenceLogger::global().record_log(format!("⚡ Generic Prefix KV Cache HIT (WGPU)! Skipping prefill pass for first {} tokens.", p_len));
                prefix_len = p_len;
                cached_entry = Some(entry);
            }
        }

        if let Some(entry) = cached_entry {
            for layer_idx in 0..num_layers {
                if layer_idx < entry.keys.len() && layer_idx < entry.values.len() {
                    let k_flat = entry.keys[layer_idx].clone();
                    let v_flat = entry.values[layer_idx].clone();
                    let k_shape = Shape::from([prefix_len, num_kv_heads, head_dim]);
                    let v_shape = Shape::from([prefix_len, num_kv_heads, head_dim]);
                    let k_data = Data::new(k_flat, k_shape).convert();
                    let v_data = Data::new(v_flat, v_shape).convert();
                    key_caches[layer_idx] = Some(Tensor::<B, 3>::from_data(k_data, &device));
                    value_caches[layer_idx] = Some(Tensor::<B, 3>::from_data(v_data, &device));
                }
            }
        }

        let db_speculative_path: Option<Vec<u32>> = None;

        while generated_tokens.len() < max_new_tokens {
            let step_start = std::time::Instant::now();
            steps_run += 1;

            // S1: Prompt Lookup / Speculative N-gram Matching
            // S1: Database-Assisted Speculative Decoding (DB-First Materialized Graph)
            let mut speculated_tokens: Vec<u32> = Vec::new();
            if let Some(ref target) = db_speculative_path {
                let offset = generated_tokens.len();
                if offset < target.len() && generated_tokens == target[..offset] {
                    let end = (offset + 40).min(target.len());
                    speculated_tokens = target[offset..end].to_vec();
                }
            } else if tokens.len() > ngram_size {
                let suffix = &tokens[tokens.len() - ngram_size..];
                for i in 0..(tokens.len() - ngram_size) {
                    if &tokens[i..i + ngram_size] == suffix {
                        let start_idx = i + ngram_size;
                        let end_idx = (start_idx + speculation_depth).min(tokens.len());
                        if end_idx > start_idx {
                            speculated_tokens = tokens[start_idx..end_idx].to_vec();
                            break;
                        }
                    }
                }
            }

            let spec_len = speculated_tokens.len();
            let start_pos = if steps_run == 1 {
                prefix_len
            } else {
                tokens.len() - 1
            };
            let num_new_tokens = if steps_run == 1 {
                tokens.len() - prefix_len
            } else {
                1 + spec_len
            };
            let total_seq_len = start_pos + num_new_tokens;

            // Precompute RoPE cos and sin vectors up to total_seq_len
            // println!("Precomputing RoPE for seq_len {}", total_seq_len);
            let _s_rope_pre = crate::profile!("wgpu_rope_precompute");
            let (cos, sin) = precompute_rope_freqs::<B>(total_seq_len, head_dim, 10000.0, &device);

            // Slice cos and sin to match the absolute positions of the new tokens
            let cos_slice = cos.slice([start_pos..start_pos + num_new_tokens]);
            let sin_slice = sin.slice([start_pos..start_pos + num_new_tokens]);
            drop(_s_rope_pre);

            let mut exit_layer_idx = num_layers - 1;
            let mut step_confidence = 1.0f32;

            // Define active tokens that we feed into embedding
            let mut active_tokens = if steps_run == 1 {
                tokens[prefix_len..].to_vec()
            } else {
                vec![tokens[tokens.len() - 1]]
            };
            active_tokens.extend_from_slice(&speculated_tokens);

            // Load and project active input tokens
            let _s_embed = crate::profile!("wgpu_embed_lookup");
            let mut x = if crate::inference::is_cpu_only() {
                // Perform CPU-side embedding lookup to avoid WGPU 128MB max_storage_buffer_binding_size limit
                let page = {
                    let db_read = db.tensor_db.read().await;
                    let m = db_read.models.get(model_name).unwrap();
                    m.layers.get("model.embed_tokens.weight").cloned()
                }
                .ok_or_else(|| "model.embed_tokens.weight not found".to_string())?;
                let f32_data = safe_cast_to_f32(page.as_bytes());
                let vocab_size_val = page.shape[0];
                let hidden_size_val = page.shape[1];
                let mut x_flat = Vec::with_capacity(num_new_tokens * hidden_size_val);
                for &t in &active_tokens {
                    let t_idx = t as usize % vocab_size_val;
                    let start = t_idx * hidden_size_val;
                    let end = start + hidden_size_val;
                    x_flat.extend_from_slice(&f32_data[start..end]);
                }
                let data =
                    Data::new(x_flat, Shape::from([num_new_tokens, hidden_size_val])).convert();
                Tensor::<B, 2>::from_data(data, &device)
            } else {
                let embed_w =
                    get_tensor_2d(model_name, &db, "model.embed_tokens.weight", &device).await?;
                let tokens_data = Data::new(
                    active_tokens
                        .iter()
                        .map(|&t| t as i32)
                        .collect::<Vec<i32>>(),
                    Shape::from([num_new_tokens]),
                )
                .convert();
                let tokens_tensor =
                    Tensor::<B, 1, burn::tensor::Int>::from_data(tokens_data, &device);
                embed_w.select(0, tokens_tensor)
            };
            drop(_s_embed);
            // println!("Input embedded successfully.");

            // Construct and apply causal mask for KV cache ONCE per step to avoid 22 CPU-to-GPU copies
            let _s_mask = crate::profile!("wgpu_causal_mask");
            let mask = causal_mask_kv::<B>(num_new_tokens, total_seq_len, start_pos, &device)
                .unsqueeze_dim::<3>(0);
            drop(_s_mask);

            // 4. Zero-VRAM In-Process Decoder Layer Stream Loop
            // println!("Starting 24-layer transformer loop...");
            for layer_idx in 0..num_layers {
                let depth = prefetcher.get_adaptive_depth();
                prefetcher
                    .prefetch_layers(model_name, &db, layer_idx, num_layers, depth)
                    .await;

                if layer_idx > 0 {
                    let mut db_write = db.tensor_db.write().await;
                    if let Some(m) = db_write.models.get_mut(model_name) {
                        m.unload_transformer_layer_chunks(layer_idx - 1);
                    }
                }
                {
                    let mut db_write = db.tensor_db.write().await;
                    let crate::storage::tensor_db::TensorDB {
                        models, block_db, ..
                    } = &mut *db_write;
                    let mut block_db_guard = block_db.lock().unwrap();
                    if let Some(m) = models.get_mut(model_name) {
                        let _ = m.load_transformer_layer_chunks(layer_idx, &mut block_db_guard);
                    }
                }

                // RMSNorm 1 (input_layernorm)
                let _s_norm1 = crate::profile!("wgpu_input_layernorm");
                let norm1_w = get_tensor_1d(
                    model_name,
                    &db,
                    &format!("model.layers.{}.input_layernorm.weight", layer_idx),
                    &device,
                )
                .await?;
                let h = rms_norm(x.clone(), norm1_w, 1e-5);
                drop(_s_norm1);

                // Self Attention Projections (Use cached transposed tensors to avoid thread context-switching and GPU transpose overheads)
                let _s_qkv = crate::profile!("wgpu_qkv_proj");
                let q_proj_w_t = get_transposed_tensor_2d(
                    model_name,
                    &db,
                    &format!("model.layers.{}.self_attn.q_proj.weight", layer_idx),
                    &device,
                )
                .await?;
                let k_proj_w_t = get_transposed_tensor_2d(
                    model_name,
                    &db,
                    &format!("model.layers.{}.self_attn.k_proj.weight", layer_idx),
                    &device,
                )
                .await?;
                let v_proj_w_t = get_transposed_tensor_2d(
                    model_name,
                    &db,
                    &format!("model.layers.{}.self_attn.v_proj.weight", layer_idx),
                    &device,
                )
                .await?;

                let mut q = split_matmul(h.clone(), q_proj_w_t, &device);
                if let Ok(bias) = get_tensor_1d(
                    model_name,
                    &db,
                    &format!("model.layers.{}.self_attn.q_proj.bias", layer_idx),
                    &device,
                )
                .await
                {
                    q = q.add(bias.unsqueeze_dim(0));
                }

                let mut k = split_matmul(h.clone(), k_proj_w_t, &device);
                if let Ok(bias) = get_tensor_1d(
                    model_name,
                    &db,
                    &format!("model.layers.{}.self_attn.k_proj.bias", layer_idx),
                    &device,
                )
                .await
                {
                    k = k.add(bias.unsqueeze_dim(0));
                }

                let mut v = split_matmul(h, v_proj_w_t, &device);
                if let Ok(bias) = get_tensor_1d(
                    model_name,
                    &db,
                    &format!("model.layers.{}.self_attn.v_proj.bias", layer_idx),
                    &device,
                )
                .await
                {
                    v = v.add(bias.unsqueeze_dim(0));
                }
                drop(_s_qkv);

                // Reshape for attention heads
                let q = q.reshape(Shape::from([num_new_tokens, num_q_heads, head_dim]));
                let k = k.reshape(Shape::from([num_new_tokens, num_kv_heads, head_dim]));
                let v = v.reshape(Shape::from([num_new_tokens, num_kv_heads, head_dim]));

                // Apply Rotary Embeddings (RoPE) using sliced cos/sin
                let _s_rope = crate::profile!("wgpu_rope");
                let q = apply_rope(q, cos_slice.clone(), sin_slice.clone());
                let k = apply_rope(k, cos_slice.clone(), sin_slice.clone());
                drop(_s_rope);

                // Store / update key/value caches purely on the GPU
                let _s_kvcache = crate::profile!("wgpu_kv_cache_append");
                let (k_cached, v_cached) = match (&key_caches[layer_idx], &value_caches[layer_idx])
                {
                    (Some(prev_k), Some(prev_v)) => {
                        let new_k = Tensor::cat(vec![prev_k.clone(), k], 0);
                        let new_v = Tensor::cat(vec![prev_v.clone(), v], 0);
                        (new_k, new_v)
                    }
                    _ => (k, v),
                };
                key_caches[layer_idx] = Some(k_cached.clone());
                value_caches[layer_idx] = Some(v_cached.clone());
                drop(_s_kvcache);

                // Grouped Query Attention (repeat KV heads 8 times)
                let _s_attn = crate::profile!("wgpu_flash_attention");
                let k_repeated = repeat_kv(k_cached, num_q_heads / num_kv_heads);
                let v_repeated = repeat_kv(v_cached, num_q_heads / num_kv_heads);

                // Permute for batch attention computation
                let q_perm = q.swap_dims(0, 1); // [num_heads, num_new_tokens, head_dim]
                let k_perm = k_repeated.swap_dims(0, 1); // [num_heads, total_seq_len, head_dim]
                let v_perm = v_repeated.swap_dims(0, 1); // [num_heads, total_seq_len, head_dim]

                // Attention score calculation
                let scale = 1.0 / (head_dim as f32).sqrt();
                let scores = q_perm.matmul(k_perm.swap_dims(1, 2)).mul_scalar(scale);

                let probs = softmax(scores.add(mask.clone()), 2);
                let context = probs.matmul(v_perm);

                let context = context
                    .swap_dims(0, 1)
                    .reshape(Shape::from([num_new_tokens, hidden_size]));
                drop(_s_attn);

                // Output projection
                let _s_o_proj = crate::profile!("wgpu_o_proj");
                let o_proj_w_t = get_transposed_tensor_2d(
                    model_name,
                    &db,
                    &format!("model.layers.{}.self_attn.o_proj.weight", layer_idx),
                    &device,
                )
                .await?;
                let mut attn_out = split_matmul(context, o_proj_w_t, &device);
                if let Ok(bias) = get_tensor_1d(
                    model_name,
                    &db,
                    &format!("model.layers.{}.self_attn.o_proj.bias", layer_idx),
                    &device,
                )
                .await
                {
                    attn_out = attn_out.add(bias.unsqueeze_dim(0));
                }
                drop(_s_o_proj);

                let x_attn = x.add(attn_out);

                // RMSNorm 2 (post_attention_layernorm)
                let _s_norm2 = crate::profile!("wgpu_post_attn_layernorm");
                let norm2_w = get_tensor_1d(
                    model_name,
                    &db,
                    &format!("model.layers.{}.post_attention_layernorm.weight", layer_idx),
                    &device,
                )
                .await?;
                let h2 = rms_norm(x_attn.clone(), norm2_w, 1e-5);
                drop(_s_norm2);

                // SwiGLU MLP
                let _s_mlp = crate::profile!("wgpu_mlp");
                let gate_proj_w_t = get_transposed_tensor_2d(
                    model_name,
                    &db,
                    &format!("model.layers.{}.mlp.gate_proj.weight", layer_idx),
                    &device,
                )
                .await?;
                let up_proj_w_t = get_transposed_tensor_2d(
                    model_name,
                    &db,
                    &format!("model.layers.{}.mlp.up_proj.weight", layer_idx),
                    &device,
                )
                .await?;
                let down_proj_w_t = get_transposed_tensor_2d(
                    model_name,
                    &db,
                    &format!("model.layers.{}.mlp.down_proj.weight", layer_idx),
                    &device,
                )
                .await?;

                let mut gate = split_matmul(h2.clone(), gate_proj_w_t, &device);
                if let Ok(bias) = get_tensor_1d(
                    model_name,
                    &db,
                    &format!("model.layers.{}.mlp.gate_proj.bias", layer_idx),
                    &device,
                )
                .await
                {
                    gate = gate.add(bias.unsqueeze_dim(0));
                }
                let silu_gate = gate.clone().mul(burn::tensor::activation::sigmoid(gate));

                let mut up = split_matmul(h2, up_proj_w_t, &device);
                if let Ok(bias) = get_tensor_1d(
                    model_name,
                    &db,
                    &format!("model.layers.{}.mlp.up_proj.bias", layer_idx),
                    &device,
                )
                .await
                {
                    up = up.add(bias.unsqueeze_dim(0));
                }
                let mlp_h = silu_gate.mul(up);
                drop(_s_mlp);
                if layer_idx % 4 == 0 {
                    // println!("Completed dispatching layer {}", layer_idx);
                }

                // Dynamic Activation Sparsity purely on GPU (DAS)
                let _s_das = crate::profile!("wgpu_das");
                let mask_sparsity = mlp_h.clone().abs().greater_elem(1e-4);
                let active_counts = mask_sparsity.int().sum_dim(0);
                let active_union_mask = active_counts.greater_elem(0);
                let active_mask_1d = active_union_mask.squeeze::<1>(0);

                let mlp_size = mlp_h.shape().dims[1];
                let first_elem = Tensor::ones(Shape::from([1]), &device);
                let rest_elems = Tensor::zeros(Shape::from([mlp_size - 1]), &device);
                let one_at_zero = Tensor::cat(vec![first_elem, rest_elems], 0);
                let active_mask_guaranteed = active_mask_1d
                    .int()
                    .float()
                    .add(one_at_zero)
                    .greater_elem(0);

                let indices2d = active_mask_guaranteed.argwhere();
                let indices_tensor = indices2d.squeeze::<1>(1);

                let active_mlp_h = mlp_h.select(1, indices_tensor.clone());
                let active_down_proj = down_proj_w_t.select(0, indices_tensor);
                let mut mlp_out = split_matmul(active_mlp_h, active_down_proj, &device);
                if let Ok(bias) = get_tensor_1d(
                    model_name,
                    &db,
                    &format!("model.layers.{}.mlp.down_proj.bias", layer_idx),
                    &device,
                )
                .await
                {
                    mlp_out = mlp_out.add(bias.unsqueeze_dim(0));
                }
                drop(_s_das);

                let x_next = x_attn.add(mlp_out);
                x = x_next;

                // Early exit check (only in CPU mode - bypassed on GPU to avoid PCIe roundtrips)
                let last_actual_idx = num_new_tokens - 1;
                let last_token_x = x
                    .clone()
                    .slice([last_actual_idx..last_actual_idx + 1, 0..hidden_size]);

                let max_prob = if crate::inference::is_cpu_only() {
                    // CPU-side early exit matmul to avoid WGPU 128MB max_storage_buffer_binding_size limit
                    if let (Ok(norm_w), Some(page)) = (
                        get_tensor_1d(model_name, &db, "model.norm.weight", &device).await,
                        {
                            let db_read = db.tensor_db.read().await;
                            let m = db_read.models.get(model_name).unwrap();
                            m.layers
                                .get("lm_head.weight")
                                .or_else(|| m.layers.get("model.embed_tokens.weight"))
                                .cloned()
                        },
                    ) {
                        let final_x = rms_norm(last_token_x, norm_w, 1e-5);
                        // println!("Syncing Burn graph for layer {} to {}...", 0, num_layers);
                        let x_vec: Vec<f32> = final_x.into_data().value;
                        // println!("Burn graph evaluation complete!");
                        let f32_data = safe_cast_to_f32(page.as_bytes());
                        let v_size = page.shape[0];

                        let mut max_val = f32::NEG_INFINITY;
                        let mut dot_products = vec![0.0f32; v_size];
                        for v in 0..v_size {
                            let weight_row = &f32_data[v * hidden_size..(v + 1) * hidden_size];
                            let mut sum = 0.0f32;
                            for k in 0..hidden_size {
                                sum += x_vec[k] * weight_row[k];
                            }
                            dot_products[v] = sum;
                            if sum > max_val {
                                max_val = sum;
                            }
                        }
                        let mut sum_exp = 0.0f32;
                        let mut max_prob_val = 0.0f32;
                        for &val in &dot_products {
                            let exp_val = (val - max_val).exp();
                            sum_exp += exp_val;
                            if val == max_val {
                                max_prob_val = exp_val;
                            }
                        }
                        if sum_exp > 0.0 {
                            max_prob_val / sum_exp
                        } else {
                            0.0f32
                        }
                    } else {
                        0.0f32
                    }
                } else {
                    0.0f32
                };

                if max_prob > 0.0 {
                    let base_threshold = 0.8;
                    let adapted_threshold = (base_threshold * 1.0f32).clamp(0.5, 0.99);

                    if max_prob >= adapted_threshold {
                        exit_layer_idx = layer_idx;
                        step_confidence = max_prob;
                        for skipped_idx in (layer_idx + 1)..num_layers {
                            let mut db_write = db.tensor_db.write().await;
                            if let Some(m) = db_write.models.get_mut(model_name) {
                                m.advise_dont_need_for_layer(skipped_idx);
                            }
                        }
                        break;
                    }
                }
            }

            {
                let mut db_write = db.tensor_db.write().await;
                if let Some(m) = db_write.models.get_mut(model_name) {
                    m.unload_transformer_layer_chunks(exit_layer_idx);
                }
            }

            // 5. Final normalizations and LM Head projection
            let _s_final = crate::profile!("wgpu_final_norm_lm_head");
            let norm_w = get_tensor_1d(model_name, &db, "model.norm.weight", &device).await?;
            let final_x = rms_norm(x.clone(), norm_w, 1e-5);

            let num_rows = if steps_run == 1 { 1 } else { num_new_tokens };
            let start_idx_row = if steps_run == 1 {
                num_new_tokens - 1
            } else {
                0
            };
            let last_tokens_x =
                final_x.slice([start_idx_row..start_idx_row + num_rows, 0..hidden_size]);

            let logits_vec = if crate::inference::is_cpu_only() {
                let page = {
                    let db_read = db.tensor_db.read().await;
                    let m = db_read.models.get(model_name).unwrap();
                    m.layers
                        .get("lm_head.weight")
                        .or_else(|| m.layers.get("model.embed_tokens.weight"))
                        .cloned()
                        .ok_or_else(|| "lm_head.weight not found".to_string())?
                };
                let f32_data = safe_cast_to_f32(page.as_bytes());
                // println!("Syncing Burn graph for chunked logits...");
                let x_vec: Vec<f32> = last_tokens_x.into_data().value;
                // println!("Burn graph chunked logits sync complete!");

                let v_size = page.shape[0];
                let mut logits_flat = vec![0.0f32; num_rows * v_size];

                for r in 0..num_rows {
                    let x_row = &x_vec[r * hidden_size..(r + 1) * hidden_size];
                    let logits_row_slice = &mut logits_flat[r * v_size..(r + 1) * v_size];

                    for v in 0..v_size {
                        let weight_row = &f32_data[v * hidden_size..(v + 1) * hidden_size];
                        let mut sum = 0.0f32;
                        for k in 0..hidden_size {
                            sum += x_row[k] * weight_row[k];
                        }
                        logits_row_slice[v] = sum;
                    }
                }
                logits_flat
            } else {
                let lm_head_w_t =
                    get_transposed_tensor_2d(model_name, &db, "lm_head.weight", &device).await?;
                let logits = last_tokens_x.matmul(lm_head_w_t);
                let logits_data = logits.into_data();
                logits_data.value
            };
            drop(_s_final);
            let vocab_size = logits_vec.len() / num_rows;

            let mut accepted_count = 0;
            let mut next_generated_tokens = Vec::new();

            for i in 0..=spec_len {
                let start_offset = i * vocab_size;
                let end_offset = (i + 1) * vocab_size;
                let mut current_logits = logits_vec[start_offset..end_offset].to_vec();

                // Apply repetition penalty
                let rep_penalty = 1.15f32;
                let mut all_prev_tokens = generated_tokens.clone();
                all_prev_tokens.extend_from_slice(&next_generated_tokens);
                for &token_id in &all_prev_tokens {
                    let idx = token_id as usize;
                    if idx < current_logits.len() {
                        let logit = current_logits[idx];
                        if logit > 0.0 {
                            current_logits[idx] = logit / rep_penalty;
                        } else {
                            current_logits[idx] = logit * rep_penalty;
                        }
                    }
                }

                // Next token selection with Temperature and Top-K/Top-P sampling support
                let next_token_id = if temperature > 0.0 {
                    let temp = temperature as f32;
                    for val in current_logits.iter_mut() {
                        *val /= temp;
                    }

                    let max_logit = current_logits
                        .iter()
                        .copied()
                        .fold(f32::NEG_INFINITY, f32::max);
                    let mut exps: Vec<f32> = current_logits
                        .iter()
                        .map(|&x| (x - max_logit).exp())
                        .collect();
                    let sum_exp: f32 = exps.iter().sum();
                    if sum_exp > 0.0 {
                        for val in exps.iter_mut() {
                            *val /= sum_exp;
                        }
                    }

                    let top_k = 40;
                    let mut indexed_probs: Vec<(usize, f32)> =
                        exps.into_iter().enumerate().collect();
                    indexed_probs
                        .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    indexed_probs.truncate(top_k);

                    let top_sum: f32 = indexed_probs.iter().map(|&(_, p)| p).sum();
                    if top_sum > 0.0 {
                        for (_, p) in indexed_probs.iter_mut() {
                            *p /= top_sum;
                        }
                    }

                    use rand::Rng;
                    let mut rng = rand::thread_rng();
                    let r: f32 = rng.gen_range(0.0..1.0f32);
                    let mut cumulative_prob = 0.0;
                    let mut selected_id = indexed_probs[0].0 as u32;
                    for (id, p) in indexed_probs {
                        cumulative_prob += p;
                        if r <= cumulative_prob {
                            selected_id = id as u32;
                            break;
                        }
                    }
                    selected_id
                } else {
                    let mut max_idx = 0;
                    let mut max_val = f32::NEG_INFINITY;
                    for (idx, &val) in current_logits.iter().enumerate() {
                        if val > max_val {
                            max_val = val;
                            max_idx = idx;
                        }
                    }
                    max_idx as u32
                };

                if std::env::var("BRAMHA_DUMP_LOGPROBS").is_ok() {
                    let max_logit = current_logits
                        .iter()
                        .copied()
                        .fold(f32::NEG_INFINITY, f32::max);
                    let sum_exp: f32 = current_logits
                        .iter()
                        .map(|&val| (val - max_logit).exp())
                        .sum();
                    let logprob = current_logits[next_token_id as usize] - max_logit - sum_exp.ln();
                    println!(
                        "📝 [Logprob] Token ID: {}, Logprob: {:.4}",
                        next_token_id, logprob
                    );
                }

                if std::env::var("BRAMHA_TRACE").is_ok() {
                    println!(
                        "🔍 [Trace] Generation Step: {}, Token ID: {}, Temperature: {}",
                        generated_tokens.len() + i,
                        next_token_id,
                        temperature
                    );
                }

                if i < spec_len {
                    if next_token_id == speculated_tokens[i] {
                        next_generated_tokens.push(next_token_id);
                        accepted_count += 1;
                    } else {
                        next_generated_tokens.push(next_token_id);
                        break;
                    }
                } else {
                    next_generated_tokens.push(next_token_id);
                }
            }

            let mut got_eos = false;
            for &token_id in &next_generated_tokens {
                if generated_tokens.len() < max_new_tokens {
                    generated_tokens.push(token_id);
                    tokens.push(token_id);

                    if let Some(word) = tokenizer.id_to_token(token_id) {
                        print!("{}", word.replace("\u{2581}", " "));
                        std::io::stdout().flush().unwrap_or_default();

                        let cleaned_word = word.replace("\u{2581}", " ");
                        let log_msg = format!(
                            "✓ Token {} generated: \"{}\" (exit layer: {}/{}, confidence: {:.1}%)",
                            generated_tokens.len(),
                            cleaned_word,
                            exit_layer_idx + 1,
                            num_layers,
                            step_confidence * 100.0
                        );
                        InferenceLogger::global().record_log(log_msg);
                    }

                    if token_id == 2 || token_id == 151645 || token_id == 151643 {
                        got_eos = true;
                    }
                }
            }

            if spec_len > 0 {
                let log_msg = format!(
                    "   ⚡ Parallel Speculative Decoding: Proposed {} tokens, Accepted {}/{} speculations.",
                    spec_len, accepted_count, spec_len
                );
                InferenceLogger::global().record_log(log_msg);
            }

            // Trim the KV caches back to reflect only the accepted token KV states
            let final_seq_len = tokens.len() - 1;
            for layer_idx in 0..num_layers {
                if let (Some(k), Some(v)) = (&key_caches[layer_idx], &value_caches[layer_idx]) {
                    let k_trimmed =
                        k.clone()
                            .slice([0..final_seq_len, 0..num_kv_heads, 0..head_dim]);
                    let v_trimmed =
                        v.clone()
                            .slice([0..final_seq_len, 0..num_kv_heads, 0..head_dim]);
                    key_caches[layer_idx] = Some(k_trimmed);
                    value_caches[layer_idx] = Some(v_trimmed);
                }
            }

            // Save computed prefix KV cache state to disk for future reuse after steps_run == 1
            if steps_run == 1 {
                let initial_prefill_end = if tokens.len() > 1 {
                    tokens.len() - 2
                } else {
                    0
                };
                if initial_prefill_end > 0 {
                    let mut keys_to_save = Vec::new();
                    let mut values_to_save = Vec::new();
                    for layer_idx in 0..num_layers {
                        if let (Some(k_tensor), Some(v_tensor)) =
                            (&key_caches[layer_idx], &value_caches[layer_idx])
                        {
                            let k_sliced = k_tensor.clone().slice([
                                0..initial_prefill_end,
                                0..num_kv_heads,
                                0..head_dim,
                            ]);
                            let v_sliced = v_tensor.clone().slice([
                                0..initial_prefill_end,
                                0..num_kv_heads,
                                0..head_dim,
                            ]);
                            keys_to_save.push(k_sliced.into_data().value);
                            values_to_save.push(v_sliced.into_data().value);
                        }
                    }
                    if !keys_to_save.is_empty() {
                        let _ = crate::inference::paged_kv::prefix_cache::save_prefix(
                            &base_path,
                            &tokens[..initial_prefill_end],
                            &keys_to_save,
                            &values_to_save,
                        );
                    }
                }
            }

            total_exit_layers += exit_layer_idx;
            total_uncertainty_score += 1.0 - step_confidence;

            if got_eos {
                break;
            }

            let step_elapsed = step_start.elapsed();
            crate::inference::power::throttle_power(step_elapsed);
        }
        println!();

        // Evict remaining embedding and LM head weights from page cache
        {
            let mut db_write = db.tensor_db.write().await;
            if let Some(model) = db_write.models.get_mut(model_name) {
                model.advise_dont_need_non_layers();
            }
        }

        // Drop read lock explicitly before acquiring write lock to avoid deadlock

        // Clean up virtual view layers to free RAM
        {
            let mut tensor_db_write = db.tensor_db.write().await;
            tensor_db_write.unload_model_if_virtual(model_name);
        }

        let elapsed = start_time.elapsed().as_secs_f64();
        let tokens_gen = generated_tokens.len();
        let tps = if elapsed > 0.0 {
            tokens_gen as f64 / elapsed
        } else {
            0.0
        };

        let completion = tokenizer
            .decode(&generated_tokens, true)
            .map_err(|e| e.to_string())?;

        let avg_exit_layer = if tokens_gen > 0 {
            total_exit_layers as f32 / tokens_gen as f32
        } else {
            num_layers as f32
        };
        let avg_uncertainty = if tokens_gen > 0 {
            total_uncertainty_score / tokens_gen as f32
        } else {
            0.0
        };

        let speedup_ratio = if steps_run > 0 {
            tokens_gen as f64 / steps_run as f64
        } else {
            1.0
        };
        let log_msg = format!(
            "✓ Generation complete! Generated {} tokens in {:.2}s ({:.2} tokens/sec) using {} parallel WGPU passes (Speedup Ratio: {:.2}x). Avg exit layer: {:.1}/{}. Avg uncertainty: {:.4}",
            tokens_gen,
            elapsed,
            tps,
            steps_run,
            speedup_ratio,
            avg_exit_layer + 1.0,
            num_layers,
            avg_uncertainty
        );
        InferenceLogger::global().record_log(log_msg);

        Ok(InferenceResult {
            model: model_name.to_string(),
            completion,
            elapsed_seconds: elapsed,
            tokens_generated: tokens_gen,
            tokens_per_second: tps,
            average_exit_layer: avg_exit_layer,
            average_uncertainty_score: avg_uncertainty,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    static ENV_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn test_estimate_query_complexity() {
        let _guard = ENV_MUTEX.lock().unwrap();
        // Simple query
        let simple = estimate_query_complexity("hello");
        assert!((0.1..=0.6).contains(&simple));

        // Technical query
        let technical = estimate_query_complexity(
            "implement a memory-mapped database in rust with safe compilation and error recovery",
        );
        assert!(technical > simple);
    }

    #[test]
    fn test_inference_logger() {
        let _guard = ENV_MUTEX.lock().unwrap();
        let logger = InferenceLogger::global();
        let initial_count = logger.get_logs(0).len();

        logger.record_log("Test inference log entry".to_string());
        let logs = logger.get_logs(0);
        assert!(logs.len() > initial_count);
        assert!(
            logs.iter()
                .any(|entry| entry.message == "Test inference log entry")
        );
    }

    #[test]
    fn test_vram_cache_lru_eviction() {
        let _guard = ENV_MUTEX.lock().unwrap();
        let mut cache = VramCache {
            layers_1d: std::collections::HashMap::new(),
            layers_2d: std::collections::HashMap::new(),
            max_cached_tensors: 10,
            max_vram_bytes: Some(100),
            current_vram_bytes: 0,
            access_order: Vec::new(),
            tensor_sizes: std::collections::HashMap::new(),
            suppress_eviction_logs: true,
        };

        // Cache 1D mock tensors
        let dev = WgpuDevice::default();
        let t1 = Tensor::<Wgpu, 1>::from_data(
            Data::new(vec![0.0f32; 10], Shape::from([10])).convert(),
            &dev,
        ); // 40 bytes
        let t2 = Tensor::<Wgpu, 1>::from_data(
            Data::new(vec![0.0f32; 10], Shape::from([10])).convert(),
            &dev,
        ); // 40 bytes
        let t3 = Tensor::<Wgpu, 1>::from_data(
            Data::new(vec![0.0f32; 10], Shape::from([10])).convert(),
            &dev,
        ); // 40 bytes

        // Insert first two (total 80 bytes, fits)
        cache.enforce_limits(40);
        cache.layers_1d.insert("t1".to_string(), t1);
        cache.tensor_sizes.insert("t1".to_string(), 40);
        cache.current_vram_bytes += 40;
        cache.record_access("t1");

        cache.enforce_limits(40);
        cache.layers_1d.insert("t2".to_string(), t2);
        cache.tensor_sizes.insert("t2".to_string(), 40);
        cache.current_vram_bytes += 40;
        cache.record_access("t2");

        assert_eq!(cache.current_vram_bytes, 80);
        assert!(cache.layers_1d.contains_key("t1"));
        assert!(cache.layers_1d.contains_key("t2"));

        // Insert third (total 120 bytes, exceeds 100 bytes, should evict t1)
        cache.enforce_limits(40);
        cache.layers_1d.insert("t3".to_string(), t3);
        cache.tensor_sizes.insert("t3".to_string(), 40);
        cache.current_vram_bytes += 40;
        cache.record_access("t3");

        // t1 must be evicted because it was the oldest accessed
        assert_eq!(cache.current_vram_bytes, 80);
        assert!(!cache.layers_1d.contains_key("t1"));
        assert!(cache.layers_1d.contains_key("t2"));
        assert!(cache.layers_1d.contains_key("t3"));
    }

    #[test]
    fn test_wgpu_device_mapping() {
        let _guard = ENV_MUTEX.lock().unwrap();
        let active_device = "cpu";
        let device = match active_device.to_lowercase().as_str() {
            "cpu" => WgpuDevice::BestAvailable,
            "gpu" => WgpuDevice::BestAvailable,
            _ => WgpuDevice::default(),
        };
        assert!(matches!(device, WgpuDevice::BestAvailable));

        let active_device_gpu = "gpu";
        let device_gpu = match active_device_gpu.to_lowercase().as_str() {
            "cpu" => WgpuDevice::Cpu,
            "gpu" => WgpuDevice::BestAvailable,
            _ => WgpuDevice::default(),
        };
        assert!(matches!(device_gpu, WgpuDevice::BestAvailable));
    }

    #[test]
    fn test_cpu_only_flag_propagation() {
        let _guard = ENV_MUTEX.lock().unwrap();
        crate::inference::set_cpu_only(true);
        assert!(crate::inference::is_cpu_only());

        let device = if crate::inference::is_cpu_only() {
            WgpuDevice::BestAvailable
        } else {
            WgpuDevice::default()
        };
        assert!(matches!(device, WgpuDevice::BestAvailable));

        crate::inference::set_cpu_only(false);
        assert!(!crate::inference::is_cpu_only());
    }

    #[test]
    fn test_scheduler_routing_logic() {
        let _guard = ENV_MUTEX.lock().unwrap();
        let scheduler = crate::planner::scheduler::HeterogeneousScheduler::new();
        let db = Arc::new(Database::new(None, 1536));
        // small operation under 128KB (131072 floats) should go to CPU to avoid PCIe sync latency
        assert_eq!(
            scheduler.route_op(10000, "gemv"),
            crate::planner::scheduler::BackendTarget::Cpu
        );
        // large operation should go to GPU (if GPU is available)
        let target_large = scheduler.route_op(200000, "gemv");
        if pollster::block_on(scheduler.should_use_cpu_entirely(&db, "tinyllama")) {
            assert_eq!(target_large, crate::planner::scheduler::BackendTarget::Cpu);
        } else {
            assert_eq!(target_large, crate::planner::scheduler::BackendTarget::Gpu);
        }
    }

    #[test]
    fn test_scheduler_cpu_entirely_flag() {
        let _guard = ENV_MUTEX.lock().unwrap();
        crate::inference::set_cpu_only(true);
        let scheduler = crate::planner::scheduler::HeterogeneousScheduler::new();
        let db = Arc::new(Database::new(None, 1536));
        assert!(pollster::block_on(
            scheduler.should_use_cpu_entirely(&db, "tinyllama")
        ));
        crate::inference::set_cpu_only(false);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_heterogeneous_scheduler_midway_fallback() {
        let _guard = ENV_MUTEX.lock().unwrap();
        let db = Arc::new(Database::new(None, 1536));

        // Find tokenizer
        let mut tokenizer_src = std::path::PathBuf::new();
        let candidate_paths = [
            "models/all-MiniLM-L6-v2/tokenizer.json",
            "tensor_data/tinyllama-1.1b/tokenizer.json",
            "tensor_data/tinyllama/tokenizer.json",
            "/home/akshay-bhalerao/tensor_data/tinyllama-1.1b/tokenizer.json",
        ];

        for path_str in &candidate_paths {
            let p = std::path::PathBuf::from(path_str);
            if p.exists() {
                tokenizer_src = p;
                break;
            }
        }

        if tokenizer_src.as_os_str().is_empty() {
            println!("Skipping midway fallback integration test: No tokenizer found.");
            return;
        }

        let temp_dir = std::env::temp_dir().join("bramha_fallback_mock_model");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();

        std::fs::copy(&tokenizer_src, temp_dir.join("tokenizer.json")).unwrap();

        let write_dummy_weight = |name: &str, size: usize| {
            let data = vec![0.0f32; size];
            let bytes = bytemuck::cast_slice(&data);
            let p = temp_dir.join(name.replace(".", "_") + ".bin");
            std::fs::write(&p, bytes).unwrap();
        };

        let vocab_size = 256;
        let hidden_size = 64;
        let head_dim = 16;
        let num_q_heads = 4;
        let num_kv_heads = 1;
        let mlp_size = 64;

        write_dummy_weight("model.embed_tokens.weight", vocab_size * hidden_size);
        write_dummy_weight("lm_head.weight", vocab_size * hidden_size);
        write_dummy_weight("model.norm.weight", hidden_size);
        write_dummy_weight("model.layers.0.input_layernorm.weight", hidden_size);
        write_dummy_weight(
            "model.layers.0.self_attn.q_proj.weight",
            (num_q_heads * head_dim) * hidden_size,
        );
        write_dummy_weight(
            "model.layers.0.self_attn.k_proj.weight",
            (num_kv_heads * head_dim) * hidden_size,
        );
        write_dummy_weight(
            "model.layers.0.self_attn.v_proj.weight",
            (num_kv_heads * head_dim) * hidden_size,
        );
        write_dummy_weight(
            "model.layers.0.self_attn.o_proj.weight",
            hidden_size * (num_q_heads * head_dim),
        );
        write_dummy_weight(
            "model.layers.0.post_attention_layernorm.weight",
            hidden_size,
        );
        write_dummy_weight(
            "model.layers.0.mlp.gate_proj.weight",
            mlp_size * hidden_size,
        );
        write_dummy_weight("model.layers.0.mlp.up_proj.weight", mlp_size * hidden_size);
        write_dummy_weight(
            "model.layers.0.mlp.down_proj.weight",
            hidden_size * mlp_size,
        );

        crate::storage::storage_manifest::write_mock_manifest(
            &temp_dir,
            "fallback-mock-model",
            vocab_size,
            hidden_size,
            num_q_heads,
            num_kv_heads,
            head_dim,
            mlp_size,
        );

        {
            let mut tensor_guard = db.tensor_db.write().await;
            tensor_guard.restore_model_at_path("fallback-mock-model".to_string(), &temp_dir);
        }

        // Enable simulated failure environment variable and bypass planner cache hits
        unsafe {
            std::env::set_var("BRAMHA_SIMULATE_GPU_FAILURE", "true");
        }
        unsafe {
            std::env::set_var("BRAMHA_PLANNER_MODE", "exact_only");
        }

        // Run generation which starts on WGPU and should failover midway to CPU transparently
        let result = InferenceEngine::new(None)
            .generate(db, "fallback-mock-model", "hi", 10, 0.0, None, None)
            .await;

        // Clean up environment and temp directory
        unsafe {
            std::env::remove_var("BRAMHA_SIMULATE_GPU_FAILURE");
        }
        unsafe {
            std::env::remove_var("BRAMHA_PLANNER_MODE");
        }
        let _ = std::fs::remove_dir_all(temp_dir);

        assert!(
            result.is_ok(),
            "Midway GPU failure did not fall back successfully: {:?}",
            result.err()
        );
        let info = result.unwrap();
        assert!(info.tokens_generated > 0);
        println!(
            "Seamless midway CPU fallback succeeded! Generated {} tokens",
            info.tokens_generated
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_planner_cache_hit_path() {
        let _guard = ENV_MUTEX.lock().unwrap();
        let temp_dir = std::env::temp_dir().join("bramha_hit_cache_test");
        let _ = std::fs::create_dir_all(&temp_dir);
        let test_cache_file = temp_dir.join("hit_answers.json");
        let _ = std::fs::remove_file(&test_cache_file);

        let mut db = Database::new(None, 1536);
        db.planner_cache_path = Some(test_cache_file.clone());
        let db_arc = Arc::new(db);

        // Pre-cache a deterministic reply
        let cache = crate::storage::answer_cache::DeterministicAnswerCache::load_from_path(
            &test_cache_file,
        );
        cache
            .insert(
                "cached query",
                "some-model",
                &[],
                "direct cache hit reply".to_string(),
            )
            .unwrap();

        // Query the engine. The planner should immediately select CachedAnswer and return without actual generation.
        let result = InferenceEngine::new(None)
            .generate(db_arc, "some-model", "cached query", 10, 0.0, None, None)
            .await;

        // Clean up
        let _ = std::fs::remove_file(&test_cache_file);
        let _ = std::fs::remove_dir_all(temp_dir);

        assert!(result.is_ok());
        let info = result.unwrap();
        assert_eq!(info.completion, "direct cache hit reply");
        assert_eq!(info.tokens_generated, 0); // 0 indicates returned directly from cache
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_planner_exact_only_override() {
        let _guard = ENV_MUTEX.lock().unwrap();
        let temp_dir = std::env::temp_dir().join("bramha_planner_exact_mock");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();

        let test_cache_file = temp_dir.join("exact_answers.json");
        let _ = std::fs::remove_file(&test_cache_file);

        let mut db = Database::new(None, 1536);
        db.planner_cache_path = Some(test_cache_file.clone());
        let db = Arc::new(db);

        // Mock a model inside the Database for generation
        let mut tokenizer_src = std::path::PathBuf::new();
        let candidate_paths = [
            "models/all-MiniLM-L6-v2/tokenizer.json",
            "tensor_data/tinyllama-1.1b/tokenizer.json",
            "tensor_data/tinyllama/tokenizer.json",
        ];
        for path_str in &candidate_paths {
            let p = std::path::PathBuf::from(path_str);
            if p.exists() {
                tokenizer_src = p;
                break;
            }
        }
        if tokenizer_src.as_os_str().is_empty() {
            println!("Skipping test: No tokenizer found.");
            return;
        }

        std::fs::copy(&tokenizer_src, temp_dir.join("tokenizer.json")).unwrap();

        let vocab_size = 256;
        let hidden_size = 64;
        let write_dummy_weight = |name: &str, size: usize| {
            let mut data = vec![1.0f32; size];
            if name == "lm_head.weight" {
                for d in 0..hidden_size {
                    data[100 * hidden_size + d] = 2.0;
                }
            }
            let bytes = bytemuck::cast_slice(&data);
            let p = temp_dir.join(name.replace(".", "_") + ".bin");
            std::fs::write(&p, bytes).unwrap();
        };

        write_dummy_weight("model.embed_tokens.weight", vocab_size * hidden_size);
        write_dummy_weight("lm_head.weight", vocab_size * hidden_size);
        write_dummy_weight("model.norm.weight", hidden_size);
        write_dummy_weight("model.layers.0.input_layernorm.weight", hidden_size);
        write_dummy_weight("model.layers.0.self_attn.q_proj.weight", 64 * 64);
        write_dummy_weight("model.layers.0.self_attn.k_proj.weight", 16 * 64);
        write_dummy_weight("model.layers.0.self_attn.v_proj.weight", 16 * 64);
        write_dummy_weight("model.layers.0.self_attn.o_proj.weight", 64 * 64);
        write_dummy_weight(
            "model.layers.0.post_attention_layernorm.weight",
            hidden_size,
        );
        write_dummy_weight("model.layers.0.mlp.gate_proj.weight", 64 * 64);
        write_dummy_weight("model.layers.0.mlp.up_proj.weight", 64 * 64);
        write_dummy_weight("model.layers.0.mlp.down_proj.weight", 64 * 64);

        crate::storage::storage_manifest::write_mock_manifest(
            &temp_dir,
            "planner-exact-mock-model",
            vocab_size,
            hidden_size,
            4,
            1,
            16,
            64,
        );

        {
            let mut tensor_guard = db.tensor_db.write().await;
            tensor_guard.restore_model_at_path("planner-exact-mock-model".to_string(), &temp_dir);
        }

        // Pre-cache deterministic reply
        let cache = crate::storage::answer_cache::DeterministicAnswerCache::load_from_path(
            &test_cache_file,
        );
        cache
            .insert(
                "query text",
                "planner-exact-mock-model",
                &[],
                "pre-cached text reply".to_string(),
            )
            .unwrap();

        // 1. Force exact-only mode via environment variable
        unsafe {
            std::env::set_var("BRAMHA_PLANNER_MODE", "exact_only");
        }

        // The planner should bypass the cache hit and execute actual generation
        let result = InferenceEngine::new(None)
            .generate(
                db,
                "planner-exact-mock-model",
                "query text",
                5,
                0.0,
                None,
                None,
            )
            .await;

        // Clean up
        unsafe {
            std::env::remove_var("BRAMHA_PLANNER_MODE");
        }
        let _ = std::fs::remove_file(&test_cache_file);
        let _ = std::fs::remove_dir_all(temp_dir);

        assert!(result.is_ok());
        let info = result.unwrap();
        assert_ne!(info.completion, "pre-cached text reply");
        assert!(info.tokens_generated > 0);
    }
}
