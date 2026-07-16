#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendTarget {
    Cpu,
    Gpu,
}

pub struct HeterogeneousScheduler {
    gpu_available: bool,
    force_cpu: bool,
}

impl HeterogeneousScheduler {
    pub fn new() -> Self {
        // Step 1: Identify if a valid wgpu adapter exists
        let gpu_available = crate::compute::wgpu_backend::get_wgpu_plane().is_some();
        let force_cpu = crate::inference::is_cpu_only();

        Self {
            gpu_available,
            force_cpu,
        }
    }

    /// Determines whether to route a specific operation to CPU or GPU based on tensor size and operation type.
    pub fn route_op(&self, size: usize, op_type: &str) -> BackendTarget {
        if self.force_cpu || !self.gpu_available {
            return BackendTarget::Cpu;
        }

        match op_type {
            // Embeddings lookups are kept on CPU to avoid PCIe host-to-device transfers
            "embedding" => BackendTarget::Cpu,
            // Sampling or tiny vector operations are extremely cheap on CPU
            "sampling" => BackendTarget::Cpu,
            // Matrix-Vector multiplications (GEMVs)
            "gemv" => {
                // If tensor size (in_features * out_features) is small (e.g. < 128KB of floats / 131,072 elements),
                // route to CPU to avoid device queue scheduling and PCIe transfer latency.
                if size < 131072 {
                    BackendTarget::Cpu
                } else {
                    BackendTarget::Gpu
                }
            }
            // General Matrix-Matrix multiplications (GEMMs)
            "gemm" => {
                // Large matrix multiplications are highly profitable on GPU
                if size >= 262144 {
                    BackendTarget::Gpu
                } else {
                    BackendTarget::Cpu
                }
            }
            _ => BackendTarget::Gpu,
        }
    }

    /// Decides if a full inference request should bypass the GPU entirely.
    pub async fn should_use_cpu_entirely(
        &self,
        db: &std::sync::Arc<crate::storage::Database>,
        model_name: &str,
    ) -> bool {
        if self.force_cpu || !self.gpu_available {
            return true;
        }

        // Get active WGPU VRAM cache capacity limit
        let max_vram_bytes = {
            let cache = crate::inference::engine::VramCache::global()
                .lock()
                .unwrap();
            cache.max_vram_bytes
        };

        if let Some(max_bytes) = max_vram_bytes {
            let tensor_db = db.tensor_db.read().await;
            if let Some(model) = tensor_db.models.get(model_name) {
                // Force CPU fallback if model target device is explicitly CPU
                if model.active_device.to_lowercase() == "cpu" {
                    return true;
                }

                // Sum all layer weight bytes to calculate exact memory footprint
                let total_model_bytes: usize = model
                    .layers
                    .values()
                    .map(|page| page.as_bytes().len())
                    .sum();

                // Route entirely to CPU if weight size exceeds active GPU VRAM cap to avoid trashing
                if total_model_bytes > max_bytes {
                    let logger = crate::inference::engine::InferenceLogger::global();
                    logger.record_log(format!(
                        "📋 [Scheduler] Model weight size ({:.2} MB) exceeds active GPU VRAM cap ({:.2} MB). Routing entirely to CPU engine to avoid thrashing.",
                        total_model_bytes as f64 / 1_000_000.0,
                        max_bytes as f64 / 1_000_000.0
                    ));
                    return true;
                }
            }
        }

        false
    }
}
