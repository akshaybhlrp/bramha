use crate::core::tensor::{DType, TensorPage};
use crate::storage::content_addressing::ContentAddressedStorage;
use crate::storage::model_view::ModelView;
use crate::storage::multi_tier::{MultiTierStorage, TierConfig};
use crate::storage::storage_manifest::ModelManifest;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
pub struct ModelTable {
    pub name: String,
    pub base_path: PathBuf,
    pub layers: HashMap<String, TensorPage>, // e.g. "layer_5_attention_weight" -> TensorPage
    pub early_exit_thresholds: Vec<f32>,
    pub active_device: String,
    pub num_experts: Option<usize>,
    pub expert_routing_top_k: Option<usize>,
    pub chunk_index: Option<crate::storage::indexing::BTreeIndex>,
    pub expert_map: Option<Vec<Vec<[Option<TensorPage>; 6]>>>,
    pub manifest: Option<ModelManifest>,
}

impl ModelTable {
    pub fn new(name: String, base_path: PathBuf) -> Self {
        ModelTable {
            name,
            base_path,
            layers: HashMap::new(),
            early_exit_thresholds: Vec::new(),
            active_device: "auto".to_string(),
            num_experts: None,
            expert_routing_top_k: None,
            chunk_index: None,
            expert_map: None,
            manifest: None,
        }
    }

    /// Simulates "inserting" a layer into the table by memory mapping a binary file on disk.
    pub fn load_layer(
        &mut self,
        layer_id: String,
        file_name: &str,
        shape: Vec<usize>,
        dtype: DType,
    ) -> std::io::Result<()> {
        let path = self.base_path.join(file_name);
        let page = TensorPage::load_mmap_single(layer_id.clone(), &path, shape, dtype)?;
        let _ = page.advise_prefetch();
        self.layers.insert(layer_id, page);
        Ok(())
    }

    /// Materialize a differential layer (AOT merge)
    pub fn materialize_differential(
        &mut self,
        layer_id: String,
        reference_id: &str,
        delta_file_name: &str,
        shape: Vec<usize>,
        dtype: DType,
    ) -> std::io::Result<()> {
        let path = self.base_path.join(delta_file_name);
        let delta_page =
            TensorPage::load_mmap_single(layer_id.clone() + "_delta", &path, shape.clone(), dtype)?;

        // Find reference page
        let ref_page = self.layers.get(reference_id).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Reference layer {} not found", reference_id),
            )
        })?;

        // Add them together (f32 only for now as requested)
        let delta_f32 = delta_page.as_f32();
        let ref_f32 = ref_page.as_f32();

        let mut buffer = vec![0.0f32; delta_f32.len()];
        for i in 0..buffer.len() {
            buffer[i] = ref_f32[i] + delta_f32[i];
        }

        let u8_buffer = bytemuck::cast_slice(&buffer).to_vec();

        let materialized_page = TensorPage::new_memory(layer_id.clone(), shape, dtype, u8_buffer);
        self.layers.insert(layer_id, materialized_page);
        Ok(())
    }

    /// Advise the OS page cache that all sharded weights for the given layer_idx are no longer needed in RAM.
    pub fn advise_dont_need_for_layer(&self, layer_idx: usize) {
        let keys = [
            format!("model.layers.{}.input_layernorm.weight", layer_idx),
            format!("model.layers.{}.self_attn.q_proj.weight", layer_idx),
            format!("model.layers.{}.self_attn.k_proj.weight", layer_idx),
            format!("model.layers.{}.self_attn.v_proj.weight", layer_idx),
            format!("model.layers.{}.self_attn.o_proj.weight", layer_idx),
            format!("model.layers.{}.self_attn.q_proj.bias", layer_idx),
            format!("model.layers.{}.self_attn.k_proj.bias", layer_idx),
            format!("model.layers.{}.self_attn.v_proj.bias", layer_idx),
            format!("model.layers.{}.self_attn.o_proj.bias", layer_idx),
            format!("model.layers.{}.post_attention_layernorm.weight", layer_idx),
            format!("model.layers.{}.mlp.gate_proj.weight", layer_idx),
            format!("model.layers.{}.mlp.up_proj.weight", layer_idx),
            format!("model.layers.{}.mlp.down_proj.weight", layer_idx),
        ];

        for key in &keys {
            if let Some(page) = self.layers.get(key) {
                let _ = page.dont_need();
            }
        }
    }

    /// Advise the OS page cache that non-layer specific model weights are no longer needed.
    pub fn advise_dont_need_non_layers(&self) {
        let keys = [
            "model.embed_tokens.weight".to_string(),
            "model.norm.weight".to_string(),
            "lm_head.weight".to_string(),
        ];
        for key in &keys {
            if let Some(page) = self.layers.get(key) {
                let _ = page.dont_need();
            }
        }
    }

    /// Shards an entire safetensors file into this model table mapping directly to F32
    pub fn load_safetensors(&mut self, file_name: &str) -> Result<(), Box<dyn std::error::Error>> {
        let path = self.base_path.join(file_name);
        crate::storage::safetensors_loader::shard_safetensors_file(self, &path)
    }

    /// Fetches a layer's raw bytes in O(1) time using the OS page cache
    pub fn fetch_layer(&self, layer_id: &str) -> Option<&[u8]> {
        self.layers.get(layer_id).map(|page| page.as_bytes())
    }

    /// Loads the chunks of a single tensor (and its scale tensor if applicable) from the block database using the B-Tree index if it is currently empty.
    pub fn load_tensor_chunks(
        &mut self,
        tensor_name: &str,
        block_db: &mut crate::storage::block_db::BlockDB,
    ) -> std::io::Result<()> {
        self.load_tensor_chunks_internal(tensor_name, block_db)?;
        let scale_name = format!("{}.scale", tensor_name);
        self.load_tensor_chunks_internal(&scale_name, block_db)?;
        Ok(())
    }

    fn load_tensor_chunks_internal(
        &mut self,
        tensor_name: &str,
        block_db: &mut crate::storage::block_db::BlockDB,
    ) -> std::io::Result<()> {
        if let Some(page) = self.layers.get(tensor_name) {
            if !page.as_bytes().is_empty() {
                return Ok(()); // Already loaded
            }
        } else {
            return Ok(()); // If tensor doesn't exist, skip
        }

        let chunk_index = match &self.chunk_index {
            Some(idx) => idx,
            None => return Ok(()), // If no chunk index, treat as already loaded/legacy
        };

        let prefix = format!("{}:", tensor_name);
        let mut hashes_with_idx = Vec::new();
        for (key, hash) in chunk_index.prefix_scan_with_keys(&prefix) {
            if let crate::storage::indexing::BTreeKey::String(s) = key {
                // Key format is "model.layers.0.input_layernorm.weight:N"
                if let Some(idx_str) = s.split(':').next_back()
                    && let Ok(idx) = idx_str.parse::<usize>()
                {
                    hashes_with_idx.push((idx, hash.clone()));
                }
            }
        }

        if hashes_with_idx.is_empty() {
            return Ok(());
        }

        // Sort chunks correctly by their index before concatenation!
        hashes_with_idx.sort_by_key(|(idx, _)| *idx);

        // Collect locations first
        let mut locations = Vec::new();
        for (_idx, hash) in &hashes_with_idx {
            if let Some(loc) = block_db.get_block_location(hash) {
                locations.push(loc);
            } else {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("Block hash {} location not found in BlockDB", hash),
                ));
            }
        }

        // Check if contiguous
        let mut is_contiguous = true;
        if !locations.is_empty() && locations[0].offset % 4 != 0 {
            is_contiguous = false;
        } else {
            for i in 0..locations.len().saturating_sub(1) {
                if locations[i].offset + locations[i].length as u64 != locations[i + 1].offset {
                    is_contiguous = false;
                    break;
                }
            }
        }

        if is_contiguous
            && !locations.is_empty()
            && let Some(mmap_data) = &block_db.mmap
        {
            let start = locations[0].offset as usize;
            let end = (locations.last().unwrap().offset + locations.last().unwrap().length as u64)
                as usize;

            if let Some(page) = self.layers.get_mut(tensor_name) {
                *page = TensorPage::new_slice(
                    tensor_name.to_string(),
                    mmap_data.clone(),
                    page.shape.clone(),
                    page.dtype,
                    start,
                    end,
                );
            }
            return Ok(());
        }

        // Fallback to allocating buffer if non-contiguous or mmap unavailable
        // We MUST ensure 4-byte alignment for bytemuck casts later!
        let total_bytes: usize = locations.iter().map(|l| l.length).sum();
        let num_u32s = total_bytes.div_ceil(4);
        let mut aligned_buffer: Vec<u32> = vec![0; num_u32s];

        let mut current_offset = 0;
        for loc in locations {
            let data = block_db.read_block(&loc)?;
            // Copy into aligned buffer safely
            let target_slice = bytemuck::cast_slice_mut::<u32, u8>(&mut aligned_buffer);
            target_slice[current_offset..current_offset + data.len()].copy_from_slice(&data);
            current_offset += data.len();
        }

        // Convert the properly aligned Vec<u32> back to Vec<u8> safely
        let ptr = aligned_buffer.as_mut_ptr() as *mut u8;
        let len = aligned_buffer.len() * 4;
        let cap = aligned_buffer.capacity() * 4;
        std::mem::forget(aligned_buffer);
        // SAFETY: Manual invariants verified for performance/FFI
        let mut buffer = unsafe { Vec::from_raw_parts(ptr, len, cap) };
        buffer.truncate(total_bytes);

        // Replace the page with a memory-backed page containing the loaded bytes
        if let Some(page) = self.layers.get_mut(tensor_name) {
            *page = TensorPage::new_memory(
                tensor_name.to_string(),
                page.shape.clone(),
                page.dtype,
                buffer,
            );
        }

        Ok(())
    }

    /// Unloads a tensor's data buffer to save RAM.
    pub fn unload_tensor_chunks(&mut self, tensor_name: &str) {
        self.unload_tensor_chunks_internal(tensor_name);
        let scale_name = format!("{}.scale", tensor_name);
        self.unload_tensor_chunks_internal(&scale_name);
    }

    fn unload_tensor_chunks_internal(&mut self, tensor_name: &str) {
        if self.chunk_index.is_none() {
            // Legacy models use Mmap directly without chunks. We cannot reload them, so do not unload them.
            return;
        }
        if let Some(page) = self.layers.get_mut(tensor_name) {
            // Free the memory block if it was populated via chunks, preserving the empty page for future reloads
            if !page.as_bytes().is_empty() {
                // If it's a Memory backend, we can just replace it with an empty Vec
                // We cannot safely drop Mmap backends if they are actively used, but Virtual View uses Memory backend.
                *page = TensorPage::new_memory(
                    page.name.clone(),
                    page.shape.clone(),
                    page.dtype,
                    Vec::new(),
                );
            }
        }
    }

    pub fn load_transformer_layer_chunks(
        &mut self,
        layer_idx: usize,
        block_db: &mut crate::storage::block_db::BlockDB,
    ) -> std::io::Result<()> {
        let components = [
            "input_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_proj.bias",
            "self_attn.k_proj.bias",
            "self_attn.v_proj.bias",
            "self_attn.o_proj.bias",
            "post_attention_layernorm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ];
        for comp in &components {
            let key = format!("model.layers.{}.{}", layer_idx, comp);
            let _ = self.load_tensor_chunks(&key, block_db);
        }
        Ok(())
    }

    pub fn unload_transformer_layer_chunks(&mut self, layer_idx: usize) {
        let components = [
            "input_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_proj.bias",
            "self_attn.k_proj.bias",
            "self_attn.v_proj.bias",
            "self_attn.o_proj.bias",
            "post_attention_layernorm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ];
        for comp in &components {
            let key = format!("model.layers.{}.{}", layer_idx, comp);
            self.unload_tensor_chunks(&key);
        }
    }

    /// Loads all base tensors required for executing layer_idx.
    pub fn load_layer_tensors(
        &mut self,
        layer_idx: usize,
        block_db: &mut crate::storage::block_db::BlockDB,
    ) -> std::io::Result<()> {
        let keys = [
            format!("model.layers.{}.input_layernorm.weight", layer_idx),
            format!("model.layers.{}.self_attn.q_proj.weight", layer_idx),
            format!("model.layers.{}.self_attn.k_proj.weight", layer_idx),
            format!("model.layers.{}.self_attn.v_proj.weight", layer_idx),
            format!("model.layers.{}.self_attn.o_proj.weight", layer_idx),
            format!("model.layers.{}.self_attn.q_proj.bias", layer_idx),
            format!("model.layers.{}.self_attn.k_proj.bias", layer_idx),
            format!("model.layers.{}.self_attn.v_proj.bias", layer_idx),
            format!("model.layers.{}.self_attn.o_proj.bias", layer_idx),
            format!("model.layers.{}.post_attention_layernorm.weight", layer_idx),
            format!("model.layers.{}.mlp.gate_proj.weight", layer_idx),
            format!("model.layers.{}.mlp.up_proj.weight", layer_idx),
            format!("model.layers.{}.mlp.down_proj.weight", layer_idx),
            format!("model.layers.{}.mlp.router.weight", layer_idx),
        ];

        for key in &keys {
            self.load_tensor_chunks(key, block_db)?;
        }
        Ok(())
    }

    /// Unloads all tensors for layer_idx.
    pub fn unload_layer_tensors(&mut self, layer_idx: usize) {
        let keys = [
            format!("model.layers.{}.input_layernorm.weight", layer_idx),
            format!("model.layers.{}.self_attn.q_proj.weight", layer_idx),
            format!("model.layers.{}.self_attn.k_proj.weight", layer_idx),
            format!("model.layers.{}.self_attn.v_proj.weight", layer_idx),
            format!("model.layers.{}.self_attn.o_proj.weight", layer_idx),
            format!("model.layers.{}.self_attn.q_proj.bias", layer_idx),
            format!("model.layers.{}.self_attn.k_proj.bias", layer_idx),
            format!("model.layers.{}.self_attn.v_proj.bias", layer_idx),
            format!("model.layers.{}.self_attn.o_proj.bias", layer_idx),
            format!("model.layers.{}.post_attention_layernorm.weight", layer_idx),
            format!("model.layers.{}.mlp.gate_proj.weight", layer_idx),
            format!("model.layers.{}.mlp.up_proj.weight", layer_idx),
            format!("model.layers.{}.mlp.down_proj.weight", layer_idx),
            format!("model.layers.{}.mlp.router.weight", layer_idx),
        ];

        for key in &keys {
            self.unload_tensor_chunks(key);
        }

        // Also unload any expert tensors if present for this layer
        if let Some(num_experts) = self.num_experts {
            for e in 0..num_experts {
                self.unload_tensor_chunks(&format!(
                    "model.layers.{}.mlp.experts.{}.gate_proj.weight",
                    layer_idx, e
                ));
                self.unload_tensor_chunks(&format!(
                    "model.layers.{}.mlp.experts.{}.up_proj.weight",
                    layer_idx, e
                ));
                self.unload_tensor_chunks(&format!(
                    "model.layers.{}.mlp.experts.{}.down_proj.weight",
                    layer_idx, e
                ));
            }
        }
    }

    /// Loads non-layer model weights (embeddings, final norms, lm_head).
    pub fn load_non_layer_tensors(
        &mut self,
        block_db: &mut crate::storage::block_db::BlockDB,
    ) -> std::io::Result<()> {
        let keys = [
            "model.embed_tokens.weight".to_string(),
            "model.norm.weight".to_string(),
            "lm_head.weight".to_string(),
        ];
        for key in &keys {
            self.load_tensor_chunks(key, block_db)?;
        }
        Ok(())
    }

    /// Unloads non-layer model weights.
    pub fn unload_non_layer_tensors(&mut self) {
        let keys = [
            "model.embed_tokens.weight".to_string(),
            "model.norm.weight".to_string(),
            "lm_head.weight".to_string(),
        ];
        for key in &keys {
            self.unload_tensor_chunks(key);
        }
    }
}

pub struct TensorDB {
    pub models: HashMap<String, ModelTable>,
    pub storage_dir: PathBuf,
    pub multi_tier: std::sync::Mutex<MultiTierStorage>,
    pub content_storage: std::sync::Mutex<ContentAddressedStorage>,
    pub block_db: std::sync::Mutex<crate::storage::block_db::BlockDB>,
}

impl TensorDB {
    pub fn new(storage_dir: PathBuf) -> Self {
        std::fs::create_dir_all(&storage_dir).unwrap_or_default();

        let hot_path = storage_dir.join("hot");
        let warm_path = storage_dir.join("warm");
        let cold_path = storage_dir.join("cold");
        let content_dir = storage_dir.join("content");

        let config = TierConfig::default();
        let multi_tier = MultiTierStorage::new(config, hot_path, warm_path, cold_path).unwrap();
        let content_storage = ContentAddressedStorage::new(content_dir.clone()).unwrap();
        let block_db = crate::storage::block_db::BlockDB::new(content_dir).unwrap();

        let mut db = TensorDB {
            models: HashMap::new(),
            storage_dir: storage_dir.clone(),
            multi_tier: std::sync::Mutex::new(multi_tier),
            content_storage: std::sync::Mutex::new(content_storage),
            block_db: std::sync::Mutex::new(block_db),
        };
        // Auto-restore any previously sharded models from disk
        db.restore_from_disk();
        db
    }

    pub fn create_model(&mut self, name: String) {
        let path = self.storage_dir.join(&name);
        std::fs::create_dir_all(&path).unwrap_or_default();
        self.models
            .insert(name.clone(), ModelTable::new(name, path));
    }

    /// Scans the storage directory for model subdirectories containing a manifest.json,
    /// and re-maps all previously sharded layers back into memory.
    fn restore_from_disk(&mut self) {
        let entries = match std::fs::read_dir(&self.storage_dir) {
            Ok(e) => e,
            Err(_) => return,
        };

        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }

            let model_name = match path.file_name().and_then(|n| n.to_str()) {
                Some(n) => n.to_string(),
                None => continue,
            };

            if model_name == "hot"
                || model_name == "warm"
                || model_name == "cold"
                || model_name == "content"
            {
                continue;
            }

            self.restore_model_at_path(model_name, &path);
        }
    }

    /// Re-maps a single model's layers from a specific filesystem path into memory.
    pub fn restore_model_at_path(&mut self, model_name: String, path: &Path) {
        if self.models.contains_key(&model_name) {
            return;
        }

        let manifest_path = path.join("manifest.json");

        let mut table = ModelTable::new(model_name.clone(), path.to_path_buf());
        table.early_exit_thresholds = Vec::new();

        if manifest_path.exists()
            && let Ok(manifest_data) = std::fs::read_to_string(&manifest_path)
        {
            if let Ok(manifest) = serde_json::from_str::<ModelManifest>(&manifest_data) {
                table.num_experts = manifest.num_experts;
                table.expert_routing_top_k = manifest.expert_routing_top_k;
                table.manifest = Some(manifest);
            } else {
                eprintln!("⚠️ Failed to parse manifest.json for model {}", model_name);
            }
        }

        println!(
            "🔄 Registered model '{}' from disk (lazy loading enabled)",
            model_name
        );
        self.models.insert(model_name, table);
    }

    /// Loads the actual layer pages of a model into memory on demand.
    pub fn load_model_layers(&mut self, model_name: &str) -> Result<(), String> {
        let (path, _, _) = {
            let model = self
                .models
                .get(model_name)
                .ok_or_else(|| format!("Model {} not registered in TensorDB", model_name))?;
            (
                model.base_path.clone(),
                model.num_experts,
                model.expert_routing_top_k,
            )
        };

        // Support bypass via manifest_load feature and BRAMHA_MANIFEST_LOAD env variable
        let manifest_load_feature = cfg!(feature = "manifest_load");
        let manifest_load_env = std::env::var("BRAMHA_MANIFEST_LOAD")
            .map(|v| v.trim().to_lowercase() != "false")
            .unwrap_or(true);
        let maybe_manifest = if manifest_load_feature && manifest_load_env {
            self.models.get(model_name).unwrap().manifest.clone()
        } else {
            None
        };

        let _manifest_path = path.join("manifest.json");
        let view_path = path.join("model_view.json");

        let mut layers = HashMap::new();
        let mut chunk_index = None;

        if view_path.exists() {
            // --- Virtual View Restoration (BUTS) ---
            let view = ModelView::load(&view_path)
                .map_err(|e| format!("Failed to load model view at {:?}: {}", view_path, e))?;

            // Create B-Tree index for chunk mapping
            let mut idx = crate::storage::indexing::BTreeIndex::new(
                format!("{}_chunks", model_name),
                "chunk_key",
                true,
            );

            let mut restored_count = 0;
            for (_, virtual_tensor) in view.tensors.into_iter() {
                let dtype =
                    crate::storage::safetensors_loader::string_to_dtype(&virtual_tensor.dtype);
                // Create an empty placeholder page
                let page = TensorPage::new_memory(
                    virtual_tensor.name.clone(),
                    virtual_tensor.shape,
                    dtype,
                    Vec::new(), // Starts empty to save DRAM
                );
                layers.insert(virtual_tensor.name.clone(), page);

                // Index each chunk using B-Tree
                for (chunk_idx, hash) in virtual_tensor.block_hashes.iter().enumerate() {
                    let key = crate::storage::indexing::BTreeKey::String(format!(
                        "{}:{:06}",
                        virtual_tensor.name, chunk_idx
                    ));
                    let _ = idx.insert(key, hash.clone());
                }
                restored_count += 1;
            }

            chunk_index = Some(idx);
            println!(
                "🔄 Loaded model '{}' from BUTS Virtual View ({} layers indexed via B-Tree)",
                model_name, restored_count
            );
        } else if let Some(manifest) = maybe_manifest {
            // --- Legacy Manifest-based restoration (full metadata) ---
            let mut restored_count = 0;

            for layer_meta in manifest.layers.values() {
                let safe_name = layer_meta.layer_id.replace(".", "_");
                use crate::storage::storage_manifest::CompressionFormat;
                let candidates: Vec<String> = match layer_meta.compression_format {
                    CompressionFormat::Int4PerChannel => vec![
                        format!("{}_u4.bin", safe_name),
                        format!("{}_scale.bin", safe_name),
                        format!("{}.bin", safe_name),
                    ],
                    CompressionFormat::Int8Linear => vec![
                        format!("{}_i8.bin", safe_name),
                        format!("{}_scale.bin", safe_name),
                        format!("{}.bin", safe_name),
                    ],
                    CompressionFormat::Svd => vec![
                        format!("{}_svd.bin", safe_name),
                        format!("{}.bin", safe_name),
                    ],
                    CompressionFormat::ColumnarDict => vec![
                        format!("{}_cd.bin", safe_name),
                        format!("{}.bin", safe_name),
                    ],
                    _ => vec![format!("{}.bin", safe_name)],
                };

                let bin_path = candidates.iter().map(|c| path.join(c)).find(|p| p.exists());

                let bin_path = match bin_path {
                    Some(p) => p,
                    None => {
                        eprintln!(
                            "⚠️ Missing shard file for layer '{}' (tried: {}), skipping.",
                            layer_meta.layer_id,
                            candidates.join(", ")
                        );
                        continue;
                    }
                };

                if !layer_meta.checksum.is_empty() {
                    use sha2::{Digest, Sha256};
                    if let Ok(bytes) = std::fs::read(&bin_path) {
                        let mut hasher = Sha256::new();
                        hasher.update(&bytes);
                        let hash_result = hasher.finalize();
                        let sha256_hex = format!("{:x}", hash_result);
                        if sha256_hex != layer_meta.checksum {
                            panic!(
                                "CRITICAL SECURITY ERROR: Checksum mismatch for tensor shard {:?}. Expected: {}, Actual: {}",
                                bin_path, layer_meta.checksum, sha256_hex
                            );
                        }
                    }
                }

                let dtype = match layer_meta.compression_format {
                    crate::storage::storage_manifest::CompressionFormat::Svd => DType::Svd,
                    crate::storage::storage_manifest::CompressionFormat::ColumnarDict => {
                        DType::ColumnarDict
                    }
                    crate::storage::storage_manifest::CompressionFormat::Int4PerChannel => {
                        DType::U4
                    }
                    _ => DType::F32,
                };
                match TensorPage::load_mmap_single(
                    layer_meta.layer_id.clone(),
                    &bin_path,
                    layer_meta.shape.clone(),
                    dtype,
                ) {
                    Ok(mut page) => {
                        if dtype == DType::Svd {
                            page.svd_rank = layer_meta.svd_rank;
                        }
                        let _ = page.advise_prefetch();

                        layers.insert(layer_meta.layer_id.clone(), page);
                        if let Ok(mut mt) = self.multi_tier.lock() {
                            let _ = mt.register_layer(
                                layer_meta.layer_id.clone(),
                                layer_meta.stored_bytes,
                                layer_meta.storage_tier,
                                bin_path,
                            );
                        }
                        restored_count += 1;
                    }
                    Err(e) => {
                        eprintln!("⚠️ Failed to mmap layer {}: {}", layer_meta.layer_id, e);
                    }
                }
            }

            println!(
                "🔄 Loaded model '{}' from legacy manifest ({} layers)",
                model_name, restored_count
            );
        } else {
            // --- Legacy fallback: scan for .bin files without manifest ---
            let bin_files: Vec<_> = match std::fs::read_dir(&path) {
                Ok(rd) => rd
                    .flatten()
                    .filter(|e| e.path().extension().is_some_and(|ext| ext == "bin"))
                    .collect(),
                Err(_) => Vec::new(),
            };

            let mut restored_count = 0;

            for bin_entry in &bin_files {
                let bin_path = bin_entry.path();
                let layer_name = bin_path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown")
                    .replace("_", ".");

                match TensorPage::load_mmap_single(
                    layer_name.clone(),
                    &bin_path,
                    vec![],
                    DType::Other,
                ) {
                    Ok(page) => {
                        let _ = page.advise_prefetch();

                        layers.insert(layer_name, page);
                        restored_count += 1;
                    }
                    Err(e) => {
                        eprintln!("⚠️ Failed to mmap legacy shard {:?}: {}", bin_path, e);
                    }
                }
            }

            println!(
                "🔄 Loaded legacy model '{}' from disk ({} shards, no manifest)",
                model_name, restored_count
            );
        }

        if let Some(model) = self.models.get_mut(model_name) {
            model.layers = layers;
            if chunk_index.is_some() {
                model.chunk_index = chunk_index;
            }

            // Hydrate expert map if it's an MoE model
            if let (Some(num_experts), Some(chunk_index)) = (model.num_experts, &model.chunk_index)
                && num_experts > 0
            {
                let num_layers = model
                    .layers
                    .keys()
                    .filter(|k| {
                        k.starts_with("model.layers.") && k.ends_with(".input_layernorm.weight")
                    })
                    .count();

                let mut expert_map: Vec<Vec<[Option<TensorPage>; 6]>> =
                    Vec::with_capacity(num_layers);
                for _ in 0..num_layers {
                    let mut layer_experts = Vec::with_capacity(num_experts);
                    for _ in 0..num_experts {
                        layer_experts.push([None, None, None, None, None, None]);
                    }
                    expert_map.push(layer_experts);
                }
                let block_db = self.block_db.lock().unwrap();

                for layer_idx in 0..num_layers {
                    for expert_idx in 0..num_experts {
                        let keys = [
                            format!(
                                "model.layers.{}.mlp.experts.{}.gate_proj.weight",
                                layer_idx, expert_idx
                            ),
                            format!(
                                "model.layers.{}.mlp.experts.{}.gate_proj.weight.scale",
                                layer_idx, expert_idx
                            ),
                            format!(
                                "model.layers.{}.mlp.experts.{}.up_proj.weight",
                                layer_idx, expert_idx
                            ),
                            format!(
                                "model.layers.{}.mlp.experts.{}.up_proj.weight.scale",
                                layer_idx, expert_idx
                            ),
                            format!(
                                "model.layers.{}.mlp.experts.{}.down_proj.weight",
                                layer_idx, expert_idx
                            ),
                            format!(
                                "model.layers.{}.mlp.experts.{}.down_proj.weight.scale",
                                layer_idx, expert_idx
                            ),
                        ];
                        for (t_idx, key) in keys.iter().enumerate() {
                            let prefix = format!("{}:", key);
                            let mut hashes_with_idx = Vec::new();
                            for (b_key, hash) in chunk_index.prefix_scan_with_keys(&prefix) {
                                if let crate::storage::indexing::BTreeKey::String(s) = b_key
                                    && let Some(idx_str) = s.split(':').next_back()
                                    && let Ok(idx) = idx_str.parse::<usize>()
                                {
                                    hashes_with_idx.push((idx, hash.clone()));
                                }
                            }
                            hashes_with_idx.sort_by_key(|(idx, _)| *idx);
                            let mut locations = Vec::new();
                            for (_idx, hash) in hashes_with_idx {
                                if let Some(loc) = block_db.get_block_location(&hash) {
                                    locations.push(loc);
                                }
                            }
                            let mut is_aligned = true;
                            if locations.is_empty() {
                                is_aligned = false;
                            } else if locations[0].offset % 4 != 0 {
                                is_aligned = false;
                            } else {
                                for i in 0..locations.len().saturating_sub(1) {
                                    if locations[i].offset + locations[i].length as u64
                                        != locations[i + 1].offset
                                    {
                                        is_aligned = false;
                                        break;
                                    }
                                }
                            }

                            if is_aligned
                                && let Some(template) = model.layers.get(key)
                                && let Some(mmap_data) = &block_db.mmap
                            {
                                expert_map[layer_idx][expert_idx][t_idx] =
                                    Some(TensorPage::new_slice(
                                        key.clone(),
                                        mmap_data.clone(),
                                        template.shape.clone(),
                                        template.dtype,
                                        locations[0].offset as usize,
                                        (locations.last().unwrap().offset
                                            + locations.last().unwrap().length as u64)
                                            as usize,
                                    ));
                            }
                        }
                    }
                }
                model.expert_map = Some(expert_map);
                println!(
                    "🔄 Hydrated Static MoE Map for {} layers, {} experts each",
                    num_layers, num_experts
                );
            }
        }

        Ok(())
    }

    /// Unloads layer pages of a model from memory to reclaim RAM.
    pub fn unload_model_layers(&mut self, model_name: &str) {
        if let Some(model) = self.models.get_mut(model_name) {
            model.layers.clear();
            model.layers.shrink_to_fit();
            println!("🔄 Unloaded model '{}' layers from memory", model_name);
        }
    }

    /// Unloads layers of a model only if it is a virtual view (BUTS) model using memory-backed pages.
    pub fn unload_model_if_virtual(&mut self, model_name: &str) {
        let is_virtual = if let Some(model) = self.models.get(model_name) {
            model.base_path.join("model_view.json").exists()
        } else {
            false
        };
        if is_virtual {
            self.unload_model_layers(model_name);
        }
    }

    /// Ensures the layers for the model are loaded into memory.
    pub fn ensure_model_loaded(&mut self, model_name: &str) -> Result<(), String> {
        let needs_load = if let Some(model) = self.models.get(model_name) {
            model.layers.is_empty()
        } else {
            false
        };
        if needs_load {
            self.load_model_layers(model_name)?;
        }
        Ok(())
    }

    /// Ensures that only a specific tensor is loaded into memory, avoiding a full model load if possible.
    pub fn ensure_tensor_loaded(&mut self, model_name: &str, layer_id: &str) -> Result<(), String> {
        // If the model is fully loaded, or the layer is already loaded, we're good
        if let Some(model) = self.models.get(model_name) {
            if model.layers.contains_key(layer_id) {
                return Ok(());
            }
        } else {
            return Err(format!("Model {} not registered in TensorDB", model_name));
        }

        let (path, manifest) = {
            let model = self.models.get(model_name).unwrap();
            (model.base_path.clone(), model.manifest.clone())
        };

        let view_path = path.join("model_view.json");
        if view_path.exists() {
            // For BUTS models, the metadata is monolithic in model_view.json, so we have to call load_model_layers,
            // but load_model_layers for BUTS only creates empty placeholders (no mmaps), so it's already fast.
            self.ensure_model_loaded(model_name)?;
            return Ok(());
        }

        // True per-tensor on-demand load for legacy/manifest models
        if let Some(manifest) = manifest {
            if let Some(layer_meta) = manifest
                .layers
                .get(layer_id)
                .or_else(|| manifest.layers.values().find(|v| v.layer_id == layer_id))
            {
                let safe_name = layer_meta.layer_id.replace(".", "_");
                let candidates = vec![
                    format!("{}_u4.bin", safe_name),
                    format!("{}_scale.bin", safe_name),
                    format!("{}_i8.bin", safe_name),
                    format!("{}_svd.bin", safe_name),
                    format!("{}_cd.bin", safe_name),
                    format!("{}.bin", safe_name),
                ];
                let bin_path = candidates.iter().map(|c| path.join(c)).find(|p| p.exists());
                if let Some(bin_path) = bin_path {
                    let dtype = match layer_meta.compression_format {
                        crate::storage::storage_manifest::CompressionFormat::Svd => DType::Svd,
                        crate::storage::storage_manifest::CompressionFormat::ColumnarDict => {
                            DType::ColumnarDict
                        }
                        crate::storage::storage_manifest::CompressionFormat::Int4PerChannel => {
                            DType::U4
                        }
                        _ => DType::F32,
                    };
                    match TensorPage::load_mmap_single(
                        layer_meta.layer_id.clone(),
                        &bin_path,
                        layer_meta.shape.clone(),
                        dtype,
                    ) {
                        Ok(mut page) => {
                            if dtype == DType::Svd {
                                page.svd_rank = layer_meta.svd_rank;
                            }
                            let _ = page.advise_prefetch();
                            if let Some(model) = self.models.get_mut(model_name) {
                                model.layers.insert(layer_meta.layer_id.clone(), page);
                            }
                            return Ok(());
                        }
                        Err(e) => {
                            return Err(format!(
                                "Failed to mmap layer {}: {}",
                                layer_meta.layer_id, e
                            ));
                        }
                    }
                } else {
                    return Err(format!("Missing shard file for layer '{}'", layer_id));
                }
            } else {
                return Err(format!("Layer {} not found in manifest", layer_id));
            }
        } else {
            // Legacy fallback (no manifest)
            let safe_name = layer_id.replace(".", "_");
            let bin_path = path.join(format!("{}.bin", safe_name));
            if bin_path.exists() {
                match TensorPage::load_mmap_single(
                    layer_id.to_string(),
                    &bin_path,
                    vec![],
                    DType::Other,
                ) {
                    Ok(page) => {
                        let _ = page.advise_prefetch();
                        if let Some(model) = self.models.get_mut(model_name) {
                            model.layers.insert(layer_id.to_string(), page);
                        }
                        return Ok(());
                    }
                    Err(e) => {
                        return Err(format!("Failed to mmap legacy shard {:?}: {}", bin_path, e));
                    }
                }
            } else {
                return Err(format!("Layer file {:?} not found", bin_path));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tensor::DType;
    use tempfile::TempDir;

    #[test]
    fn test_model_table_creation() {
        let temp_dir = TempDir::new().unwrap();
        let table = ModelTable::new("tiny-llama".to_string(), temp_dir.path().to_path_buf());
        assert_eq!(table.name, "tiny-llama");
        assert_eq!(table.base_path, temp_dir.path().to_path_buf());
        assert!(table.layers.is_empty());
        assert_eq!(table.active_device, "auto");
    }

    #[test]
    fn test_model_table_materialize_differential() {
        let temp_dir = TempDir::new().unwrap();
        let mut table = ModelTable::new("diff-model".to_string(), temp_dir.path().to_path_buf());

        // 1. Create reference layer in table
        let ref_data = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32];
        let ref_bytes = bytemuck::cast_slice(&ref_data).to_vec();
        let ref_page =
            TensorPage::new_memory("layer_0".to_string(), vec![2, 2], DType::F32, ref_bytes);
        table.layers.insert("layer_0".to_string(), ref_page);

        // 2. Create delta file on disk
        let delta_data = vec![0.5f32, -1.0f32, 2.0f32, 10.0f32];
        let delta_bytes = bytemuck::cast_slice(&delta_data).to_vec();
        let delta_file = "layer_0_delta.bin";
        std::fs::write(temp_dir.path().join(delta_file), delta_bytes).unwrap();

        // 3. Materialize differential
        table
            .materialize_differential(
                "layer_0_materialized".to_string(),
                "layer_0",
                delta_file,
                vec![2, 2],
                DType::F32,
            )
            .unwrap();

        // 4. Verify output page (sum of ref + delta)
        assert!(table.layers.contains_key("layer_0_materialized"));
        let page = table.layers.get("layer_0_materialized").unwrap();
        let result_f32 = page.as_f32();

        assert_eq!(result_f32.len(), 4);
        assert_eq!(result_f32[0], 1.5);
        assert_eq!(result_f32[1], 1.0);
        assert_eq!(result_f32[2], 5.0);
        assert_eq!(result_f32[3], 14.0);
    }

    #[test]
    fn test_advise_dont_need() {
        let temp_dir = TempDir::new().unwrap();
        let mut table = ModelTable::new(
            "model-cache-test".to_string(),
            temp_dir.path().to_path_buf(),
        );

        // Insert a couple dummy pages
        let page1 = TensorPage::new_memory(
            "model.layers.0.input_layernorm.weight".to_string(),
            vec![1],
            DType::F32,
            vec![0; 4],
        );
        let page2 = TensorPage::new_memory(
            "model.embed_tokens.weight".to_string(),
            vec![1],
            DType::F32,
            vec![0; 4],
        );
        table
            .layers
            .insert("model.layers.0.input_layernorm.weight".to_string(), page1);
        table
            .layers
            .insert("model.embed_tokens.weight".to_string(), page2);

        // Call advise functions (should run and not panic)
        table.advise_dont_need_for_layer(0);
        table.advise_dont_need_non_layers();
    }

    #[test]
    fn test_tensor_db_load_unload_lifecycle() {
        let temp_dir = TempDir::new().unwrap();
        let mut db = TensorDB::new(temp_dir.path().to_path_buf());

        // Register a dummy model
        db.restore_model_at_path("llama-dummy".to_string(), temp_dir.path());
        assert!(db.models.contains_key("llama-dummy"));

        // By default, the model is registered but has empty layers
        let model = db.models.get("llama-dummy").unwrap();
        assert!(model.layers.is_empty());

        // Populate with a test layer
        {
            let model_mut = db.models.get_mut("llama-dummy").unwrap();
            let page = TensorPage::new_memory(
                "model.embed_tokens.weight".to_string(),
                vec![2, 2],
                DType::F32,
                vec![0; 16],
            );
            model_mut
                .layers
                .insert("model.embed_tokens.weight".to_string(), page);
        }

        // Call ensure loaded - should be a no-op because layers are not empty
        db.ensure_model_loaded("llama-dummy").unwrap();
        assert!(!db.models.get("llama-dummy").unwrap().layers.is_empty());

        // Call unload
        db.unload_model_layers("llama-dummy");
        assert!(db.models.get("llama-dummy").unwrap().layers.is_empty());
    }

    #[test]
    fn test_manifest_load_bypass() {
        let temp_dir = TempDir::new().unwrap();
        let mut db = TensorDB::new(temp_dir.path().to_path_buf());

        // Write a test manifest into the model directory
        crate::storage::storage_manifest::write_test_manifest(
            temp_dir.path(),
            "bypass-model",
            100, // vocab_size
            64,  // hidden_size
            8,   // num_q_heads
            2,   // num_kv_heads
            8,   // head_dim
            128, // mlp_size
        );

        // Also write some dummy .bin files to simulate the fallback path
        let dummy_bin = temp_dir.path().join("model.embed_tokens.weight.bin");
        std::fs::write(&dummy_bin, vec![0u8; 100 * 64 * 4]).unwrap();

        // 1. With BRAMHA_MANIFEST_LOAD unset or true (should load via manifest)
        // SAFETY: Manual invariants verified for performance/FFI
        unsafe {
            std::env::set_var("BRAMHA_MANIFEST_LOAD", "true");
        }
        db.restore_model_at_path("bypass-model".to_string(), temp_dir.path());
        assert!(db.models.get("bypass-model").unwrap().manifest.is_some());

        db.load_model_layers("bypass-model").unwrap();
        // Since the files for other layers in manifest are missing, loading will log warnings but register what it can.
        // Let's unload and test the bypass now.
        db.unload_model_layers("bypass-model");

        // 2. With BRAMHA_MANIFEST_LOAD=false
        // SAFETY: Manual invariants verified for performance/FFI
        unsafe {
            std::env::set_var("BRAMHA_MANIFEST_LOAD", "false");
        }
        // Clear the model and restore again to verify manifest is not used during load
        db.models.clear();
        db.restore_model_at_path("bypass-model".to_string(), temp_dir.path());

        db.load_model_layers("bypass-model").unwrap();
        // The manifest has been bypassed, so it fell back to scanning the folder and found only the model.embed_tokens.weight shard
        let model = db.models.get("bypass-model").unwrap();
        assert!(model.layers.contains_key("model.embed.tokens.weight"));

        // SAFETY: Manual invariants verified for performance/FFI
        unsafe {
            std::env::remove_var("BRAMHA_MANIFEST_LOAD");
        }
    }
}
