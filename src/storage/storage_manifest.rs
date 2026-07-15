// Sprint 8 Deliverable — FOUNDATION COMPLETE
// Performance targets: UNVALIDATED until Sprint 9 integration benchmarks pass
// Current status: Module-level unit tests pass. End-to-end metrics are projections.

/// Storage Manifest: Metadata layer for intelligent model storage
///
/// This module tracks:
/// - Model versions and variants
/// - Layer importance tiers (critical, important, robust, redundant)
/// - Quantization levels per layer
/// - Compression formats used
/// - Storage locations and offsets
/// - Compression ratios achieved
/// - Access patterns and statistics
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum StorageTier {
    /// DRAM, always loaded, full precision required
    Critical,
    /// SSD Tier 1, preloaded for most inferences, INT8 or higher
    Important,
    /// SSD Tier 2, lazily loaded, INT4 safe
    Robust,
    /// HDD/Network, rarely loaded, extreme compression or computed on-demand
    Redundant,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum CompressionFormat {
    /// No compression, raw f32
    None,
    /// INT8 quantization with linear scaling
    Int8Linear,
    /// INT4 quantization with per-channel scales
    Int4PerChannel,
    /// INT4 with lookup table (LUT) dequantization
    Int4Lut,
    /// Extreme compression: mostly 0s or constants
    Dictionary,
    /// SVD factorization: U @ S @ Vt
    Svd,
    /// Delta from reference layer
    DeltaFromLayer(String),
    /// Columnar storage with dictionary encoding
    ColumnarDict,
    /// GGUF layout fallback
    Gguf,
    /// Differential compression: stores Delta relative to a reference
    Differential {
        delta_format: Box<CompressionFormat>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AccessStats {
    pub access_count: u64,
    pub last_accessed_timestamp: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerMetadata {
    /// Layer identifier (e.g., "model.layers.0.self_attn.q_proj.weight")
    pub layer_id: String,

    /// Shape of the tensor (e.g., [2048, 8192])
    pub shape: Vec<usize>,

    /// Total number of elements
    pub num_elements: usize,

    /// Original size in bytes (if full f32)
    pub original_bytes: u64,

    /// Actual stored bytes after compression
    pub stored_bytes: u64,

    /// Compression format used
    pub compression_format: CompressionFormat,

    /// Storage tier classification
    pub storage_tier: StorageTier,

    /// If this layer is differentially compressed, the ID of the reference layer
    pub reference_tensor: Option<String>,

    #[serde(default)]
    pub statistics: AccessStats,

    /// Is this layer quantized? If so, at what bit width
    pub quantization_bits: Option<u32>,

    /// Reference layer (for delta compression)
    pub reference_layer: Option<String>,

    /// Checksum for integrity verification
    pub checksum: String,

    /// SVD rank (if using SVD compression)
    pub svd_rank: Option<usize>,

    /// Dictionary size (if using dictionary encoding)
    pub dict_size: Option<usize>,

    /// Chunk index: names of sub-layer chunks that make up this layer
    #[serde(default)]
    pub chunks: Option<Vec<String>>,

    // ── Phase 5: Pipeline Parallelism / Dynamic Tensor Sharding ──────────────
    /// Which compute device owns this layer shard, e.g. "cpu:0", "cpu:1", "wgpu:0".
    /// None = unassigned (planner will decide at runtime).
    #[serde(default)]
    pub device_assignment: Option<String>,

    /// For tensor-parallel splits: rank of this shard within the world.
    /// E.g. rank=1 in world_size=4 means this file holds rows [N/4 .. N/2].
    #[serde(default)]
    pub shard_rank: Option<usize>,

    /// Total number of shards this layer is split into (tensor parallelism).
    #[serde(default)]
    pub shard_world_size: Option<usize>,
}

impl LayerMetadata {
    pub fn new(layer_id: String, shape: Vec<usize>) -> Self {
        let num_elements: usize = shape.iter().product();
        let original_bytes = (num_elements as u64) * 4; // f32 = 4 bytes

        LayerMetadata {
            layer_id,
            shape,
            num_elements,
            original_bytes,
            stored_bytes: original_bytes,
            compression_format: CompressionFormat::None,
            storage_tier: StorageTier::Important,
            reference_tensor: None,
            statistics: AccessStats::default(),
            quantization_bits: None,
            reference_layer: None,
            checksum: String::new(),
            svd_rank: None,
            dict_size: None,
            chunks: None,
            device_assignment: None,
            shard_rank: None,
            shard_world_size: None,
        }
    }

    /// Calculate compression ratio
    pub fn compression_ratio(&self) -> f64 {
        if self.stored_bytes == 0 {
            0.0
        } else {
            self.original_bytes as f64 / self.stored_bytes as f64
        }
    }

    /// Get human-readable compression format description
    pub fn compression_description(&self) -> String {
        match self.compression_format {
            CompressionFormat::None => "None (F32)".to_string(),
            CompressionFormat::Int8Linear => "INT8 (linear scale)".to_string(),
            CompressionFormat::Int4PerChannel => "INT4 (per-channel)".to_string(),
            CompressionFormat::Int4Lut => "INT4 (LUT deq)".to_string(),
            CompressionFormat::Dictionary => "Dictionary encoded".to_string(),
            CompressionFormat::Svd => {
                format!("SVD (rank={})", self.svd_rank.unwrap_or(256))
            }
            CompressionFormat::DeltaFromLayer(ref ref_layer) => {
                format!("Delta from {}", ref_layer)
            }
            CompressionFormat::ColumnarDict => "Columnar + Dict".to_string(),
            CompressionFormat::Gguf => "GGUF".to_string(),
            CompressionFormat::Differential { .. } => "Differential".to_string(),
        }
    }

    /// Get storage tier description
    pub fn tier_description(&self) -> &'static str {
        match self.storage_tier {
            StorageTier::Critical => "DRAM (critical path, always loaded)",
            StorageTier::Important => "SSD Tier 1 (preloaded, fast access)",
            StorageTier::Robust => "SSD Tier 2 (lazy loaded, normal access)",
            StorageTier::Redundant => "HDD/Network (on-demand, slow access)",
        }
    }

    /// Mark this layer as accessed and update timestamp
    pub fn record_access(&mut self, timestamp: u64) {
        self.statistics.access_count += 1;
        self.statistics.last_accessed_timestamp = Some(timestamp);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelManifest {
    /// Model name (e.g., "tinyllama")
    pub name: String,

    /// Model version/variant (e.g., "base", "q4", "q8")
    pub variant: String,

    /// Total number of layers
    pub num_layers: usize,

    /// Model architecture (e.g., "llama", "qwen", "phi")
    pub architecture: String,

    /// Embedding dimension
    pub hidden_size: usize,

    /// Number of attention heads
    pub num_heads: usize,

    /// Number of KV heads (for grouped query attention)
    pub num_kv_heads: usize,

    /// Metadata for each layer
    pub layers: HashMap<String, LayerMetadata>,

    /// Total original size (all layers f32)
    pub total_original_size: u64,

    /// Total stored size (after compression)
    pub total_stored_size: u64,

    /// Creation timestamp
    pub created_at: u64,

    /// Last modified timestamp
    pub last_modified: u64,

    /// Checksum of entire manifest
    pub manifest_checksum: String,

    /// Storage directory path
    pub storage_path: PathBuf,

    /// Compression statistics
    pub avg_compression_ratio: f64,
    pub max_compression_ratio: f64,
    pub min_compression_ratio: f64,

    /// Quantization summary
    pub quantized_layers: usize,
    pub full_precision_layers: usize,

    /// Tier distribution
    pub critical_count: usize,
    pub important_count: usize,
    pub robust_count: usize,
    pub redundant_count: usize,

    /// MoE Architecture support
    #[serde(default)]
    pub num_experts: Option<usize>,
    #[serde(default)]
    pub expert_routing_top_k: Option<usize>,
}

impl ModelManifest {
    pub fn new(
        name: String,
        variant: String,
        num_layers: usize,
        architecture: String,
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        storage_path: PathBuf,
    ) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        ModelManifest {
            name,
            variant,
            num_layers,
            architecture,
            hidden_size,
            num_heads,
            num_kv_heads,
            layers: HashMap::new(),
            total_original_size: 0,
            total_stored_size: 0,
            created_at: now,
            last_modified: now,
            manifest_checksum: String::new(),
            storage_path,
            avg_compression_ratio: 0.0,
            max_compression_ratio: 0.0,
            min_compression_ratio: 0.0,
            quantized_layers: 0,
            full_precision_layers: 0,
            critical_count: 0,
            important_count: 0,
            robust_count: 0,
            redundant_count: 0,
            num_experts: None,
            expert_routing_top_k: None,
        }
    }

    /// Add layer metadata to manifest
    pub fn add_layer(&mut self, metadata: LayerMetadata) {
        self.total_original_size += metadata.original_bytes;
        self.total_stored_size += metadata.stored_bytes;

        if metadata.quantization_bits.is_some() {
            self.quantized_layers += 1;
        } else {
            self.full_precision_layers += 1;
        }

        match metadata.storage_tier {
            StorageTier::Critical => self.critical_count += 1,
            StorageTier::Important => self.important_count += 1,
            StorageTier::Robust => self.robust_count += 1,
            StorageTier::Redundant => self.redundant_count += 1,
        }

        self.layers.insert(metadata.layer_id.clone(), metadata);
        self.update_statistics();
    }

    /// Update compression and tier statistics
    pub fn update_statistics(&mut self) {
        if self.layers.is_empty() {
            return;
        }

        let ratios: Vec<f64> = self
            .layers
            .values()
            .map(|layer| layer.compression_ratio())
            .collect();

        self.avg_compression_ratio = ratios.iter().sum::<f64>() / ratios.len() as f64;
        self.max_compression_ratio = ratios.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        self.min_compression_ratio = ratios.iter().cloned().fold(f64::INFINITY, f64::min);

        self.last_modified = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
    }

    /// Get layers by tier
    pub fn get_layers_by_tier(&self, tier: StorageTier) -> Vec<&LayerMetadata> {
        self.layers
            .values()
            .filter(|layer| layer.storage_tier == tier)
            .collect()
    }

    /// Get total size for a specific tier
    pub fn size_for_tier(&self, tier: StorageTier) -> u64 {
        self.get_layers_by_tier(tier)
            .iter()
            .map(|layer| layer.stored_bytes)
            .sum()
    }

    /// Estimate DRAM usage for this model (sum of critical + important tiers)
    pub fn estimated_dram_usage(&self) -> u64 {
        self.size_for_tier(StorageTier::Critical) + self.size_for_tier(StorageTier::Important)
    }

    /// Generate a human-readable report
    pub fn report(&self) {
        println!("\n📦 Model Storage Manifest Report");
        println!("═══════════════════════════════════════════════════════════");
        println!(
            "Model: {} (variant: {}, architecture: {})",
            self.name, self.variant, self.architecture
        );
        println!(
            "Layers: {} | Heads: {} (KV heads: {})",
            self.num_layers, self.num_heads, self.num_kv_heads
        );
        println!("\n📊 Size Statistics");
        println!("───────────────────────────────────────────────────────────");
        println!(
            "Original size: {:.2} MB",
            self.total_original_size as f64 / 1024.0 / 1024.0
        );
        println!(
            "Stored size:   {:.2} MB",
            self.total_stored_size as f64 / 1024.0 / 1024.0
        );
        println!(
            "Compression:   {:.2}x (min: {:.2}x, max: {:.2}x, avg: {:.2}x)",
            self.total_original_size as f64 / (self.total_stored_size as f64 + 1e-10),
            self.min_compression_ratio,
            self.max_compression_ratio,
            self.avg_compression_ratio,
        );
        println!(
            "Saved: {:.2} MB",
            (self.total_original_size as i64 - self.total_stored_size as i64) as f64
                / 1024.0
                / 1024.0
        );

        println!("\n🔐 Quantization Statistics");
        println!("───────────────────────────────────────────────────────────");
        println!("Full precision: {} layers", self.full_precision_layers);
        println!("Quantized:      {} layers", self.quantized_layers);
        println!(
            "Ratio:          {:.1}% quantized",
            (self.quantized_layers as f64 / self.num_layers as f64) * 100.0
        );

        println!("\n📍 Storage Tier Distribution");
        println!("───────────────────────────────────────────────────────────");
        println!(
            "Critical (DRAM):     {} layers, {:.2} MB",
            self.critical_count,
            self.size_for_tier(StorageTier::Critical) as f64 / 1024.0 / 1024.0
        );
        println!(
            "Important (SSD Tier 1): {} layers, {:.2} MB",
            self.important_count,
            self.size_for_tier(StorageTier::Important) as f64 / 1024.0 / 1024.0
        );
        println!(
            "Robust (SSD Tier 2):    {} layers, {:.2} MB",
            self.robust_count,
            self.size_for_tier(StorageTier::Robust) as f64 / 1024.0 / 1024.0
        );
        println!(
            "Redundant (HDD):    {} layers, {:.2} MB",
            self.redundant_count,
            self.size_for_tier(StorageTier::Redundant) as f64 / 1024.0 / 1024.0
        );
        println!(
            "Estimated DRAM:     {:.2} MB",
            self.estimated_dram_usage() as f64 / 1024.0 / 1024.0
        );

        println!(
            "\n⏰ Created: {} | Modified: {}",
            self.created_at, self.last_modified
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_metadata() {
        let mut layer = LayerMetadata::new("q_proj.weight".to_string(), vec![2048, 8192]);
        assert_eq!(layer.num_elements, 2048 * 8192);
        assert_eq!(layer.original_bytes, 2048 * 8192 * 4);
        assert_eq!(layer.compression_ratio(), 1.0);

        layer.stored_bytes = layer.original_bytes / 4;
        assert_eq!(layer.compression_ratio(), 4.0);
    }

    #[test]
    fn test_model_manifest() {
        let mut manifest = ModelManifest::new(
            "tinyllama".to_string(),
            "base".to_string(),
            22,
            "llama".to_string(),
            2048,
            32,
            8,
            PathBuf::from("/tmp/models"),
        );

        assert_eq!(manifest.num_layers, 22);
        assert_eq!(manifest.estimated_dram_usage(), 0);

        let mut layer = LayerMetadata::new("layer_0.q_proj".to_string(), vec![2048, 8192]);
        layer.storage_tier = StorageTier::Critical;
        manifest.add_layer(layer);

        assert_eq!(manifest.critical_count, 1);
        assert!(manifest.estimated_dram_usage() > 0);
    }

    #[test]
    fn test_manifest_serialization_roundtrip() {
        let temp_dir = tempfile::tempdir().unwrap();
        let manifest_path = temp_dir.path().join("manifest.json");

        let mut manifest = ModelManifest::new(
            "test_model".to_string(),
            "f32".to_string(),
            12,
            "llama".to_string(),
            2048,
            32,
            8,
            temp_dir.path().to_path_buf(),
        );

        let mut layer = LayerMetadata::new("layer_0".to_string(), vec![1024, 1024]);
        layer.storage_tier = StorageTier::Important;
        layer.compression_format = CompressionFormat::Int8Linear;
        manifest.add_layer(layer);

        let json_str = serde_json::to_string(&manifest).unwrap();
        std::fs::write(&manifest_path, &json_str).unwrap();

        let loaded_json = std::fs::read_to_string(&manifest_path).unwrap();
        let loaded_manifest: ModelManifest = serde_json::from_str(&loaded_json).unwrap();

        assert_eq!(loaded_manifest.name, "test_model");
        assert_eq!(loaded_manifest.num_layers, 12);
        assert_eq!(loaded_manifest.important_count, 1);
        assert_eq!(
            loaded_manifest
                .layers
                .get("layer_0")
                .unwrap()
                .compression_format,
            CompressionFormat::Int8Linear
        );
    }

    #[test]
    fn test_mock_manifest_helpers() {
        let temp_dir = tempfile::tempdir().unwrap();
        write_mock_manifest(temp_dir.path(), "qwen", 32000, 2048, 32, 8, 64, 5632);

        let manifest_path = temp_dir.path().join("manifest.json");
        assert!(manifest_path.exists());

        let json_str = std::fs::read_to_string(&manifest_path).unwrap();
        let manifest: ModelManifest = serde_json::from_str(&json_str).unwrap();
        assert_eq!(manifest.name, "qwen");
        assert!(manifest.layers.contains_key("model.embed_tokens.weight"));
    }
}

pub fn write_mock_manifest(
    dir: &std::path::Path,
    name: &str,
    vocab_size: usize,
    hidden_size: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    mlp_size: usize,
) {
    let mut manifest = ModelManifest::new(
        name.to_string(),
        "base".to_string(),
        1,
        "llama".to_string(),
        hidden_size,
        num_q_heads,
        num_kv_heads,
        dir.to_path_buf(),
    );

    let mut add_mock_layer = |layer_id: &str, shape: Vec<usize>| {
        let mut meta = LayerMetadata::new(layer_id.to_string(), shape);
        meta.storage_tier = if layer_id.contains("embed")
            || layer_id.contains("lm_head")
            || layer_id.contains("norm")
            || layer_id.contains("attn")
        {
            StorageTier::Critical
        } else {
            StorageTier::Important
        };
        manifest.add_layer(meta);
    };

    add_mock_layer("model.embed_tokens.weight", vec![vocab_size, hidden_size]);
    add_mock_layer("lm_head.weight", vec![vocab_size, hidden_size]);
    add_mock_layer("model.norm.weight", vec![hidden_size]);
    add_mock_layer("model.layers.0.input_layernorm.weight", vec![hidden_size]);
    add_mock_layer(
        "model.layers.0.self_attn.q_proj.weight",
        vec![num_q_heads * head_dim, hidden_size],
    );
    add_mock_layer(
        "model.layers.0.self_attn.k_proj.weight",
        vec![num_kv_heads * head_dim, hidden_size],
    );
    add_mock_layer(
        "model.layers.0.self_attn.v_proj.weight",
        vec![num_kv_heads * head_dim, hidden_size],
    );
    add_mock_layer(
        "model.layers.0.self_attn.o_proj.weight",
        vec![hidden_size, num_q_heads * head_dim],
    );
    add_mock_layer(
        "model.layers.0.post_attention_layernorm.weight",
        vec![hidden_size],
    );
    add_mock_layer(
        "model.layers.0.mlp.gate_proj.weight",
        vec![mlp_size, hidden_size],
    );
    add_mock_layer(
        "model.layers.0.mlp.up_proj.weight",
        vec![mlp_size, hidden_size],
    );
    add_mock_layer(
        "model.layers.0.mlp.down_proj.weight",
        vec![hidden_size, mlp_size],
    );

    let manifest_json = serde_json::to_string_pretty(&manifest).unwrap();
    std::fs::write(dir.join("manifest.json"), manifest_json).unwrap();
}

pub fn write_mock_moe_manifest(
    dir: &std::path::Path,
    name: &str,
    vocab_size: usize,
    hidden_size: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    mlp_size: usize,
    num_experts: usize,
    expert_routing_top_k: usize,
) {
    let mut manifest = ModelManifest::new(
        name.to_string(),
        "base".to_string(),
        1,
        "llama".to_string(),
        hidden_size,
        num_q_heads,
        num_kv_heads,
        dir.to_path_buf(),
    );
    manifest.num_experts = Some(num_experts);
    manifest.expert_routing_top_k = Some(expert_routing_top_k);

    let mut add_mock_layer = |layer_id: &str, shape: Vec<usize>, chunks: Option<Vec<String>>| {
        let mut meta = LayerMetadata::new(layer_id.to_string(), shape);
        meta.storage_tier = if layer_id.contains("embed")
            || layer_id.contains("lm_head")
            || layer_id.contains("norm")
            || layer_id.contains("attn")
        {
            StorageTier::Critical
        } else {
            StorageTier::Important
        };
        meta.chunks = chunks;
        manifest.add_layer(meta);
    };

    add_mock_layer(
        "model.embed_tokens.weight",
        vec![vocab_size, hidden_size],
        None,
    );
    add_mock_layer("lm_head.weight", vec![vocab_size, hidden_size], None);
    add_mock_layer("model.norm.weight", vec![hidden_size], None);
    add_mock_layer(
        "model.layers.0.input_layernorm.weight",
        vec![hidden_size],
        None,
    );
    add_mock_layer(
        "model.layers.0.self_attn.q_proj.weight",
        vec![num_q_heads * head_dim, hidden_size],
        None,
    );
    add_mock_layer(
        "model.layers.0.self_attn.k_proj.weight",
        vec![num_kv_heads * head_dim, hidden_size],
        None,
    );
    add_mock_layer(
        "model.layers.0.self_attn.v_proj.weight",
        vec![num_kv_heads * head_dim, hidden_size],
        None,
    );
    add_mock_layer(
        "model.layers.0.self_attn.o_proj.weight",
        vec![hidden_size, num_q_heads * head_dim],
        None,
    );
    add_mock_layer(
        "model.layers.0.post_attention_layernorm.weight",
        vec![hidden_size],
        None,
    );

    // Router / Gate Weight
    add_mock_layer(
        "model.layers.0.mlp.router.weight",
        vec![num_experts, hidden_size],
        None,
    );

    // Dynamic Chunks for Experts
    let mut mlp_chunks = Vec::new();
    for e in 0..num_experts {
        let gate_name = format!("model.layers.0.mlp.experts.{}.gate_proj.weight", e);
        let up_name = format!("model.layers.0.mlp.experts.{}.up_proj.weight", e);
        let down_name = format!("model.layers.0.mlp.experts.{}.down_proj.weight", e);

        add_mock_layer(&gate_name, vec![mlp_size, hidden_size], None);
        add_mock_layer(&up_name, vec![mlp_size, hidden_size], None);
        add_mock_layer(&down_name, vec![hidden_size, mlp_size], None);

        mlp_chunks.push(gate_name);
        mlp_chunks.push(up_name);
        mlp_chunks.push(down_name);
    }

    // Add main MLP layer which owns/indexes all the expert chunks
    add_mock_layer("model.layers.0.mlp", vec![0], Some(mlp_chunks));

    let manifest_json = serde_json::to_string_pretty(&manifest).unwrap();
    std::fs::write(dir.join("manifest.json"), manifest_json).unwrap();
}
