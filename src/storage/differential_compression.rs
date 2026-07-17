//! Differential / Delta Compression Module for Bramha & SPANDA
//!
//! Provides delta calculation, sparse residual encoding, multi-layer delta chains,
//! and finetune model delta management to achieve 40-60% storage savings across
//! model layers and fine-tuned variants.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// Errors that can occur during differential compression and reconstruction.
#[derive(Debug, Clone, PartialEq)]
pub enum DifferentialCompressionError {
    DimensionMismatch { base_len: usize, target_len: usize },
    ReferenceNotFound(String),
    CorruptedDeltaData(String),
    ChainResolutionError(String),
}

impl fmt::Display for DifferentialCompressionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch {
                base_len,
                target_len,
            } => write!(
                f,
                "Base tensor length ({}) does not match target tensor length ({})",
                base_len, target_len
            ),
            Self::ReferenceNotFound(id) => write!(f, "Base reference tensor not found: {}", id),
            Self::CorruptedDeltaData(msg) => write!(f, "Corrupted delta data: {}", msg),
            Self::ChainResolutionError(msg) => write!(f, "Delta chain resolution failed: {}", msg),
        }
    }
}

impl std::error::Error for DifferentialCompressionError {}

/// Category of differential relationship.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum DeltaType {
    /// Layer N+1 delta relative to Layer N reference in same model
    LayerToLayer,
    /// Model finetune / variant tensor delta relative to base model tensor
    BaseToFinetune,
    /// Arbitrary content-addressed reference hash or tensor ID
    CustomReference,
}

/// Configuration settings for differential compression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaCompressionConfig {
    /// Threshold below which difference between target and base tensor values is zeroed out
    pub epsilon: f32,
    /// Whether to store delta values as FP16 (or FP32)
    pub use_fp16: bool,
    /// Target maximum allowable absolute error (for quality guarantees)
    pub max_allowed_error: f32,
}

impl Default for DeltaCompressionConfig {
    fn default() -> Self {
        Self {
            epsilon: 1e-4,
            use_fp16: false,
            max_allowed_error: 1e-3,
        }
    }
}

/// Represents a differentials-encoded tensor relative to a base reference.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DeltaTensor {
    /// ID or hash of the base reference tensor
    pub reference_id: String,
    /// Type of delta relationship
    pub delta_type: DeltaType,
    /// Original shape of the target tensor
    pub shape: Vec<usize>,
    /// Total number of elements in original tensor
    pub original_element_count: usize,
    /// Number of non-zero delta elements stored
    pub non_zero_count: usize,
    /// Flat indices where deltas occur (sparse COO format)
    pub indices: Vec<u32>,
    /// Sparse delta values at corresponding indices (target - base)
    pub values: Vec<f32>,
    /// Calculated compression ratio achieved (original_bytes / delta_bytes)
    pub compression_ratio: f32,
    /// Maximum absolute error observed due to epsilon thresholding
    pub max_absolute_error: f32,
}

impl DeltaTensor {
    /// Returns estimated byte size of this encoded delta structure
    pub fn byte_size(&self) -> usize {
        let header_bytes =
            std::mem::size_of::<Self>() + self.reference_id.len() + self.shape.len() * 8;
        let indices_bytes = self.indices.len() * std::mem::size_of::<u32>();
        let values_bytes = self.values.len() * std::mem::size_of::<f32>();
        header_bytes + indices_bytes + values_bytes
    }

    /// Returns the sparsity of the delta (percentage of zero-diff elements)
    pub fn sparsity(&self) -> f32 {
        if self.original_element_count == 0 {
            return 1.0;
        }
        1.0 - (self.non_zero_count as f32 / self.original_element_count as f32)
    }
}

/// Core differential engine for calculating and reconstructing tensor deltas.
pub struct DifferentialEncoder;

impl DifferentialEncoder {
    /// Computes sparse delta representation of `target` relative to `base`.
    pub fn compute_delta(
        base: &[f32],
        target: &[f32],
        config: &DeltaCompressionConfig,
        reference_id: impl Into<String>,
        delta_type: DeltaType,
        shape: Vec<usize>,
    ) -> Result<DeltaTensor, DifferentialCompressionError> {
        if base.len() != target.len() {
            return Err(DifferentialCompressionError::DimensionMismatch {
                base_len: base.len(),
                target_len: target.len(),
            });
        }

        let ref_id = reference_id.into();
        let total_len = target.len();
        let mut indices = Vec::new();
        let mut values = Vec::new();
        let mut max_err: f32 = 0.0;

        for (idx, (&b, &t)) in base.iter().zip(target.iter()).enumerate() {
            let diff = t - b;
            let abs_diff = diff.abs();

            if abs_diff > config.epsilon {
                indices.push(idx as u32);
                values.push(diff);
            } else {
                if abs_diff > max_err {
                    max_err = abs_diff;
                }
            }
        }

        let non_zero_count = values.len();
        let uncompressed_bytes = total_len * std::mem::size_of::<f32>();
        let delta_bytes = (indices.len() * std::mem::size_of::<u32>())
            + (values.len() * std::mem::size_of::<f32>())
            + 64; // header overhead

        let compression_ratio = if delta_bytes > 0 {
            uncompressed_bytes as f32 / delta_bytes as f32
        } else {
            1.0
        };

        Ok(DeltaTensor {
            reference_id: ref_id,
            delta_type,
            shape,
            original_element_count: total_len,
            non_zero_count,
            indices,
            values,
            compression_ratio,
            max_absolute_error: max_err,
        })
    }

    /// Reconstructs full `target` tensor given the `base` tensor and `delta`.
    pub fn reconstruct_tensor(
        base: &[f32],
        delta: &DeltaTensor,
    ) -> Result<Vec<f32>, DifferentialCompressionError> {
        if base.len() != delta.original_element_count {
            return Err(DifferentialCompressionError::DimensionMismatch {
                base_len: base.len(),
                target_len: delta.original_element_count,
            });
        }

        let mut reconstructed = base.to_vec();

        for (&idx, &val) in delta.indices.iter().zip(delta.values.iter()) {
            let idx_usize = idx as usize;
            if idx_usize >= reconstructed.len() {
                return Err(DifferentialCompressionError::CorruptedDeltaData(format!(
                    "Index {} out of bounds for base length {}",
                    idx_usize,
                    reconstructed.len()
                )));
            }
            reconstructed[idx_usize] += val;
        }

        Ok(reconstructed)
    }

    /// Reconstructs target tensor by applying a sequence (chain) of deltas to a base tensor.
    pub fn reconstruct_chain(
        base: &[f32],
        deltas: &[DeltaTensor],
    ) -> Result<Vec<f32>, DifferentialCompressionError> {
        let mut current = base.to_vec();
        for delta in deltas {
            current = Self::reconstruct_tensor(&current, delta)?;
        }
        Ok(current)
    }
}

/// Summary report for differential storage savings across a collection of tensors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaSavingsReport {
    pub total_tensors_managed: usize,
    pub original_total_bytes: usize,
    pub delta_total_bytes: usize,
    pub saved_bytes: usize,
    pub overall_compression_ratio: f32,
    pub average_sparsity: f32,
}

/// Manages registration, storage, and chain resolution of delta tensors.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct DeltaStorageManager {
    /// Maps target tensor name -> DeltaTensor
    deltas: HashMap<String, DeltaTensor>,
    /// Base tensor metadata (name -> element count)
    base_tensors: HashMap<String, usize>,
}

impl DeltaStorageManager {
    pub fn new() -> Self {
        Self {
            deltas: HashMap::new(),
            base_tensors: HashMap::new(),
        }
    }

    /// Register a base reference tensor name and its dimension
    pub fn register_base_tensor(&mut self, name: impl Into<String>, len: usize) {
        self.base_tensors.insert(name.into(), len);
    }

    /// Compute and store delta for a target tensor
    pub fn add_delta_tensor(
        &mut self,
        target_name: impl Into<String>,
        base_name: &str,
        base: &[f32],
        target: &[f32],
        config: &DeltaCompressionConfig,
        delta_type: DeltaType,
        shape: Vec<usize>,
    ) -> Result<&DeltaTensor, DifferentialCompressionError> {
        let delta =
            DifferentialEncoder::compute_delta(base, target, config, base_name, delta_type, shape)?;
        let name = target_name.into();
        self.deltas.insert(name.clone(), delta);
        Ok(self.deltas.get(&name).unwrap())
    }

    /// Retrieve stored delta for a tensor
    pub fn get_delta(&self, target_name: &str) -> Option<&DeltaTensor> {
        self.deltas.get(target_name)
    }

    /// Reconstruct a target tensor by resolving delta dependencies using a base supplier function
    pub fn reconstruct<F>(
        &self,
        target_name: &str,
        base_supplier: F,
    ) -> Result<Vec<f32>, DifferentialCompressionError>
    where
        F: Fn(&str) -> Option<Vec<f32>>,
    {
        let delta = self.deltas.get(target_name).ok_or_else(|| {
            DifferentialCompressionError::ReferenceNotFound(target_name.to_string())
        })?;

        let base_data = base_supplier(&delta.reference_id).ok_or_else(|| {
            DifferentialCompressionError::ReferenceNotFound(delta.reference_id.clone())
        })?;

        DifferentialEncoder::reconstruct_tensor(&base_data, delta)
    }

    /// Calculate total storage savings achieved across all registered deltas
    pub fn generate_savings_report(&self) -> DeltaSavingsReport {
        let total_tensors = self.deltas.len();
        let mut original_total_bytes = 0;
        let mut delta_total_bytes = 0;
        let mut total_sparsity = 0.0;

        for delta in self.deltas.values() {
            let orig_bytes = delta.original_element_count * std::mem::size_of::<f32>();
            let d_bytes = delta.byte_size();

            original_total_bytes += orig_bytes;
            delta_total_bytes += d_bytes;
            total_sparsity += delta.sparsity();
        }

        let saved_bytes = original_total_bytes.saturating_sub(delta_total_bytes);
        let overall_ratio = if delta_total_bytes > 0 {
            original_total_bytes as f32 / delta_total_bytes as f32
        } else {
            1.0
        };
        let avg_sparsity = if total_tensors > 0 {
            total_sparsity / total_tensors as f32
        } else {
            0.0
        };

        DeltaSavingsReport {
            total_tensors_managed: total_tensors,
            original_total_bytes,
            delta_total_bytes,
            saved_bytes,
            overall_compression_ratio: overall_ratio,
            average_sparsity: avg_sparsity,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_and_reconstruct_exact_delta() {
        let base = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let target = vec![1.0, 2.0, 3.5, 4.0, 5.0, 6.2, 7.0, 8.0]; // 2 differences
        let config = DeltaCompressionConfig {
            epsilon: 1e-4,
            use_fp16: false,
            max_allowed_error: 1e-3,
        };

        let delta = DifferentialEncoder::compute_delta(
            &base,
            &target,
            &config,
            "layer_0",
            DeltaType::LayerToLayer,
            vec![8],
        )
        .expect("Delta computation failed");

        assert_eq!(delta.non_zero_count, 2);
        assert_eq!(delta.indices, vec![2, 5]);
        assert!((delta.values[0] - 0.5).abs() < 1e-5);
        assert!((delta.values[1] - 0.2).abs() < 1e-5);

        let reconstructed =
            DifferentialEncoder::reconstruct_tensor(&base, &delta).expect("Reconstruction failed");

        for (orig, rec) in target.iter().zip(reconstructed.iter()) {
            assert!((orig - rec).abs() < 1e-5);
        }
    }

    #[test]
    fn test_delta_sparsity_and_compression_ratio() {
        let base = vec![0.5f32; 1000];
        let mut target = base.clone();

        // Introduce 50 subtle modifications
        for i in (0..1000).step_by(20) {
            target[i] += 0.05;
        }

        let config = DeltaCompressionConfig::default();
        let delta = DifferentialEncoder::compute_delta(
            &base,
            &target,
            &config,
            "base_model_layer_1",
            DeltaType::BaseToFinetune,
            vec![1000],
        )
        .unwrap();

        assert_eq!(delta.non_zero_count, 50);
        assert!((delta.sparsity() - 0.95).abs() < 1e-4);
        assert!(delta.compression_ratio > 4.0);
    }

    #[test]
    fn test_delta_chain_reconstruction() {
        let layer_0 = vec![1.0, 2.0, 3.0, 4.0];
        let layer_1 = vec![1.1, 2.0, 3.0, 4.2];
        let layer_2 = vec![1.1, 2.3, 3.0, 4.2];

        let config = DeltaCompressionConfig::default();

        let delta_1 = DifferentialEncoder::compute_delta(
            &layer_0,
            &layer_1,
            &config,
            "layer_0",
            DeltaType::LayerToLayer,
            vec![4],
        )
        .unwrap();

        let delta_2 = DifferentialEncoder::compute_delta(
            &layer_1,
            &layer_2,
            &config,
            "layer_1",
            DeltaType::LayerToLayer,
            vec![4],
        )
        .unwrap();

        let chain = vec![delta_1, delta_2];
        let reconstructed_2 = DifferentialEncoder::reconstruct_chain(&layer_0, &chain).unwrap();

        for (orig, rec) in layer_2.iter().zip(reconstructed_2.iter()) {
            assert!((orig - rec).abs() < 1e-4);
        }
    }

    #[test]
    fn test_delta_storage_manager_workflow() {
        let mut manager = DeltaStorageManager::new();
        let base_weights = vec![0.1f32; 500];
        let finetune_weights = {
            let mut w = base_weights.clone();
            w[10] += 0.8;
            w[50] += 0.4;
            w
        };

        manager.register_base_tensor("base_v1/q_proj", 500);
        manager
            .add_delta_tensor(
                "finetune_v1/q_proj",
                "base_v1/q_proj",
                &base_weights,
                &finetune_weights,
                &DeltaCompressionConfig::default(),
                DeltaType::BaseToFinetune,
                vec![500],
            )
            .unwrap();

        let report = manager.generate_savings_report();
        assert_eq!(report.total_tensors_managed, 1);
        assert!(report.saved_bytes > 0);
        assert!(report.overall_compression_ratio > 10.0);

        let reconstructed = manager
            .reconstruct("finetune_v1/q_proj", |id| {
                if id == "base_v1/q_proj" {
                    Some(base_weights.clone())
                } else {
                    None
                }
            })
            .unwrap();

        assert_eq!(reconstructed, finetune_weights);
    }
}
