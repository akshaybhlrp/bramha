use serde::{Deserialize, Serialize};
/// Phase 5 — Pipeline Parallelism & Dynamic Tensor Sharding
///
/// # Design: Devices as DB Replicas
///
/// Bramha treats physical compute devices the same way it treats database shards:
///
/// - A [`DeviceMesh`] is a logical shard map — an ordered list of compute slots.
/// - A [`ShardingPlanner`] is a query planner — it reads `LayerMetadata.size_bytes` from
///   the `ModelManifest` and greedily assigns consecutive layer groups to devices up to their
///   DRAM budget, exactly like a DB cost-based optimizer fills a buffer pool.
/// - A [`PipelineExecutor`] streams activations through stage N → N+1 via `Arc<Vec<f32>>`,
///   analogous to row pipelining in a query execution engine.
///
/// # Hardware Agnosticism
///
/// This module never uses `#[cfg(target_feature)]` or hardware intrinsics.
/// - CPU stages use independent Rayon thread pools (already in-process).
/// - GPU stages reuse the existing WGPU path in `engine.rs`.
/// - If only a single slot is present, execution is identical to the original monolithic loop.
/// - Degrades gracefully on any hardware configuration.
use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Device Topology
// ─────────────────────────────────────────────────────────────────────────────

/// What kind of compute resource backs a pipeline stage.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DeviceKind {
    /// A CPU stage backed by a Rayon thread pool with `num_threads` workers.
    CpuPool { num_threads: usize },
    /// A WGPU (GPU) stage. Reuses the existing `generate_wgpu` dispatch path.
    Wgpu,
    /// Placeholder for future multi-node transport (gRPC / RDMA).
    Remote { endpoint: String },
}

impl DeviceKind {
    /// Human-readable label for logging.
    pub fn label(&self) -> String {
        match self {
            DeviceKind::CpuPool { num_threads } => format!("cpu({}t)", num_threads),
            DeviceKind::Wgpu => "wgpu".to_string(),
            DeviceKind::Remote { endpoint } => format!("remote({})", endpoint),
        }
    }
}

/// One slot in the device mesh: a compute unit with a memory budget.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceSlot {
    /// Unique identifier, e.g. `"cpu:0"`, `"cpu:1"`, `"wgpu:0"`.
    pub id: String,
    /// The kind of compute backing this slot.
    pub kind: DeviceKind,
    /// Maximum bytes this slot can hold in DRAM simultaneously.
    pub dram_budget_bytes: u64,
    /// Layer indices (0-indexed within the model) assigned to this slot.
    pub assigned_layers: Vec<usize>,
}

impl DeviceSlot {
    pub fn new(id: impl Into<String>, kind: DeviceKind, dram_budget_bytes: u64) -> Self {
        DeviceSlot {
            id: id.into(),
            kind,
            dram_budget_bytes,
            assigned_layers: Vec::new(),
        }
    }
}
use std::sync::atomic::{AtomicU32, Ordering};

pub static SYSTEM_RESOURCE_CAP: AtomicU32 = AtomicU32::new(0);

pub fn set_system_resource_cap(limit: f32) {
    SYSTEM_RESOURCE_CAP.store(limit.to_bits(), Ordering::Relaxed);
}

pub fn get_system_resource_cap() -> f32 {
    let bits = SYSTEM_RESOURCE_CAP.load(Ordering::Relaxed);
    if bits == 0 {
        if let Ok(val) = std::env::var("BRAMHA_RESOURCE_CAP") {
            val.parse::<f32>().unwrap_or(0.60)
        } else {
            0.60
        }
    } else {
        f32::from_bits(bits)
    }
}

pub fn get_system_ram_bytes() -> u64 {
    let mut physical_ram = 8 * 1024 * 1024 * 1024; // Default fallback

    #[cfg(target_os = "linux")]
    {
        unsafe {
            let pages = libc::sysconf(libc::_SC_PHYS_PAGES);
            let page_size = libc::sysconf(libc::_SC_PAGESIZE);
            if pages > 0 && page_size > 0 {
                physical_ram = (pages as u64).saturating_mul(page_size as u64);
            }
        }

        // Try reading cgroup v2 limit for the current process
        if let Ok(cgroup_content) = std::fs::read_to_string("/proc/self/cgroup") {
            for line in cgroup_content.lines() {
                // cgroup v2 format: "0::/path" or similar
                let parts: Vec<&str> = line.split(':').collect();
                if parts.len() >= 3 && (parts[0] == "0" || parts[0].is_empty()) {
                    let cg_path = parts[2].trim();
                    let memory_max_path = format!("/sys/fs/cgroup{}/memory.max", cg_path);
                    if let Ok(max_str) = std::fs::read_to_string(&memory_max_path) {
                        let max_trimmed = max_str.trim();
                        if max_trimmed != "max" && !max_trimmed.is_empty()
                            && let Ok(limit) = max_trimmed.parse::<u64>()
                                && limit > 0 {
                                    physical_ram = physical_ram.min(limit);
                                }
                    }
                }
            }
        }

        // Try reading cgroup v2 root limit
        if let Ok(max_str) = std::fs::read_to_string("/sys/fs/cgroup/memory.max") {
            let max_trimmed = max_str.trim();
            if max_trimmed != "max" && !max_trimmed.is_empty()
                && let Ok(limit) = max_trimmed.parse::<u64>()
                    && limit > 0 {
                        physical_ram = physical_ram.min(limit);
                    }
        }

        // Try reading cgroup v1 limit
        if let Ok(limit_str) =
            std::fs::read_to_string("/sys/fs/cgroup/memory/memory.limit_in_bytes")
            && let Ok(limit) = limit_str.trim().parse::<u64>() {
                // A value of 9223372036854771712 or similar means unlimited in cgroup v1
                if limit > 0 && limit < 9000000000000000000 {
                    physical_ram = physical_ram.min(limit);
                }
            }
    }

    physical_ram
}

pub fn get_capped_system_ram_bytes() -> u64 {
    let total_ram = get_system_ram_bytes();
    let cap = get_system_resource_cap();
    (total_ram as f64 * cap as f64) as u64
}

/// The full device topology — an ordered list of pipeline stages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceMesh {
    pub slots: Vec<DeviceSlot>,
}

impl DeviceMesh {
    /// Single CPU slot — always safe, equivalent to the original monolithic loop.
    pub fn single_cpu() -> Self {
        let threads = rayon::current_num_threads().max(1);
        DeviceMesh {
            slots: vec![DeviceSlot::new(
                "cpu:0",
                DeviceKind::CpuPool {
                    num_threads: threads,
                },
                get_capped_system_ram_bytes(),
            )],
        }
    }

    /// Two-stage CPU pipeline: Stage 0 handles first half of layers, Stage 1 the second half.
    /// Each stage gets its own Rayon thread pool with `threads_per_stage` workers.
    pub fn dual_cpu(threads_per_stage: usize) -> Self {
        let budget = get_capped_system_ram_bytes() / 2;
        DeviceMesh {
            slots: vec![
                DeviceSlot::new(
                    "cpu:0",
                    DeviceKind::CpuPool {
                        num_threads: threads_per_stage,
                    },
                    budget,
                ),
                DeviceSlot::new(
                    "cpu:1",
                    DeviceKind::CpuPool {
                        num_threads: threads_per_stage,
                    },
                    budget,
                ),
            ],
        }
    }

    /// CPU Stage 0 + WGPU Stage 1: CPU handles early layers (embedding + attention),
    /// GPU handles later layers (MLP-heavy) and the LM head.
    pub fn cpu_gpu(cpu_threads: usize) -> Self {
        DeviceMesh {
            slots: vec![
                DeviceSlot::new(
                    "cpu:0",
                    DeviceKind::CpuPool {
                        num_threads: cpu_threads,
                    },
                    get_capped_system_ram_bytes(),
                ),
                DeviceSlot::new("wgpu:0", DeviceKind::Wgpu, 4 * 1024 * 1024 * 1024),
            ],
        }
    }

    /// Build a mesh from the `BRAMHA_PIPELINE_STAGES` environment variable.
    /// Falls back to single_cpu() if the variable is absent or ≤ 1.
    pub fn from_env() -> Self {
        let stages = std::env::var("BRAMHA_PIPELINE_STAGES")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(1);

        let use_wgpu = std::env::var("BRAMHA_PIPELINE_WGPU").is_ok();
        let threads = (rayon::current_num_threads() / stages.max(1)).max(1);

        if stages <= 1 {
            return Self::single_cpu();
        }

        if use_wgpu && stages == 2 {
            return Self::cpu_gpu(threads * 2);
        }

        let budget = get_capped_system_ram_bytes() / stages as u64;
        let slots = (0..stages)
            .map(|i| {
                DeviceSlot::new(
                    format!("cpu:{}", i),
                    DeviceKind::CpuPool {
                        num_threads: threads,
                    },
                    budget,
                )
            })
            .collect();
        DeviceMesh { slots }
    }

    pub fn num_slots(&self) -> usize {
        self.slots.len()
    }

    /// Find which slot owns a given layer index.
    pub fn slot_for_layer(&self, layer_idx: usize) -> Option<usize> {
        for (si, slot) in self.slots.iter().enumerate() {
            if slot.assigned_layers.contains(&layer_idx) {
                return Some(si);
            }
        }
        None
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Sharding Planner
// ─────────────────────────────────────────────────────────────────────────────

/// Maps each model layer index to a slot in the DeviceMesh.
#[derive(Debug, Clone)]
pub struct LayerAssignment {
    /// layer_idx → slot_id
    pub map: HashMap<usize, String>,
    /// Ordered list of (slot_id, layer_range_start..layer_range_end) for the executor.
    pub stages: Vec<(String, std::ops::Range<usize>)>,
}

impl LayerAssignment {
    pub fn slot_for(&self, layer_idx: usize) -> Option<&str> {
        self.map.get(&layer_idx).map(|s| s.as_str())
    }
}

/// Plans which layers go to which device slot.
///
/// Strategy: greedy fill — layers are assigned in order (0, 1, 2, …) to slots
/// until the slot's DRAM budget is exhausted, then overflow to the next slot.
/// This is identical to a DB page replacement policy (LRU fill).
pub struct ShardingPlanner;

impl ShardingPlanner {
    /// Assign `num_layers` layers across the slots of `mesh`.
    ///
    /// `layer_sizes_bytes`: optional per-layer byte sizes from the manifest.
    /// If absent, assumes uniform distribution.
    pub fn plan(
        mesh: &mut DeviceMesh,
        num_layers: usize,
        layer_sizes_bytes: &[u64],
    ) -> LayerAssignment {
        // Clear any previous assignment
        for slot in mesh.slots.iter_mut() {
            slot.assigned_layers.clear();
        }

        let num_slots = mesh.slots.len();
        if num_slots == 0 || num_layers == 0 {
            return LayerAssignment {
                map: HashMap::new(),
                stages: Vec::new(),
            };
        }

        // If only one slot, trivially assign everything to it
        if num_slots == 1 {
            mesh.slots[0].assigned_layers = (0..num_layers).collect();
            let mut map = HashMap::new();
            for l in 0..num_layers {
                map.insert(l, mesh.slots[0].id.clone());
            }
            let stages = vec![(mesh.slots[0].id.clone(), 0..num_layers)];
            return LayerAssignment { map, stages };
        }

        // Greedy fill: walk layers in order, fill current slot until budget exhausted
        let avg_size = if layer_sizes_bytes.is_empty() {
            // Assume 500 MB per layer as a safe default for 7B-class models
            500 * 1024 * 1024u64
        } else {
            layer_sizes_bytes.iter().sum::<u64>() / layer_sizes_bytes.len().max(1) as u64
        };

        let mut slot_idx = 0;
        let mut slot_used = 0u64;
        let mut map = HashMap::new();
        let mut stage_starts = vec![0usize; num_slots];
        let mut slot_layer_counts = vec![0usize; num_slots];

        for layer_idx in 0..num_layers {
            let size = layer_sizes_bytes
                .get(layer_idx)
                .copied()
                .unwrap_or(avg_size);

            // Move to next slot if current is full (and there is a next slot)
            if slot_idx + 1 < num_slots && slot_used + size > mesh.slots[slot_idx].dram_budget_bytes
            {
                stage_starts[slot_idx + 1] = layer_idx;
                slot_idx += 1;
                slot_used = 0;
            }

            mesh.slots[slot_idx].assigned_layers.push(layer_idx);
            map.insert(layer_idx, mesh.slots[slot_idx].id.clone());
            slot_used += size;
            slot_layer_counts[slot_idx] += 1;
        }

        // Build ordered stage list (contiguous ranges for the executor)
        let mut stages = Vec::new();
        let mut cursor = 0usize;
        for (si, slot) in mesh.slots.iter().enumerate() {
            let count = slot_layer_counts[si];
            if count > 0 {
                stages.push((slot.id.clone(), cursor..cursor + count));
                cursor += count;
            }
        }

        LayerAssignment { map, stages }
    }

    /// Convenience: plan using manifest layer sizes (reads `stored_bytes` per layer).
    pub fn plan_from_manifest(
        mesh: &mut DeviceMesh,
        num_layers: usize,
        manifest: &crate::storage::storage_manifest::ModelManifest,
    ) -> LayerAssignment {
        // Collect per-layer byte sizes in layer order
        let mut sizes: Vec<(usize, u64)> = manifest
            .layers
            .values()
            .filter_map(|meta| {
                // Extract layer index from names like "model.layers.5.self_attn..."
                let parts: Vec<&str> = meta.layer_id.split('.').collect();
                if parts.len() >= 3 && parts[1] == "layers" {
                    parts[2]
                        .parse::<usize>()
                        .ok()
                        .map(|idx| (idx, meta.stored_bytes))
                } else {
                    None
                }
            })
            .collect();
        sizes.sort_by_key(|&(idx, _)| idx);
        sizes.dedup_by_key(|&mut (idx, _)| idx);

        let layer_sizes: Vec<u64> = sizes.into_iter().map(|(_, s)| s).collect();
        Self::plan(mesh, num_layers, &layer_sizes)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Pipeline Executor
// ─────────────────────────────────────────────────────────────────────────────

/// Drives the full token generation loop across a multi-stage `DeviceMesh`.
///
/// Each stage executes its assigned layer range and passes the resulting
/// activation vector to the next stage via `Arc<Vec<f32>>`.
///
/// # Activation Streaming
///
/// Between in-process stages, activations are transferred as `Arc<Vec<f32>>` —
/// a reference-counted pointer, so the handoff is O(1) and zero-copy.
/// For `Remote` stages this would be replaced by serialization + network send,
/// but the trait interface is identical.
pub struct PipelineExecutor {
    pub mesh: DeviceMesh,
    pub assignment: LayerAssignment,
}

impl PipelineExecutor {
    /// Build from environment variables (single_cpu by default).
    pub fn from_env(num_layers: usize) -> Self {
        let mut mesh = DeviceMesh::from_env();
        let assignment = ShardingPlanner::plan(&mut mesh, num_layers, &[]);
        PipelineExecutor { mesh, assignment }
    }

    /// Build with an explicit mesh and pre-computed layer sizes.
    pub fn new(mut mesh: DeviceMesh, num_layers: usize, layer_sizes: &[u64]) -> Self {
        let assignment = ShardingPlanner::plan(&mut mesh, num_layers, layer_sizes);
        PipelineExecutor { mesh, assignment }
    }

    /// Returns a summary of the pipeline configuration for logging.
    pub fn describe(&self) -> String {
        let parts: Vec<String> = self
            .assignment
            .stages
            .iter()
            .map(|(slot_id, range)| format!("{}: layers {}..{}", slot_id, range.start, range.end))
            .collect();
        format!("Pipeline[{}]", parts.join(" | "))
    }

    /// Execute one token decode step across the pipeline.
    ///
    /// `run_layer_range` is a callback (provided by `generate_cpu`) that executes
    /// a contiguous range of transformer layers on the current activation and
    /// returns the updated activation. This keeps `PipelineExecutor` free of any
    /// `LayerWeights` lifetime complexity.
    ///
    /// ```text
    /// activation → [Stage 0: layers 0..N/2] → activation → [Stage 1: layers N/2..N] → activation
    /// ```
    pub fn execute_step<F>(
        &self,
        initial_activation: Vec<f32>,
        run_layer_range: F,
    ) -> Result<Vec<f32>, String>
    where
        F: Fn(Vec<f32>, &std::ops::Range<usize>, &str) -> Result<Vec<f32>, String>,
    {
        let mut activation = initial_activation;
        for (slot_id, layer_range) in &self.assignment.stages {
            activation = run_layer_range(activation, layer_range, slot_id)?;
        }
        Ok(activation)
    }

    /// Returns `true` if the mesh has more than one stage (real pipeline).
    pub fn is_multi_stage(&self) -> bool {
        self.assignment.stages.len() > 1
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify single-slot mesh assigns all layers to slot 0.
    #[test]
    fn test_device_mesh_single_cpu() {
        let mut mesh = DeviceMesh::single_cpu();
        assert_eq!(mesh.num_slots(), 1);
        assert_eq!(mesh.slots[0].id, "cpu:0");

        let assignment = ShardingPlanner::plan(&mut mesh, 22, &[]);
        assert_eq!(assignment.stages.len(), 1);
        assert_eq!(assignment.stages[0].1, 0..22);
        assert_eq!(assignment.map.len(), 22);
        for l in 0..22 {
            assert_eq!(assignment.slot_for(l), Some("cpu:0"));
        }
    }

    /// Verify layers split across two CPU slots when budget is tight.
    #[test]
    fn test_sharding_planner_two_stage() {
        // Budget: 6 layers per slot × 500MB each = 3GB per slot
        let budget_per_slot = 6 * 500 * 1024 * 1024u64;
        let mut mesh = DeviceMesh {
            slots: vec![
                DeviceSlot::new(
                    "cpu:0",
                    DeviceKind::CpuPool { num_threads: 4 },
                    budget_per_slot,
                ),
                DeviceSlot::new(
                    "cpu:1",
                    DeviceKind::CpuPool { num_threads: 4 },
                    budget_per_slot,
                ),
            ],
        };
        // 12 layers, 500MB each
        let sizes = vec![500 * 1024 * 1024u64; 12];
        let assignment = ShardingPlanner::plan(&mut mesh, 12, &sizes);

        assert_eq!(assignment.stages.len(), 2);
        // Slot 0 should hold 6 layers (fills budget exactly), slot 1 the rest
        assert_eq!(assignment.stages[0].1, 0..6);
        assert_eq!(assignment.stages[1].1, 6..12);
        for l in 0..6 {
            assert_eq!(assignment.slot_for(l), Some("cpu:0"));
        }
        for l in 6..12 {
            assert_eq!(assignment.slot_for(l), Some("cpu:1"));
        }
    }

    /// PipelineExecutor with a single stage should pass activation through unchanged.
    #[test]
    fn test_pipeline_executor_single_stage() {
        let hidden_size = 64usize;
        let mut mesh = DeviceMesh::single_cpu();
        let num_layers = 4;
        let assignment = ShardingPlanner::plan(&mut mesh, num_layers, &[]);
        let executor = PipelineExecutor { mesh, assignment };

        // The callback just scales the activation by 1.0 (identity for each layer in range)
        let initial: Vec<f32> = (0..hidden_size).map(|i| i as f32).collect();
        let result = executor.execute_step(initial.clone(), |act, range, _slot_id| {
            // Simulate: each layer adds 1.0 to every element
            let mut out = act;
            for _layer in range.clone() {
                for v in out.iter_mut() {
                    *v += 1.0;
                }
            }
            Ok(out)
        });

        assert!(result.is_ok());
        let out = result.unwrap();
        assert_eq!(out.len(), hidden_size);
        // Each of `num_layers` layers added 1.0, so each element = initial + num_layers
        for (i, &v) in out.iter().enumerate() {
            assert!(
                (v - (i as f32 + num_layers as f32)).abs() < 1e-5,
                "Mismatch at index {}: expected {}, got {}",
                i,
                i as f32 + num_layers as f32,
                v
            );
        }
    }

    /// Two-stage pipeline produces the same final activation as a single-stage reference.
    #[test]
    fn test_pipeline_executor_two_stage_matches_reference() {
        let hidden_size = 32usize;
        let num_layers = 8usize;

        // ── Single-stage reference ──────────────────────────────────────────
        let mut single_mesh = DeviceMesh::single_cpu();
        let single_assignment = ShardingPlanner::plan(&mut single_mesh, num_layers, &[]);
        let single_executor = PipelineExecutor {
            mesh: single_mesh,
            assignment: single_assignment,
        };

        let initial: Vec<f32> = (0..hidden_size).map(|i| i as f32 * 0.5).collect();

        let layer_fn = |act: Vec<f32>, range: &std::ops::Range<usize>, _slot: &str| {
            let mut out = act;
            for layer_idx in range.clone() {
                // Deterministic: multiply element j by (1.0 + 0.01 * layer_idx)
                let scale = 1.0 + 0.01 * layer_idx as f32;
                for v in out.iter_mut() {
                    *v *= scale;
                }
            }
            Ok(out)
        };

        let reference = single_executor
            .execute_step(initial.clone(), layer_fn)
            .unwrap();

        // ── Two-stage pipeline ──────────────────────────────────────────────
        // Budget forces split at layer 4 (half)
        let half_budget = 4 * 500 * 1024 * 1024u64;
        let mut dual_mesh = DeviceMesh {
            slots: vec![
                DeviceSlot::new("cpu:0", DeviceKind::CpuPool { num_threads: 2 }, half_budget),
                DeviceSlot::new("cpu:1", DeviceKind::CpuPool { num_threads: 2 }, half_budget),
            ],
        };
        let sizes = vec![500 * 1024 * 1024u64; num_layers];
        let dual_assignment = ShardingPlanner::plan(&mut dual_mesh, num_layers, &sizes);
        let dual_executor = PipelineExecutor {
            mesh: dual_mesh,
            assignment: dual_assignment,
        };

        assert!(dual_executor.is_multi_stage(), "Expected 2 stages");

        let dual_result = dual_executor
            .execute_step(initial.clone(), |act, range, _slot| {
                let mut out = act;
                for layer_idx in range.clone() {
                    let scale = 1.0 + 0.01 * layer_idx as f32;
                    for v in out.iter_mut() {
                        *v *= scale;
                    }
                }
                Ok(out)
            })
            .unwrap();

        // Both must produce identical activations (pipeline is mathematically equivalent)
        assert_eq!(reference.len(), dual_result.len());
        for (i, (&ref_v, &dual_v)) in reference.iter().zip(dual_result.iter()).enumerate() {
            assert!(
                (ref_v - dual_v).abs() < 1e-4,
                "Mismatch at index {}: reference={}, pipeline={}",
                i,
                ref_v,
                dual_v
            );
        }
    }
}
