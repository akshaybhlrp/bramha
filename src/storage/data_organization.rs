//! # Database-Native Data Organization for Bramha
//!
//! Implements four data organization strategies for neural database workloads:
//!
//! - **Partitioning**: Split collections/data by key range, time, or size thresholds
//! - **Sharding**: Distribute data across directories, files, or storage tiers
//! - **MaterializedView**: Precomputed query results stored as first-class objects
//! - **Denormalization**: Flatten related metadata into vector storage for zero-join reads
//!
//! All strategies are designed for zero-copy, cache-friendly access patterns.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

// ─── Partitioning ───────────────────────────────────────────────────────────

/// Strategy for partitioning data.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PartitionStrategy {
    /// Range partitioning by a key (e.g., timestamp ranges, score ranges)
    Range {
        /// The field name to partition on
        field: String,
        /// Partition boundaries
        boundaries: Vec<PartitionBoundary>,
    },
    /// List partitioning by discrete values (e.g., model_name, collection)
    List {
        field: String,
        /// Each partition maps to a set of values
        partitions: Vec<ListPartition>,
    },
    /// Hash partitioning for uniform distribution
    Hash {
        field: String,
        num_partitions: usize,
    },
    /// Time-based partitioning (daily, weekly, monthly)
    Time {
        field: String,
        interval: TimeInterval,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PartitionBoundary {
    Inclusive(BTreeKey),
    Exclusive(BTreeKey),
    Max,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ListPartition {
    pub name: String,
    pub values: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TimeInterval {
    Hourly,
    Daily,
    Weekly,
    Monthly,
    Yearly,
}

/// A single partition holding a subset of data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Partition {
    pub name: String,
    pub description: String,
    pub strategy: PartitionStrategy,
    /// Path to the partition's data directory
    pub data_path: PathBuf,
    /// Number of records in this partition
    pub record_count: usize,
    /// Approximate size in bytes
    pub size_bytes: u64,
    /// When this partition was created
    pub created_at: u64,
}

/// Manages partitioning across a collection or database.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionManager {
    partitions: Vec<Partition>,
    default_strategy: PartitionStrategy,
}

impl PartitionManager {
    pub fn new(strategy: PartitionStrategy) -> Self {
        PartitionManager {
            partitions: Vec::new(),
            default_strategy: strategy,
        }
    }

    pub fn add_partition(&mut self, partition: Partition) {
        self.partitions.push(partition);
    }

    pub fn partitions(&self) -> &[Partition] {
        &self.partitions
    }

    pub fn partitions_mut(&mut self) -> &mut Vec<Partition> {
        &mut self.partitions
    }

    /// Find which partition a key belongs to.
    pub fn locate_partition(&self, key: &BTreeKey) -> Option<&Partition> {
        match &self.default_strategy {
            PartitionStrategy::Range { boundaries, .. } => {
                for (i, partition) in self.partitions.iter().enumerate() {
                    let boundary = &boundaries[i.min(boundaries.len().saturating_sub(1))];
                    match boundary {
                        PartitionBoundary::Inclusive(b) => {
                            if key <= b {
                                return Some(partition);
                            }
                        }
                        PartitionBoundary::Exclusive(b) => {
                            if key < b {
                                return Some(partition);
                            }
                        }
                        PartitionBoundary::Max => {
                            return Some(partition);
                        }
                    }
                }
                self.partitions.last()
            }
            PartitionStrategy::List { partitions, .. } => {
                // For list partitioning, we need the key as a string
                let key_str = match key {
                    BTreeKey::String(s) => s.as_str(),
                    _ => return None,
                };
                for partition in &self.partitions {
                    if let Some(list_part) = partitions.iter().find(|lp| lp.name == partition.name)
                        && list_part.values.iter().any(|v| v == key_str) {
                            return Some(partition);
                        }
                }
                None
            }
            PartitionStrategy::Hash { num_partitions, .. } => {
                let hash = match key {
                    BTreeKey::String(s) => {
                        let mut h = 0u64;
                        for b in s.bytes() {
                            h = h.wrapping_mul(31).wrapping_add(b as u64);
                        }
                        h
                    }
                    BTreeKey::Integer(i) => i.wrapping_abs() as u64,
                    BTreeKey::Float(bits) => *bits,
                };
                let idx = (hash % *num_partitions as u64) as usize;
                self.partitions.get(idx)
            }
            PartitionStrategy::Time { .. } => {
                // Time partitioning: find the partition whose range contains the key
                self.partitions.last()
            }
        }
    }

    /// Get total record count across all partitions.
    pub fn total_records(&self) -> usize {
        self.partitions.iter().map(|p| p.record_count).sum()
    }

    /// Get total size across all partitions.
    pub fn total_size(&self) -> u64 {
        self.partitions.iter().map(|p| p.size_bytes).sum()
    }

    pub fn strategy(&self) -> &PartitionStrategy {
        &self.default_strategy
    }
}

// ─── Sharding ───────────────────────────────────────────────────────────────

/// Strategy for distributing data across shards.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ShardStrategy {
    /// Hash-based sharding (consistent hashing)
    ConsistentHash {
        num_shards: usize,
        virtual_nodes: usize,
    },
    /// Range-based sharding
    Range { shard_key: String },
    /// Directory-based sharding (each shard is a directory)
    Directory { base_path: PathBuf },
}

/// A single shard holding a subset of data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Shard {
    pub id: usize,
    pub name: String,
    pub data_path: PathBuf,
    pub record_count: usize,
    pub size_bytes: u64,
    pub is_active: bool,
}

/// Manages sharding across storage tiers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardManager {
    shards: Vec<Shard>,
    strategy: ShardStrategy,
}

impl ShardManager {
    pub fn new(strategy: ShardStrategy) -> Self {
        ShardManager {
            shards: Vec::new(),
            strategy,
        }
    }

    pub fn add_shard(&mut self, shard: Shard) {
        self.shards.push(shard);
    }

    pub fn shards(&self) -> &[Shard] {
        &self.shards
    }

    pub fn shards_mut(&mut self) -> &mut Vec<Shard> {
        &mut self.shards
    }

    /// Determine which shard a key belongs to using consistent hashing.
    pub fn locate_shard(&self, key: &str) -> Option<&Shard> {
        if self.shards.is_empty() {
            return None;
        }

        match &self.strategy {
            ShardStrategy::ConsistentHash {
                num_shards,
                virtual_nodes,
            } => {
                let shard_len = self.shards.len();
                let effective_shards = num_shards.min(&shard_len);
                if *effective_shards == 0 {
                    return self.shards.first();
                }

                // Simple hash ring: hash the key, mod by effective shards
                let mut h: u64 = 0;
                for b in key.bytes() {
                    h = h.wrapping_mul(31).wrapping_add(b as u64);
                }
                // Add virtual node influence
                let vn = *virtual_nodes;
                let mut best_shard = 0usize;
                let mut best_hash = u64::MAX;
                for s in 0..*effective_shards {
                    for v in 0..vn {
                        let vh = h.wrapping_mul(17).wrapping_add((s * vn + v) as u64);
                        let dist = vh.wrapping_sub(h);
                        if dist < best_hash {
                            best_hash = dist;
                            best_shard = s;
                        }
                    }
                }
                self.shards.get(best_shard)
            }
            ShardStrategy::Range { .. } => {
                // Simple round-robin for range-based
                let mut h: u64 = 0;
                for b in key.bytes() {
                    h = h.wrapping_mul(31).wrapping_add(b as u64);
                }
                let idx = (h % self.shards.len() as u64) as usize;
                self.shards.get(idx)
            }
            ShardStrategy::Directory { .. } => {
                // Directory-based: hash to a shard directory
                let mut h: u64 = 0;
                for b in key.bytes() {
                    h = h.wrapping_mul(31).wrapping_add(b as u64);
                }
                let idx = (h % self.shards.len() as u64) as usize;
                self.shards.get(idx)
            }
        }
    }

    /// Get total stats across all shards.
    pub fn total_records(&self) -> usize {
        self.shards.iter().map(|s| s.record_count).sum()
    }

    pub fn total_size(&self) -> u64 {
        self.shards.iter().map(|s| s.size_bytes).sum()
    }

    pub fn strategy(&self) -> &ShardStrategy {
        &self.strategy
    }
}

// ─── Materialized View ──────────────────────────────────────────────────────

/// A precomputed query result stored as a first-class database object.
/// Extends the existing `ActivationMaterializedView` with full query support.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterializedView {
    pub name: String,
    /// The query this view was built from (for refresh/invalidation)
    pub source_query: String,
    /// The precomputed result data
    pub data: Vec<serde_json::Value>,
    /// Schema of the result columns
    pub schema: Vec<ViewColumn>,
    /// When this view was last refreshed
    pub last_refreshed: u64,
    /// How often to auto-refresh (0 = manual only)
    pub refresh_interval_secs: u64,
    /// Size of the materialized data in bytes
    pub size_bytes: u64,
    /// Whether this view is stale and needs refresh
    pub is_stale: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViewColumn {
    pub name: String,
    pub data_type: String,
}

/// Manages materialized views for the database.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterializedViewManager {
    views: HashMap<String, MaterializedView>,
}

impl MaterializedViewManager {
    pub fn new() -> Self {
        MaterializedViewManager {
            views: HashMap::new(),
        }
    }

    /// Create a new materialized view.
    pub fn create(
        &mut self,
        name: impl Into<String>,
        source_query: impl Into<String>,
        schema: Vec<ViewColumn>,
        refresh_interval_secs: u64,
    ) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let name: String = name.into();
        self.views.insert(
            name.clone(),
            MaterializedView {
                name,
                source_query: source_query.into(),
                data: Vec::new(),
                schema,
                last_refreshed: now,
                refresh_interval_secs,
                size_bytes: 0,
                is_stale: false,
            },
        );
    }

    /// Refresh a view with new data.
    pub fn refresh(&mut self, name: &str, data: Vec<serde_json::Value>) -> Option<()> {
        let view = self.views.get_mut(name)?;
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        view.data = data;
        view.last_refreshed = now;
        view.is_stale = false;
        // Estimate size
        view.size_bytes = serde_json::to_vec(&view.data).unwrap_or_default().len() as u64;
        Some(())
    }

    /// Get a view's data.
    pub fn get(&self, name: &str) -> Option<&MaterializedView> {
        self.views.get(name)
    }

    /// Check if a view needs refresh.
    pub fn needs_refresh(&self, name: &str) -> bool {
        if let Some(view) = self.views.get(name) {
            if view.is_stale {
                return true;
            }
            if view.refresh_interval_secs == 0 {
                return false; // manual only
            }
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            now - view.last_refreshed > view.refresh_interval_secs
        } else {
            false
        }
    }

    /// Mark a view as stale (e.g., when source data changes).
    pub fn mark_stale(&mut self, name: &str) {
        if let Some(view) = self.views.get_mut(name) {
            view.is_stale = true;
        }
    }

    /// Drop a materialized view.
    pub fn drop(&mut self, name: &str) -> Option<MaterializedView> {
        self.views.remove(name)
    }

    /// List all views.
    pub fn list(&self) -> Vec<&MaterializedView> {
        self.views.values().collect()
    }

    pub fn len(&self) -> usize {
        self.views.len()
    }

    pub fn is_empty(&self) -> bool {
        self.views.is_empty()
    }
}

impl Default for MaterializedViewManager {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Denormalization ────────────────────────────────────────────────────────

/// A denormalized field stored inline with the primary record.
/// Eliminates the need for JOINs by flattening related data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DenormalizedField {
    pub field_name: String,
    pub source_path: String, // e.g., "metadata.author.name"
    pub value: serde_json::Value,
}

/// Configuration for denormalization rules.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DenormalizationRule {
    pub name: String,
    /// The source field path in the related entity
    pub source_field: String,
    /// The target field name in the denormalized record
    pub target_field: String,
    /// How to handle updates to the source (cascade, nullify, snapshot)
    pub update_strategy: DenormalizeUpdateStrategy,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DenormalizeUpdateStrategy {
    /// Recompute on every source change
    Cascade,
    /// Set to null when source is deleted
    Nullify,
    /// Keep the value as a snapshot (never update)
    Snapshot,
}

/// Manages denormalization rules and applies them to records.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DenormalizationManager {
    rules: Vec<DenormalizationRule>,
}

impl DenormalizationManager {
    pub fn new() -> Self {
        DenormalizationManager { rules: Vec::new() }
    }

    pub fn add_rule(&mut self, rule: DenormalizationRule) {
        self.rules.push(rule);
    }

    pub fn rules(&self) -> &[DenormalizationRule] {
        &self.rules
    }

    /// Apply denormalization rules to a record, flattening related data.
    pub fn apply(
        &self,
        record: &mut serde_json::Value,
        related_data: &HashMap<String, serde_json::Value>,
    ) {
        for rule in &self.rules {
            if let Some(source_value) = related_data.get(&rule.source_field) {
                if let Some(target) = record.pointer_mut(&format!("/{}", rule.target_field)) {
                    *target = source_value.clone();
                } else if let Some(obj) = record.as_object_mut() {
                    obj.insert(rule.target_field.clone(), source_value.clone());
                }
            }
        }
    }

    pub fn len(&self) -> usize {
        self.rules.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }
}

impl Default for DenormalizationManager {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Data Organization Manager ──────────────────────────────────────────────

/// Central coordinator for all data organization strategies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataOrganizationManager {
    pub partition_manager: PartitionManager,
    pub shard_manager: ShardManager,
    pub view_manager: MaterializedViewManager,
    pub denorm_manager: DenormalizationManager,
}

impl DataOrganizationManager {
    pub fn new(partition_strategy: PartitionStrategy, shard_strategy: ShardStrategy) -> Self {
        DataOrganizationManager {
            partition_manager: PartitionManager::new(partition_strategy),
            shard_manager: ShardManager::new(shard_strategy),
            view_manager: MaterializedViewManager::new(),
            denorm_manager: DenormalizationManager::new(),
        }
    }

    /// Report comprehensive organization statistics.
    pub fn report(&self) -> OrganizationReport {
        OrganizationReport {
            num_partitions: self.partition_manager.partitions().len(),
            num_shards: self.shard_manager.shards().len(),
            num_materialized_views: self.view_manager.len(),
            num_denorm_rules: self.denorm_manager.len(),
            total_partition_records: self.partition_manager.total_records(),
            total_shard_records: self.shard_manager.total_records(),
            partition_strategy: format!("{:?}", self.partition_manager.strategy()),
            shard_strategy: format!("{:?}", self.shard_manager.strategy()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrganizationReport {
    pub num_partitions: usize,
    pub num_shards: usize,
    pub num_materialized_views: usize,
    pub num_denorm_rules: usize,
    pub total_partition_records: usize,
    pub total_shard_records: usize,
    pub partition_strategy: String,
    pub shard_strategy: String,
}

// Re-export BTreeKey from indexing module for use in partitioning
use crate::storage::indexing::BTreeKey;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partition_range_lookup() {
        let strategy = PartitionStrategy::Range {
            field: "score".into(),
            boundaries: vec![
                PartitionBoundary::Inclusive(BTreeKey::from_f64(0.5)),
                PartitionBoundary::Inclusive(BTreeKey::from_f64(0.8)),
                PartitionBoundary::Max,
            ],
        };
        let mut pm = PartitionManager::new(strategy);
        pm.add_partition(Partition {
            name: "low".into(),
            description: "scores <= 0.5".into(),
            strategy: PartitionStrategy::Range {
                field: "score".into(),
                boundaries: vec![],
            },
            data_path: PathBuf::from("/data/low"),
            record_count: 100,
            size_bytes: 1024,
            created_at: 0,
        });
        pm.add_partition(Partition {
            name: "medium".into(),
            description: "0.5 < scores <= 0.8".into(),
            strategy: PartitionStrategy::Range {
                field: "score".into(),
                boundaries: vec![],
            },
            data_path: PathBuf::from("/data/medium"),
            record_count: 200,
            size_bytes: 2048,
            created_at: 0,
        });
        pm.add_partition(Partition {
            name: "high".into(),
            description: "scores > 0.8".into(),
            strategy: PartitionStrategy::Range {
                field: "score".into(),
                boundaries: vec![],
            },
            data_path: PathBuf::from("/data/high"),
            record_count: 50,
            size_bytes: 512,
            created_at: 0,
        });

        let p = pm.locate_partition(&BTreeKey::from_f64(0.3));
        assert!(p.is_some());
        assert_eq!(p.unwrap().name, "low");

        let p = pm.locate_partition(&BTreeKey::from_f64(0.9));
        assert!(p.is_some());
        assert_eq!(p.unwrap().name, "high");
    }

    #[test]
    fn test_shard_consistent_hash() {
        let strategy = ShardStrategy::ConsistentHash {
            num_shards: 3,
            virtual_nodes: 10,
        };
        let mut sm = ShardManager::new(strategy);
        for i in 0..3 {
            sm.add_shard(Shard {
                id: i,
                name: format!("shard_{}", i),
                data_path: PathBuf::from(format!("/data/shard_{}", i)),
                record_count: 0,
                size_bytes: 0,
                is_active: true,
            });
        }

        let s1 = sm.locate_shard("tensor_001").map(|s| s.id);
        let s2 = sm.locate_shard("tensor_002").map(|s| s.id);
        let s3 = sm.locate_shard("tensor_001").map(|s| s.id);

        // Same key always goes to same shard
        assert_eq!(s1, s3);
        // Different keys may go to different shards
        assert!(s1.is_some());
        assert!(s2.is_some());
    }

    #[test]
    fn test_materialized_view_lifecycle() {
        let mut vm = MaterializedViewManager::new();
        vm.create(
            "top_scores",
            "SELECT * FROM vectors ORDER BY score DESC LIMIT 10",
            vec![
                ViewColumn {
                    name: "id".into(),
                    data_type: "string".into(),
                },
                ViewColumn {
                    name: "score".into(),
                    data_type: "float".into(),
                },
            ],
            3600,
        );

        assert!(vm.get("top_scores").is_some());
        assert!(!vm.needs_refresh("top_scores"));

        vm.refresh(
            "top_scores",
            vec![
                serde_json::json!({"id": "doc1", "score": 0.99}),
                serde_json::json!({"id": "doc2", "score": 0.95}),
            ],
        )
        .unwrap();

        let view = vm.get("top_scores").unwrap();
        assert_eq!(view.data.len(), 2);
        assert!(!view.is_stale);

        vm.mark_stale("top_scores");
        assert!(vm.needs_refresh("top_scores"));
    }

    #[test]
    fn test_denormalization_apply() {
        let mut dm = DenormalizationManager::new();
        dm.add_rule(DenormalizationRule {
            name: "author_name".into(),
            source_field: "author.name".into(),
            target_field: "author_name".into(),
            update_strategy: DenormalizeUpdateStrategy::Cascade,
        });

        let mut record = serde_json::json!({"id": "doc1", "title": "Hello"});
        let mut related = HashMap::new();
        related.insert("author.name".to_string(), serde_json::json!("Alice"));

        dm.apply(&mut record, &related);
        assert_eq!(record["author_name"], serde_json::json!("Alice"));
    }

    #[test]
    fn test_data_organization_manager_report() {
        let dom = DataOrganizationManager::new(
            PartitionStrategy::Hash {
                field: "id".into(),
                num_partitions: 4,
            },
            ShardStrategy::ConsistentHash {
                num_shards: 3,
                virtual_nodes: 10,
            },
        );
        let report = dom.report();
        assert_eq!(report.num_partitions, 0);
        assert_eq!(report.num_shards, 0);
    }
}
