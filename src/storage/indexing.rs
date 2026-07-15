//! # Database-Native Indexing Techniques for Bramha
//!
//! Implements four indexing strategies adapted for neural database workloads:
//!
//! - **BTreeIndex**: Range-scanable balanced tree for metadata fields (timestamps, scores, sequence positions)
//! - **HashIndex**: O(1) exact-match lookup for tensor IDs, document keys, and cache hashes
//! - **CompositeIndex**: Multi-column index for common query patterns (model+layer, collection+timestamp)
//! - **CoveringIndex**: Index that stores all queried columns, eliminating table lookups for hot paths
//!
//! All indexes are thread-safe, serializable, and designed for zero-copy reads.

use std::collections::{BTreeMap, HashMap};
use serde::{Serialize, Deserialize};

// ─── B-Tree Index ───────────────────────────────────────────────────────────

/// A B-Tree index for range-scanable metadata fields.
/// Supports exact lookup, range queries (>, <, BETWEEN), and prefix scans.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BTreeIndex {
    name: String,
    /// The metadata field path this index is built on (e.g., "timestamp", "score", "sequence_pos")
    key_field: String,
    /// Internal BTreeMap: key -> sorted set of record IDs
    entries: BTreeMap<BTreeKey, Vec<String>>,
    /// Whether this index supports unique constraints
    unique: bool,
}

/// A key in the B-Tree index — supports numeric, string, and composite comparisons.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, PartialOrd, Ord, Eq)]
pub enum BTreeKey {
    String(String),
    Integer(i64),
    Float(u64), // stored as sortable bits for total order
}

impl BTreeKey {
    pub fn from_f64(val: f64) -> Self {
        BTreeKey::Float(val.to_bits())
    }

    pub fn to_f64(&self) -> Option<f64> {
        match self {
            BTreeKey::Float(bits) => Some(f64::from_bits(*bits)),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            BTreeKey::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            BTreeKey::Integer(i) => Some(*i),
            _ => None,
        }
    }
}

impl From<String> for BTreeKey {
    fn from(s: String) -> Self {
        BTreeKey::String(s)
    }
}

impl From<i64> for BTreeKey {
    fn from(i: i64) -> Self {
        BTreeKey::Integer(i)
    }
}

impl From<f64> for BTreeKey {
    fn from(f: f64) -> Self {
        BTreeKey::from_f64(f)
    }
}

impl BTreeIndex {
    pub fn new(name: impl Into<String>, key_field: impl Into<String>, unique: bool) -> Self {
        BTreeIndex {
            name: name.into(),
            key_field: key_field.into(),
            entries: BTreeMap::new(),
            unique,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn key_field(&self) -> &str {
        &self.key_field
    }

    /// Insert a record ID under the given key.
    pub fn insert(&mut self, key: BTreeKey, record_id: String) -> Result<(), String> {
        if self.unique {
            // Check for duplicate key
            if self.entries.contains_key(&key) {
                return Err(format!(
                    "BTreeIndex '{}': duplicate key {:?} violates unique constraint",
                    self.name, key
                ));
            }
            self.entries.insert(key, vec![record_id]);
        } else {
            self.entries.entry(key).or_default().push(record_id);
        }
        Ok(())
    }

    /// Remove a record ID from the index.
    pub fn remove(&mut self, key: &BTreeKey, record_id: &str) -> bool {
        if let Some(ids) = self.entries.get_mut(key) {
            ids.retain(|id| id != record_id);
            if ids.is_empty() {
                self.entries.remove(key);
            }
            return true;
        }
        false
    }

    /// Exact lookup — returns all record IDs for the given key.
    pub fn get(&self, key: &BTreeKey) -> Option<&Vec<String>> {
        self.entries.get(key)
    }

    /// Range query: all records with keys in [start, end] (inclusive).
    pub fn range(&self, start: &BTreeKey, end: &BTreeKey) -> Vec<&String> {
        let mut results = Vec::new();
        for (_key, ids) in self.entries.range(start..=end) {
            results.extend(ids.iter());
        }
        results
    }

    /// Prefix scan: all records whose string key starts with the given prefix.
    pub fn prefix_scan(&self, prefix: &str) -> Vec<&String> {
        let mut results = Vec::new();
        let start = BTreeKey::String(prefix.to_string());
        // The next lexicographic string after any prefix
        let mut end_bytes = prefix.as_bytes().to_vec();
        if let Some(last) = end_bytes.last_mut() {
            *last = last.wrapping_add(1);
        }
        let end = BTreeKey::String(String::from_utf8_lossy(&end_bytes).to_string());
        for (_key, ids) in self.entries.range(start..end) {
            results.extend(ids.iter());
        }
        results
    }

    /// Prefix scan: returns (key, value) pairs
    pub fn prefix_scan_with_keys(&self, prefix: &str) -> Vec<(&BTreeKey, &String)> {
        let mut results = Vec::new();
        let start = BTreeKey::String(prefix.to_string());
        let mut end_bytes = prefix.as_bytes().to_vec();
        if let Some(last) = end_bytes.last_mut() {
            *last = last.wrapping_add(1);
        }
        let end = BTreeKey::String(String::from_utf8_lossy(&end_bytes).to_string());
        for (key, ids) in self.entries.range(start..end) {
            for id in ids {
                results.push((key, id));
            }
        }
        results
    }

    /// Get all records with key >= start.
    pub fn range_from(&self, start: &BTreeKey) -> Vec<&String> {
        let mut results = Vec::new();
        for (_key, ids) in self.entries.range(start..) {
            results.extend(ids.iter());
        }
        results
    }

    /// Get all records with key <= end.
    pub fn range_to(&self, end: &BTreeKey) -> Vec<&String> {
        let mut results = Vec::new();
        for (_key, ids) in self.entries.range(..=end) {
            results.extend(ids.iter());
        }
        results
    }

    /// Number of unique keys in the index.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ─── Hash Index ─────────────────────────────────────────────────────────────

/// O(1) exact-match hash index for fast key-value lookups.
/// Used for tensor IDs, document keys, cache hashes, and model registry lookups.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HashIndex {
    name: String,
    key_field: String,
    /// Internal HashMap: hash key -> record ID (or multiple for non-unique)
    entries: HashMap<String, Vec<String>>,
    unique: bool,
}

impl HashIndex {
    pub fn new(name: impl Into<String>, key_field: impl Into<String>, unique: bool) -> Self {
        HashIndex {
            name: name.into(),
            key_field: key_field.into(),
            entries: HashMap::new(),
            unique,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn key_field(&self) -> &str {
        &self.key_field
    }

    /// Insert a record ID under the given key.
    pub fn insert(&mut self, key: String, record_id: String) -> Result<(), String> {
        if self.unique {
            if self.entries.contains_key(&key) {
                return Err(format!(
                    "HashIndex '{}': duplicate key '{}' violates unique constraint",
                    self.name, key
                ));
            }
            self.entries.insert(key, vec![record_id]);
        } else {
            self.entries.entry(key).or_default().push(record_id);
        }
        Ok(())
    }

    /// O(1) exact lookup.
    pub fn get(&self, key: &str) -> Option<&Vec<String>> {
        self.entries.get(key)
    }

    /// Remove a record ID from the index.
    pub fn remove(&mut self, key: &str, record_id: &str) -> bool {
        if let Some(ids) = self.entries.get_mut(key) {
            ids.retain(|id| id != record_id);
            if ids.is_empty() {
                self.entries.remove(key);
            }
            return true;
        }
        false
    }

    /// Check if a key exists.
    pub fn contains(&self, key: &str) -> bool {
        self.entries.contains_key(key)
    }

    /// Number of unique keys.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ─── Composite Index ────────────────────────────────────────────────────────

/// A multi-column index for common query patterns.
/// Supports prefix lookups (first N columns) and full composite lookups.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositeIndex {
    name: String,
    /// Ordered list of field names in this composite index
    fields: Vec<String>,
    entries: BTreeMap<CompositeKey, Vec<String>>,
    unique: bool,
}

/// A composite key made of ordered components.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, PartialOrd, Ord, Eq)]
pub struct CompositeKey {
    components: Vec<BTreeKey>,
}

impl CompositeKey {
    pub fn new(components: Vec<BTreeKey>) -> Self {
        CompositeKey { components }
    }

    pub fn components(&self) -> &[BTreeKey] {
        &self.components
    }
}

impl CompositeIndex {
    pub fn new(name: impl Into<String>, fields: Vec<String>, unique: bool) -> Self {
        CompositeIndex {
            name: name.into(),
            fields,
            entries: BTreeMap::new(),
            unique,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn fields(&self) -> &[String] {
        &self.fields
    }

    /// Insert a record ID under the given composite key.
    /// The number of components must match the number of fields.
    pub fn insert(&mut self, key: CompositeKey, record_id: String) -> Result<(), String> {
        if key.components.len() != self.fields.len() {
            return Err(format!(
                "CompositeIndex '{}': expected {} components, got {}",
                self.name,
                self.fields.len(),
                key.components.len()
            ));
        }
        if self.unique {
            if self.entries.contains_key(&key) {
                return Err(format!(
                    "CompositeIndex '{}': duplicate key violates unique constraint",
                    self.name
                ));
            }
            self.entries.insert(key, vec![record_id]);
        } else {
            self.entries.entry(key).or_default().push(record_id);
        }
        Ok(())
    }

    /// Exact lookup on all columns.
    pub fn get(&self, key: &CompositeKey) -> Option<&Vec<String>> {
        self.entries.get(key)
    }

    /// Prefix lookup: find all records matching the first N components.
    pub fn prefix_lookup(&self, prefix_components: &[BTreeKey]) -> Vec<&String> {
        if prefix_components.is_empty() || prefix_components.len() > self.fields.len() {
            return Vec::new();
        }

        // Build a prefix composite key and find the range
        let prefix_key = CompositeKey::new(prefix_components.to_vec());

        // The end key is the prefix with the last component incremented
        let mut end_components = prefix_components.to_vec();
        if let Some(last) = end_components.last_mut() {
            match last {
                BTreeKey::String(s) => {
                    let mut bytes = s.as_bytes().to_vec();
                    if let Some(b) = bytes.last_mut() {
                        *b = b.wrapping_add(1);
                    }
                    *last = BTreeKey::String(String::from_utf8_lossy(&bytes).to_string());
                }
                BTreeKey::Integer(i) => {
                    *last = BTreeKey::Integer(i.wrapping_add(1));
                }
                BTreeKey::Float(bits) => {
                    *last = BTreeKey::Float(bits.wrapping_add(1));
                }
            }
        }
        let end_key = CompositeKey::new(end_components);

        let mut results = Vec::new();
        for (_key, ids) in self.entries.range(prefix_key..end_key) {
            results.extend(ids.iter());
        }
        results
    }

    /// Remove a record ID from the index.
    pub fn remove(&mut self, key: &CompositeKey, record_id: &str) -> bool {
        if let Some(ids) = self.entries.get_mut(key) {
            ids.retain(|id| id != record_id);
            if ids.is_empty() {
                self.entries.remove(key);
            }
            return true;
        }
        false
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ─── Covering Index ─────────────────────────────────────────────────────────

/// An index that stores all data needed to answer a query, eliminating table lookups.
/// For hot query paths, this provides the fastest possible read performance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoveringIndex {
    name: String,
    /// The fields used as the lookup key
    key_fields: Vec<String>,
    /// The fields whose values are stored directly in the index (covering data)
    covered_fields: Vec<String>,
    /// Internal storage: key -> stored values for covered fields
    entries: BTreeMap<CompositeKey, Vec<CoveredRow>>,
}

/// A row of pre-materialized data stored in a covering index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoveredRow {
    pub record_id: String,
    /// The values of the covered fields, in the same order as `covered_fields`
    pub values: Vec<serde_json::Value>,
}

impl CoveringIndex {
    pub fn new(
        name: impl Into<String>,
        key_fields: Vec<String>,
        covered_fields: Vec<String>,
    ) -> Self {
        CoveringIndex {
            name: name.into(),
            key_fields,
            covered_fields,
            entries: BTreeMap::new(),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn key_fields(&self) -> &[String] {
        &self.key_fields
    }

    pub fn covered_fields(&self) -> &[String] {
        &self.covered_fields
    }

    /// Insert a covered row into the index.
    pub fn insert(
        &mut self,
        key: CompositeKey,
        record_id: String,
        values: Vec<serde_json::Value>,
    ) -> Result<(), String> {
        if values.len() != self.covered_fields.len() {
            return Err(format!(
                "CoveringIndex '{}': expected {} covered values, got {}",
                self.name,
                self.covered_fields.len(),
                values.len()
            ));
        }
        self.entries
            .entry(key)
            .or_default()
            .push(CoveredRow { record_id, values });
        Ok(())
    }

    /// Lookup by exact key — returns all covered rows without touching the base table.
    pub fn get(&self, key: &CompositeKey) -> Option<&Vec<CoveredRow>> {
        self.entries.get(key)
    }

    /// Range scan on the key — returns all covered rows in key order.
    pub fn range(&self, start: &CompositeKey, end: &CompositeKey) -> Vec<&CoveredRow> {
        let mut results = Vec::new();
        for (_key, rows) in self.entries.range(start..=end) {
            results.extend(rows.iter());
        }
        results
    }

    /// Remove a specific row from the covering index.
    pub fn remove(&mut self, key: &CompositeKey, record_id: &str) -> bool {
        if let Some(rows) = self.entries.get_mut(key) {
            rows.retain(|r| r.record_id != record_id);
            if rows.is_empty() {
                self.entries.remove(key);
            }
            return true;
        }
        false
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ─── Index Manager ──────────────────────────────────────────────────────────

/// Central registry for all indexes in the database.
/// Provides unified lookup across B-Tree, Hash, Composite, and Covering indexes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexManager {
    btree_indexes: HashMap<String, BTreeIndex>,
    hash_indexes: HashMap<String, HashIndex>,
    composite_indexes: HashMap<String, CompositeIndex>,
    covering_indexes: HashMap<String, CoveringIndex>,
}

impl IndexManager {
    pub fn new() -> Self {
        IndexManager {
            btree_indexes: HashMap::new(),
            hash_indexes: HashMap::new(),
            composite_indexes: HashMap::new(),
            covering_indexes: HashMap::new(),
        }
    }

    // ── B-Tree Index Operations ──

    pub fn create_btree(
        &mut self,
        name: impl Into<String>,
        key_field: impl Into<String>,
        unique: bool,
    ) {
        let name_s = name.into();
        self.btree_indexes
            .insert(name_s.clone(), BTreeIndex::new(name_s, key_field, unique));
    }

    pub fn btree(&self, name: &str) -> Option<&BTreeIndex> {
        self.btree_indexes.get(name)
    }

    pub fn btree_mut(&mut self, name: &str) -> Option<&mut BTreeIndex> {
        self.btree_indexes.get_mut(name)
    }

    // ── Hash Index Operations ──

    pub fn create_hash(
        &mut self,
        name: impl Into<String>,
        key_field: impl Into<String>,
        unique: bool,
    ) {
        let name_s = name.into();
        self.hash_indexes
            .insert(name_s.clone(), HashIndex::new(name_s, key_field, unique));
    }

    pub fn hash(&self, name: &str) -> Option<&HashIndex> {
        self.hash_indexes.get(name)
    }

    pub fn hash_mut(&mut self, name: &str) -> Option<&mut HashIndex> {
        self.hash_indexes.get_mut(name)
    }

    // ── Composite Index Operations ──

    pub fn create_composite(
        &mut self,
        name: impl Into<String>,
        fields: Vec<String>,
        unique: bool,
    ) {
        let name_s = name.into();
        self.composite_indexes
            .insert(name_s.clone(), CompositeIndex::new(name_s, fields, unique));
    }

    pub fn composite(&self, name: &str) -> Option<&CompositeIndex> {
        self.composite_indexes.get(name)
    }

    pub fn composite_mut(&mut self, name: &str) -> Option<&mut CompositeIndex> {
        self.composite_indexes.get_mut(name)
    }

    // ── Covering Index Operations ──

    pub fn create_covering(
        &mut self,
        name: impl Into<String>,
        key_fields: Vec<String>,
        covered_fields: Vec<String>,
    ) {
        let name_s = name.into();
        self.covering_indexes.insert(
            name_s.clone(),
            CoveringIndex::new(name_s, key_fields, covered_fields),
        );
    }

    pub fn covering(&self, name: &str) -> Option<&CoveringIndex> {
        self.covering_indexes.get(name)
    }

    pub fn covering_mut(&mut self, name: &str) -> Option<&mut CoveringIndex> {
        self.covering_indexes.get_mut(name)
    }

    /// Report index statistics for observability.
    pub fn report(&self) -> IndexReport {
        IndexReport {
            total_btree: self.btree_indexes.len(),
            total_hash: self.hash_indexes.len(),
            total_composite: self.composite_indexes.len(),
            total_covering: self.covering_indexes.len(),
            btree_details: self
                .btree_indexes
                .iter()
                .map(|(n, idx)| IndexDetail {
                    name: n.clone(),
                    index_type: "BTree".into(),
                    entry_count: idx.len(),
                    key_field: idx.key_field().to_string(),
                })
                .collect(),
            hash_details: self
                .hash_indexes
                .iter()
                .map(|(n, idx)| IndexDetail {
                    name: n.clone(),
                    index_type: "Hash".into(),
                    entry_count: idx.len(),
                    key_field: idx.key_field().to_string(),
                })
                .collect(),
            composite_details: self
                .composite_indexes
                .iter()
                .map(|(n, idx)| IndexDetail {
                    name: n.clone(),
                    index_type: "Composite".into(),
                    entry_count: idx.len(),
                    key_field: idx.fields().join(", "),
                })
                .collect(),
            covering_details: self
                .covering_indexes
                .iter()
                .map(|(n, idx)| IndexDetail {
                    name: n.clone(),
                    index_type: "Covering".into(),
                    entry_count: idx.len(),
                    key_field: format!("{} | covered: {}", idx.key_fields().join(", "), idx.covered_fields().join(", ")),
                })
                .collect(),
        }
    }
}

impl Default for IndexManager {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexReport {
    pub total_btree: usize,
    pub total_hash: usize,
    pub total_composite: usize,
    pub total_covering: usize,
    pub btree_details: Vec<IndexDetail>,
    pub hash_details: Vec<IndexDetail>,
    pub composite_details: Vec<IndexDetail>,
    pub covering_details: Vec<IndexDetail>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexDetail {
    pub name: String,
    pub index_type: String,
    pub entry_count: usize,
    pub key_field: String,
}

// ─── Default Index Schemas for Bramha ──────────────────────────────────────

/// Pre-configure the standard indexes that Bramha uses for its core operations.
pub fn setup_default_indexes(manager: &mut IndexManager) {
    // B-Tree indexes for range queries
    manager.create_btree("idx_tensor_timestamp", "timestamp", false);
    manager.create_btree("idx_collection_score", "score", false);
    manager.create_btree("idx_sequence_position", "sequence_pos", false);

    // Hash indexes for O(1) lookups
    manager.create_hash("idx_tensor_id", "tensor_id", true);
    manager.create_hash("idx_document_key", "doc_key", true);
    manager.create_hash("idx_cache_hash", "cache_hash", true);
    manager.create_hash("idx_model_name", "model_name", true);

    // Composite indexes for common query patterns
    manager.create_composite(
        "idx_model_layer",
        vec!["model_name".into(), "layer_id".into()],
        true,
    );
    manager.create_composite(
        "idx_collection_timestamp",
        vec!["collection_name".into(), "timestamp".into()],
        false,
    );

    // Covering indexes for hot query paths
    manager.create_covering(
        "cov_tensor_metadata",
        vec!["tensor_id".into()],
        vec!["shape".into(), "dtype".into(), "size_bytes".into(), "storage_tier".into()],
    );
    manager.create_covering(
        "cov_model_summary",
        vec!["model_name".into()],
        vec!["layer_count".into(), "total_params".into(), "quantization".into(), "device".into()],
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_btree_index_basic() {
        let mut idx = BTreeIndex::new("test_btree", "score", false);
        idx.insert(BTreeKey::from_f64(0.95), "doc1".into()).unwrap();
        idx.insert(BTreeKey::from_f64(0.80), "doc2".into()).unwrap();
        idx.insert(BTreeKey::from_f64(0.99), "doc3".into()).unwrap();

        let results = idx.range(&BTreeKey::from_f64(0.90), &BTreeKey::from_f64(1.0));
        assert_eq!(results.len(), 2);
        assert!(results.contains(&&"doc1".to_string()));
        assert!(results.contains(&&"doc3".to_string()));
    }

    #[test]
    fn test_hash_index_unique() {
        let mut idx = HashIndex::new("test_hash", "tensor_id", true);
        idx.insert("tensor_001".into(), "record_a".into()).unwrap();
        assert!(idx.insert("tensor_001".into(), "record_b".into()).is_err());
        assert_eq!(idx.get("tensor_001").unwrap().len(), 1);
    }

    #[test]
    fn test_composite_index_prefix() {
        let mut idx = CompositeIndex::new(
            "test_composite",
            vec!["model".into(), "layer".into()],
            false,
        );
        idx.insert(
            CompositeKey::new(vec![
                BTreeKey::String("tinyllama".into()),
                BTreeKey::Integer(0),
            ]),
            "layer_0".into(),
        )
        .unwrap();
        idx.insert(
            CompositeKey::new(vec![
                BTreeKey::String("tinyllama".into()),
                BTreeKey::Integer(1),
            ]),
            "layer_1".into(),
        )
        .unwrap();
        idx.insert(
            CompositeKey::new(vec![
                BTreeKey::String("llama2".into()),
                BTreeKey::Integer(0),
            ]),
            "llama2_layer_0".into(),
        )
        .unwrap();

        let results = idx.prefix_lookup(&[BTreeKey::String("tinyllama".into())]);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_covering_index_eliminates_table_lookup() {
        let mut idx = CoveringIndex::new(
            "cov_test",
            vec!["tensor_id".into()],
            vec!["shape".into(), "dtype".into()],
        );
        idx.insert(
            CompositeKey::new(vec![BTreeKey::String("t_001".into())]),
            "rec_1".into(),
            vec![
                serde_json::json!([2048, 2048]),
                serde_json::json!("f32"),
            ],
        )
        .unwrap();

        let rows = idx.get(&CompositeKey::new(vec![BTreeKey::String("t_001".into())]));
        assert!(rows.is_some());
        assert_eq!(rows.unwrap()[0].values[1], serde_json::json!("f32"));
    }

    #[test]
    fn test_index_manager_default_setup() {
        let mut manager = IndexManager::new();
        setup_default_indexes(&mut manager);

        assert!(manager.btree("idx_tensor_timestamp").is_some());
        assert!(manager.hash("idx_tensor_id").is_some());
        assert!(manager.composite("idx_model_layer").is_some());
        assert!(manager.covering("cov_tensor_metadata").is_some());

        let report = manager.report();
        assert_eq!(report.total_btree, 3);
        assert_eq!(report.total_hash, 4);
        assert_eq!(report.total_composite, 2);
        assert_eq!(report.total_covering, 2);
    }
}