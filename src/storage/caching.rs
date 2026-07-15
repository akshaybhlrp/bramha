//! # Database-Native Caching Strategies for Bramha
//!
//! Implements four caching strategies for neural database workloads:
//!
//! - **InMemoryDatabase**: Hot dataset held entirely in RAM with LRU eviction
//! - **ReadReplica**: Secondary read-only copy of a collection for read scaling
//! - **BufferPool**: Page-level cache for tensor data with LRU-K replacement
//! - **EdgeCache**: Geographic routing hints and latency-based tier selection
//!
//! All caches are thread-safe, support zero-copy reads where possible,
//! and integrate with the existing multi-tier storage system.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{SystemTime, UNIX_EPOCH};

// ─── In-Memory Database ─────────────────────────────────────────────────────

/// An entry in the in-memory database.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemEntry {
    pub key: String,
    pub value: Vec<u8>,
    pub size_bytes: usize,
    pub access_count: u64,
    pub last_access: u64,
    pub created_at: u64,
    pub ttl_secs: Option<u64>, // None = no expiry
}

/// LRU eviction policy for the in-memory database.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum EvictionPolicy {
    /// Least Recently Used
    Lru,
    /// Least Frequently Used
    Lfu,
    /// Time-To-Live based
    Ttl,
    /// First In First Out
    Fifo,
}

/// In-memory database for hot dataset caching.
/// Stores frequently accessed records entirely in RAM for sub-millisecond access.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InMemoryDatabase {
    name: String,
    /// Maximum memory usage in bytes
    max_size_bytes: u64,
    /// Current memory usage
    current_size: u64,
    /// The entries
    entries: HashMap<String, MemEntry>,
    /// LRU order tracking (most recent at back)
    access_order: VecDeque<String>,
    /// Eviction policy
    eviction_policy: EvictionPolicy,
    /// Hit/miss counters
    hits: u64,
    misses: u64,
}

impl InMemoryDatabase {
    pub fn new(name: impl Into<String>, max_size_mb: u64, policy: EvictionPolicy) -> Self {
        InMemoryDatabase {
            name: name.into(),
            max_size_bytes: max_size_mb * 1024 * 1024,
            current_size: 0,
            entries: HashMap::new(),
            access_order: VecDeque::new(),
            eviction_policy: policy,
            hits: 0,
            misses: 0,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// Insert a value into the in-memory database.
    pub fn insert(&mut self, key: String, value: Vec<u8>, ttl_secs: Option<u64>) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let size = value.len();

        // If key already exists, remove old entry first
        if let Some(old) = self.entries.remove(&key) {
            self.current_size = self.current_size.saturating_sub(old.size_bytes as u64);
            self.access_order.retain(|k| k != &key);
        }

        // Evict until we have space
        while self.current_size + size as u64 > self.max_size_bytes && !self.entries.is_empty() {
            self.evict_one();
        }

        let entry = MemEntry {
            key: key.clone(),
            value,
            size_bytes: size,
            access_count: 0,
            last_access: now,
            created_at: now,
            ttl_secs,
        };

        self.current_size += size as u64;
        self.entries.insert(key.clone(), entry);
        self.access_order.push_back(key);
    }

    /// Get a value from the in-memory database.
    pub fn get(&mut self, key: &str) -> Option<&Vec<u8>> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Check TTL expiry first (needs separate lookup to avoid borrow conflicts)
        let is_expired = self
            .entries
            .get(key)
            .and_then(|e| e.ttl_secs)
            .map_or(false, |ttl| {
                let created = self.entries.get(key).map(|e| e.created_at).unwrap_or(0);
                now - created >= ttl
            });

        if is_expired {
            if let Some(entry) = self.entries.remove(key) {
                self.current_size = self.current_size.saturating_sub(entry.size_bytes as u64);
            }
            self.access_order.retain(|k| k != key);
            self.misses += 1;
            return None;
        }

        if let Some(entry) = self.entries.get_mut(key) {
            entry.access_count += 1;
            entry.last_access = now;

            // Update LRU order
            if self.eviction_policy == EvictionPolicy::Lru {
                self.access_order.retain(|k| k != key);
                self.access_order.push_back(key.to_string());
            }

            self.hits += 1;
            Some(&entry.value)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Check if a key exists without updating access stats.
    pub fn contains(&self, key: &str) -> bool {
        self.entries.contains_key(key)
    }

    /// Remove a key from the cache.
    pub fn remove(&mut self, key: &str) -> bool {
        if let Some(entry) = self.entries.remove(key) {
            self.current_size = self.current_size.saturating_sub(entry.size_bytes as u64);
            self.access_order.retain(|k| k != key);
            true
        } else {
            false
        }
    }

    /// Evict a single entry based on the current policy.
    fn evict_one(&mut self) {
        let victim_key = match self.eviction_policy {
            EvictionPolicy::Lru | EvictionPolicy::Fifo => self.access_order.pop_front(),
            EvictionPolicy::Lfu => {
                // Find least frequently used
                let mut min_freq = u64::MAX;
                let mut victim = None;
                for k in &self.access_order {
                    if let Some(entry) = self.entries.get(k) {
                        if entry.access_count < min_freq {
                            min_freq = entry.access_count;
                            victim = Some(k.clone());
                        }
                    }
                }
                victim
            }
            EvictionPolicy::Ttl => {
                // Find entry closest to expiry
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                let mut min_remaining = u64::MAX;
                let mut victim = None;
                for k in &self.access_order {
                    if let Some(entry) = self.entries.get(k) {
                        let remaining = entry
                            .ttl_secs
                            .map(|ttl| ttl.saturating_sub(now - entry.created_at))
                            .unwrap_or(u64::MAX);
                        if remaining < min_remaining {
                            min_remaining = remaining;
                            victim = Some(k.clone());
                        }
                    }
                }
                victim
            }
        };

        if let Some(key) = victim_key {
            if let Some(entry) = self.entries.remove(&key) {
                self.current_size = self.current_size.saturating_sub(entry.size_bytes as u64);
            }
        }
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.access_order.clear();
        self.current_size = 0;
    }

    /// Current memory usage in bytes.
    pub fn current_size(&self) -> u64 {
        self.current_size
    }

    /// Maximum memory capacity in bytes.
    pub fn max_size(&self) -> u64 {
        self.max_size_bytes
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Hit rate as a float between 0.0 and 1.0.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            1.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    pub fn hits(&self) -> u64 {
        self.hits
    }

    pub fn misses(&self) -> u64 {
        self.misses
    }
}

// ─── Read Replica ───────────────────────────────────────────────────────────

/// Status of a read replica.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ReplicaStatus {
    Syncing,
    Ready,
    Stale,
    Offline,
}

/// A read-only replica of a collection or database.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadReplica {
    pub name: String,
    /// The source collection this replica mirrors
    pub source_collection: String,
    /// Path to the replica's data directory
    pub data_path: String,
    /// Current status
    pub status: ReplicaStatus,
    /// When this replica was last synchronized
    pub last_synced: u64,
    /// Sync lag in milliseconds
    pub sync_lag_ms: u64,
    /// Number of records in this replica
    pub record_count: usize,
    /// Whether this replica is preferred for read affinity
    pub preferred: bool,
}

/// Manages read replicas for a database.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadReplicaManager {
    replicas: Vec<ReadReplica>,
    /// Round-robin index for load balancing
    rr_index: usize,
}

impl ReadReplicaManager {
    pub fn new() -> Self {
        ReadReplicaManager {
            replicas: Vec::new(),
            rr_index: 0,
        }
    }

    /// Register a new read replica.
    pub fn add_replica(&mut self, replica: ReadReplica) {
        self.replicas.push(replica);
    }

    /// Remove a replica.
    pub fn remove_replica(&mut self, name: &str) -> Option<ReadReplica> {
        if let Some(pos) = self.replicas.iter().position(|r| r.name == name) {
            Some(self.replicas.remove(pos))
        } else {
            None
        }
    }

    /// Get the best replica for a read operation.
    /// Uses read affinity (preferred replicas first), then round-robin.
    pub fn get_read_replica(&mut self) -> Option<&ReadReplica> {
        // First, try preferred replicas that are ready
        let preferred: Vec<usize> = self
            .replicas
            .iter()
            .enumerate()
            .filter(|(_, r)| r.status == ReplicaStatus::Ready && r.preferred)
            .map(|(i, _)| i)
            .collect();

        if !preferred.is_empty() {
            let idx = preferred[self.rr_index % preferred.len()];
            self.rr_index = self.rr_index.wrapping_add(1);
            return self.replicas.get(idx);
        }

        // Fall back to any ready replica
        let ready: Vec<usize> = self
            .replicas
            .iter()
            .enumerate()
            .filter(|(_, r)| r.status == ReplicaStatus::Ready)
            .map(|(i, _)| i)
            .collect();

        if !ready.is_empty() {
            let idx = ready[self.rr_index % ready.len()];
            self.rr_index = self.rr_index.wrapping_add(1);
            return self.replicas.get(idx);
        }

        None
    }

    /// Get all replicas.
    pub fn replicas(&self) -> &[ReadReplica] {
        &self.replicas
    }

    pub fn replicas_mut(&mut self) -> &mut Vec<ReadReplica> {
        &mut self.replicas
    }

    /// Update a replica's sync status.
    pub fn update_sync(
        &mut self,
        name: &str,
        status: ReplicaStatus,
        sync_lag_ms: u64,
    ) -> Option<()> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let replica = self.replicas.iter_mut().find(|r| r.name == name)?;
        replica.status = status;
        replica.last_synced = now;
        replica.sync_lag_ms = sync_lag_ms;
        Some(())
    }

    /// Number of ready replicas.
    pub fn ready_count(&self) -> usize {
        self.replicas
            .iter()
            .filter(|r| r.status == ReplicaStatus::Ready)
            .count()
    }

    pub fn len(&self) -> usize {
        self.replicas.len()
    }

    pub fn is_empty(&self) -> bool {
        self.replicas.is_empty()
    }
}

impl Default for ReadReplicaManager {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Buffer Pool ────────────────────────────────────────────────────────────

/// A page in the buffer pool.
#[derive(Debug, Clone)]
pub struct BufferPage {
    /// Unique page identifier (e.g., "tensor_data/0001.page")
    pub page_id: String,
    /// The raw page data
    pub data: Vec<u8>,
    /// Size of the page
    pub size: usize,
    /// When this page was last accessed
    pub last_access: u64,
    /// Access frequency for LRU-K
    pub access_history: VecDeque<u64>,
    /// Whether this page is dirty (needs write-back)
    pub is_dirty: bool,
    /// Whether this page is pinned (cannot be evicted)
    pub pin_count: u32,
}

/// Buffer pool with LRU-K replacement policy.
/// Manages page-level caching for tensor data files.
#[derive(Debug, Clone)]
pub struct BufferPool {
    name: String,
    /// Maximum number of pages
    max_pages: usize,
    /// Page size in bytes
    page_size: usize,
    /// The cached pages
    pages: HashMap<String, BufferPage>,
    /// LRU-K parameter (K=2 means consider the last 2 accesses)
    k: usize,
    /// Statistics
    hits: u64,
    misses: u64,
    evictions: u64,
}

impl BufferPool {
    pub fn new(name: impl Into<String>, max_pages: usize, page_size: usize) -> Self {
        BufferPool {
            name: name.into(),
            max_pages,
            page_size,
            pages: HashMap::new(),
            k: 2, // LRU-2 by default
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// Fetch a page from the buffer pool. Returns None if not cached (page fault).
    pub fn fetch(&mut self, page_id: &str) -> Option<&Vec<u8>> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        if let Some(page) = self.pages.get_mut(page_id) {
            page.last_access = now;
            page.access_history.push_back(now);
            if page.access_history.len() > self.k {
                page.access_history.pop_front();
            }
            self.hits += 1;
            Some(&page.data)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Insert a page into the buffer pool.
    pub fn insert(&mut self, page_id: String, data: Vec<u8>) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        // If page already exists, just update
        if let Some(page) = self.pages.get_mut(&page_id) {
            page.data = data;
            page.last_access = now;
            page.is_dirty = false;
            return;
        }

        // Evict if at capacity
        while self.pages.len() >= self.max_pages {
            self.evict_one();
        }

        let size = data.len();
        let mut history = VecDeque::new();
        history.push_back(now);

        self.pages.insert(
            page_id.clone(),
            BufferPage {
                page_id,
                data,
                size,
                last_access: now,
                access_history: history,
                is_dirty: false,
                pin_count: 0,
            },
        );
    }

    /// Mark a page as dirty (needs write-back to disk).
    pub fn mark_dirty(&mut self, page_id: &str) -> bool {
        if let Some(page) = self.pages.get_mut(page_id) {
            page.is_dirty = true;
            true
        } else {
            false
        }
    }

    /// Pin a page in memory (prevents eviction).
    pub fn pin(&mut self, page_id: &str) -> bool {
        if let Some(page) = self.pages.get_mut(page_id) {
            page.pin_count += 1;
            true
        } else {
            false
        }
    }

    /// Unpin a page.
    pub fn unpin(&mut self, page_id: &str) -> bool {
        if let Some(page) = self.pages.get_mut(page_id) {
            page.pin_count = page.pin_count.saturating_sub(1);
            true
        } else {
            false
        }
    }

    /// Evict one page using LRU-K policy.
    fn evict_one(&mut self) {
        let mut victim_id: Option<String> = None;
        let mut best_score = f64::MAX;

        for (id, page) in &self.pages {
            if page.pin_count > 0 {
                continue; // Don't evict pinned pages
            }

            // LRU-K score: use the K-th most recent access time
            // Lower score = better eviction candidate
            let score = if page.access_history.len() < self.k {
                // Not enough history — use last access time
                page.last_access as f64
            } else {
                // Use the K-th most recent access
                page.access_history[0] as f64
            };

            if score < best_score {
                best_score = score;
                victim_id = Some(id.clone());
            }
        }

        if let Some(id) = victim_id {
            self.pages.remove(&id);
            self.evictions += 1;
        }
    }

    /// Get all dirty pages for write-back.
    pub fn dirty_pages(&self) -> Vec<&BufferPage> {
        self.pages.values().filter(|p| p.is_dirty).collect()
    }

    /// Clear all pages.
    pub fn clear(&mut self) {
        self.pages.clear();
    }

    pub fn len(&self) -> usize {
        self.pages.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pages.is_empty()
    }

    pub fn max_pages(&self) -> usize {
        self.max_pages
    }

    pub fn page_size(&self) -> usize {
        self.page_size
    }

    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            1.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    pub fn evictions(&self) -> u64 {
        self.evictions
    }
}

// ─── Edge Cache ─────────────────────────────────────────────────────────────

/// Geographic region for edge caching.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Hash, Eq)]
pub enum GeoRegion {
    UsEast,
    UsWest,
    EuWest,
    EuCentral,
    AsiaEast,
    AsiaSouthEast,
    Oceania,
    Custom(String),
}

impl std::fmt::Display for GeoRegion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GeoRegion::UsEast => write!(f, "us-east"),
            GeoRegion::UsWest => write!(f, "us-west"),
            GeoRegion::EuWest => write!(f, "eu-west"),
            GeoRegion::EuCentral => write!(f, "eu-central"),
            GeoRegion::AsiaEast => write!(f, "asia-east"),
            GeoRegion::AsiaSouthEast => write!(f, "asia-southeast"),
            GeoRegion::Oceania => write!(f, "oceania"),
            GeoRegion::Custom(s) => write!(f, "{}", s),
        }
    }
}

/// A cache entry with geographic routing information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeCacheEntry {
    pub key: String,
    pub value: Vec<u8>,
    pub region: GeoRegion,
    pub cached_at: u64,
    pub ttl_secs: u64,
    pub size_bytes: usize,
}

/// Edge cache for geographic routing and latency-based tier selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeCache {
    name: String,
    /// Per-region caches
    region_caches: HashMap<GeoRegion, HashMap<String, EdgeCacheEntry>>,
    /// Latency measurements per region (in ms)
    region_latency: HashMap<GeoRegion, f64>,
    /// Default TTL for cache entries
    default_ttl_secs: u64,
    /// Maximum entries per region
    max_entries_per_region: usize,
    /// Statistics
    hits: u64,
    misses: u64,
}

impl EdgeCache {
    pub fn new(
        name: impl Into<String>,
        default_ttl_secs: u64,
        max_entries_per_region: usize,
    ) -> Self {
        EdgeCache {
            name: name.into(),
            region_caches: HashMap::new(),
            region_latency: HashMap::new(),
            default_ttl_secs,
            max_entries_per_region,
            hits: 0,
            misses: 0,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// Insert a value into the edge cache for a specific region.
    pub fn insert(
        &mut self,
        key: String,
        value: Vec<u8>,
        region: GeoRegion,
        ttl_secs: Option<u64>,
    ) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let size = value.len();

        let cache = self.region_caches.entry(region.clone()).or_default();

        // Evict if at capacity
        while cache.len() >= self.max_entries_per_region {
            // Evict oldest entry
            let oldest_key = cache
                .iter()
                .min_by_key(|(_, e)| e.cached_at)
                .map(|(k, _)| k.clone());
            if let Some(k) = oldest_key {
                cache.remove(&k);
            } else {
                break;
            }
        }

        cache.insert(
            key.clone(),
            EdgeCacheEntry {
                key,
                value,
                region,
                cached_at: now,
                ttl_secs: ttl_secs.unwrap_or(self.default_ttl_secs),
                size_bytes: size,
            },
        );
    }

    /// Get a value from the edge cache for a specific region.
    pub fn get(&mut self, key: &str, region: &GeoRegion) -> Option<&Vec<u8>> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Check TTL expiry first (separate lookup to avoid borrow conflicts)
        let expired = self
            .region_caches
            .get(region)
            .and_then(|cache| cache.get(key).filter(|e| now - e.cached_at > e.ttl_secs))
            .is_some();

        if expired {
            if let Some(cache) = self.region_caches.get_mut(region) {
                cache.remove(key);
            }
            self.misses += 1;
            return None;
        }

        if let Some(cache) = self.region_caches.get_mut(region) {
            if let Some(entry) = cache.get(key) {
                self.hits += 1;
                return Some(&entry.value);
            }
        }
        self.misses += 1;
        None
    }

    /// Get the best region for a given client location.
    /// Returns the region with the lowest measured latency.
    pub fn best_region(&self, client_region: &GeoRegion) -> Option<GeoRegion> {
        if self.region_latency.is_empty() {
            // Fall back to client's own region
            return Some(client_region.clone());
        }

        self.region_latency
            .iter()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(region, _)| region.clone())
    }

    /// Record a latency measurement for a region.
    pub fn record_latency(&mut self, region: GeoRegion, latency_ms: f64) {
        // Exponential moving average
        let alpha = 0.3;
        let entry = self.region_latency.entry(region).or_insert(latency_ms);
        *entry = alpha * latency_ms + (1.0 - alpha) * *entry;
    }

    /// Invalidate entries by key across all regions.
    pub fn invalidate(&mut self, key: &str) {
        for cache in self.region_caches.values_mut() {
            cache.remove(key);
        }
    }

    /// Clear all entries for a specific region.
    pub fn clear_region(&mut self, region: &GeoRegion) {
        self.region_caches.remove(region);
    }

    /// Clear all entries across all regions.
    pub fn clear_all(&mut self) {
        self.region_caches.clear();
    }

    /// Get the number of cached entries for a region.
    pub fn region_size(&self, region: &GeoRegion) -> usize {
        self.region_caches.get(region).map(|c| c.len()).unwrap_or(0)
    }

    /// Total cached entries across all regions.
    pub fn total_entries(&self) -> usize {
        self.region_caches.values().map(|c| c.len()).sum()
    }

    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            1.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

// ─── Caching Manager ────────────────────────────────────────────────────────

/// Central coordinator for all caching strategies.
#[derive(Debug, Clone)]
pub struct CachingManager {
    pub in_memory_db: InMemoryDatabase,
    pub replica_manager: ReadReplicaManager,
    pub buffer_pool: BufferPool,
    pub edge_cache: EdgeCache,
}

impl CachingManager {
    pub fn new(
        in_memory_db: InMemoryDatabase,
        replica_manager: ReadReplicaManager,
        buffer_pool: BufferPool,
        edge_cache: EdgeCache,
    ) -> Self {
        CachingManager {
            in_memory_db,
            replica_manager,
            buffer_pool,
            edge_cache,
        }
    }

    /// Report comprehensive caching statistics.
    pub fn report(&self) -> CachingReport {
        CachingReport {
            in_memory_entries: self.in_memory_db.len(),
            in_memory_size_mb: self.in_memory_db.current_size() as f64 / (1024.0 * 1024.0),
            in_memory_max_mb: self.in_memory_db.max_size() as f64 / (1024.0 * 1024.0),
            in_memory_hit_rate: self.in_memory_db.hit_rate(),
            replicas_total: self.replica_manager.len(),
            replicas_ready: self.replica_manager.ready_count(),
            buffer_pool_pages: self.buffer_pool.len(),
            buffer_pool_max_pages: self.buffer_pool.max_pages(),
            buffer_pool_hit_rate: self.buffer_pool.hit_rate(),
            buffer_pool_evictions: self.buffer_pool.evictions(),
            edge_cache_entries: self.edge_cache.total_entries(),
            edge_cache_hit_rate: self.edge_cache.hit_rate(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachingReport {
    pub in_memory_entries: usize,
    pub in_memory_size_mb: f64,
    pub in_memory_max_mb: f64,
    pub in_memory_hit_rate: f64,
    pub replicas_total: usize,
    pub replicas_ready: usize,
    pub buffer_pool_pages: usize,
    pub buffer_pool_max_pages: usize,
    pub buffer_pool_hit_rate: f64,
    pub buffer_pool_evictions: u64,
    pub edge_cache_entries: usize,
    pub edge_cache_hit_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_in_memory_db_basic() {
        let mut db = InMemoryDatabase::new("test", 10, EvictionPolicy::Lru);
        db.insert("key1".into(), vec![1, 2, 3], None);
        db.insert("key2".into(), vec![4, 5, 6], None);

        assert_eq!(db.len(), 2);
        assert!(db.get("key1").is_some());
        assert!(db.get("key1").is_some());
        assert!(db.get("key3").is_none());
        assert!(db.hit_rate() > 0.5);
    }

    #[test]
    fn test_in_memory_db_eviction() {
        let mut db = InMemoryDatabase::new("test", 1, EvictionPolicy::Lru);
        // max_size is 1MB, insert 600KB entries
        let big_data = vec![0u8; 600_000];
        db.insert("key1".into(), big_data.clone(), None);
        db.insert("key2".into(), big_data.clone(), None);

        // Should have evicted key1
        assert_eq!(db.len(), 1);
        assert!(db.get("key2").is_some());
    }

    #[test]
    fn test_in_memory_db_ttl() {
        let mut db = InMemoryDatabase::new("test", 10, EvictionPolicy::Ttl);
        db.insert("key1".into(), vec![1, 2, 3], Some(0)); // Already expired
        assert!(db.get("key1").is_none());
        assert_eq!(db.misses(), 1);
    }

    #[test]
    fn test_read_replica_round_robin() {
        let mut rm = ReadReplicaManager::new();
        for i in 0..3 {
            rm.add_replica(ReadReplica {
                name: format!("replica_{}", i),
                source_collection: "main".into(),
                data_path: format!("/data/replica_{}", i),
                status: ReplicaStatus::Ready,
                last_synced: 0,
                sync_lag_ms: 0,
                record_count: 100,
                preferred: false,
            });
        }

        let r1 = rm.get_read_replica().map(|r| r.name.clone());
        let r2 = rm.get_read_replica().map(|r| r.name.clone());
        let r3 = rm.get_read_replica().map(|r| r.name.clone());

        assert!(r1.is_some());
        assert!(r2.is_some());
        assert!(r3.is_some());
        // Round-robin should give different replicas
        assert!(r1 != r2 || r2 != r3);
    }

    #[test]
    fn test_buffer_pool_lru_k() {
        let mut bp = BufferPool::new("test", 3, 4096);
        bp.insert("page_1".into(), vec![0u8; 100]);
        std::thread::sleep(std::time::Duration::from_millis(5));
        bp.insert("page_2".into(), vec![0u8; 100]);
        std::thread::sleep(std::time::Duration::from_millis(5));
        bp.insert("page_3".into(), vec![0u8; 100]);
        std::thread::sleep(std::time::Duration::from_millis(5));

        // Access page_1 twice to make it hot
        bp.fetch("page_1");
        std::thread::sleep(std::time::Duration::from_millis(5));
        bp.fetch("page_1");
        std::thread::sleep(std::time::Duration::from_millis(5));

        // Insert page_4 — should evict the coldest (page_2 or page_3)
        bp.insert("page_4".into(), vec![0u8; 100]);

        assert_eq!(bp.len(), 3);
        // page_1 should still be there (it's hot)
        assert!(bp.fetch("page_1").is_some());
    }

    #[test]
    fn test_edge_cache_region_routing() {
        let mut ec = EdgeCache::new("test", 3600, 100);
        ec.insert(
            "model:tinyllama".into(),
            vec![1, 2, 3],
            GeoRegion::UsEast,
            None,
        );
        ec.insert(
            "model:tinyllama".into(),
            vec![4, 5, 6],
            GeoRegion::EuWest,
            None,
        );

        // Same key, different regions = different cached values
        let us_val = ec.get("model:tinyllama", &GeoRegion::UsEast).cloned();
        let eu_val = ec.get("model:tinyllama", &GeoRegion::EuWest).cloned();

        assert!(us_val.is_some());
        assert!(eu_val.is_some());
        assert_ne!(us_val.as_ref().unwrap(), eu_val.as_ref().unwrap());
    }

    #[test]
    fn test_caching_manager_report() {
        let imdb = InMemoryDatabase::new("hot_cache", 100, EvictionPolicy::Lru);
        let rm = ReadReplicaManager::new();
        let bp = BufferPool::new("tensor_pool", 1000, 4096);
        let ec = EdgeCache::new("edge", 3600, 1000);

        let cm = CachingManager::new(imdb, rm, bp, ec);
        let report = cm.report();
        assert_eq!(report.in_memory_entries, 0);
        assert_eq!(report.replicas_total, 0);
        assert_eq!(report.buffer_pool_pages, 0);
        assert_eq!(report.edge_cache_entries, 0);
    }
}
use crate::core::vector::Metric;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheItem {
    pub prompt: String,
    pub prompt_vector: Vec<f32>,
    pub completion: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticCache {
    pub dimension: usize,
    pub metric: Metric,
    pub items: Vec<CacheItem>,
    pub threshold: f32,
}

impl SemanticCache {
    pub fn new(dimension: usize, metric: Metric, threshold: f32) -> Self {
        Self {
            dimension,
            metric,
            items: Vec::new(),
            threshold,
        }
    }

    pub fn lookup(&self, vector: &[f32]) -> Option<(String, f32)> {
        for item in &self.items {
            let sim = self.metric.distance(vector, &item.prompt_vector);
            if sim >= self.threshold {
                return Some((item.completion.clone(), sim));
            }
        }
        None
    }

    pub fn insert(
        &mut self,
        prompt: String,
        vector: Vec<f32>,
        completion: String,
    ) -> Result<(), String> {
        self.items.push(CacheItem {
            prompt,
            prompt_vector: vector,
            completion,
        });
        Ok(())
    }

    pub fn clear(&mut self) {
        self.items.clear();
    }
}
