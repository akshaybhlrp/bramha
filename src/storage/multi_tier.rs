use crate::storage::storage_manifest::StorageTier;
use serde::{Deserialize, Serialize};
/// Multi-Tier Storage: DRAM/SSD/HDD routing and promotion/demotion
///
/// This module implements database buffer pool patterns for neural network models:
/// - Tier 0 (Hot): DRAM cache, <1ms access, limited capacity (100-200MB)
/// - Tier 1 (Warm): NVMe/SSD, <10ms access, moderate capacity (2-5GB)
/// - Tier 2 (Cold): HDD/Network, 10-100ms access, large capacity (unlimited)
///
/// Policies:
/// - Frequently accessed layers promoted to hot tier
/// - Rarely accessed layers demoted to cold tier
/// - Prefetching: load layer N+1 while GPU/CPU processes layer N
/// - LRU eviction: when hot tier is full, evict least recently used
use std::collections::HashMap;
use std::path::PathBuf;

const HOT_TIER_MAX_BYTES: u64 = 200 * 1024 * 1024; // 200 MB DRAM
const WARM_TIER_MAX_BYTES: u64 = 5 * 1024 * 1024 * 1024; // 5 GB SSD

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TierConfig {
    /// Maximum bytes in hot tier (DRAM)
    pub hot_max_bytes: u64,
    /// Maximum bytes in warm tier (SSD)
    pub warm_max_bytes: u64,
    /// Promotion threshold: access count before promoting hot
    pub promotion_threshold: u32,
    /// Demotion threshold: inactivity time (seconds) before demoting
    pub demotion_threshold_secs: u64,
    /// Prefetch distance: how many layers ahead to prefetch
    pub prefetch_distance: usize,
}

impl Default for TierConfig {
    fn default() -> Self {
        TierConfig {
            hot_max_bytes: HOT_TIER_MAX_BYTES,
            warm_max_bytes: WARM_TIER_MAX_BYTES,
            promotion_threshold: 2,        // promoted on second access
            demotion_threshold_secs: 3600, // 1 hour of inactivity
            prefetch_distance: 2,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TierEntry {
    pub layer_id: String,
    pub tier: StorageTier,
    pub size_bytes: u64,
    pub access_count: u64,
    pub last_accessed: u64,
    pub path: PathBuf,
}

/// Multi-tier storage manager
pub struct MultiTierStorage {
    pub config: TierConfig,
    /// Hot tier: DRAM cache
    hot_tier: HashMap<String, TierEntry>,
    hot_used_bytes: u64,
    /// Warm tier: SSD cache
    warm_tier: HashMap<String, TierEntry>,
    warm_used_bytes: u64,
    /// Cold tier: HDD/Network (just track metadata, not actual storage)
    cold_tier: HashMap<String, TierEntry>,
    /// Base paths for each tier
    _hot_path: PathBuf,
    _warm_path: PathBuf,
    _cold_path: PathBuf,
    /// Statistics
    pub stats: TierStats,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TierStats {
    pub hot_hits: u64,
    pub warm_hits: u64,
    pub cold_hits: u64,
    pub promotions: u64,
    pub demotions: u64,
    pub evictions: u64,
    pub prefetch_requests: u64,
    pub total_accessed: u64,
}

impl MultiTierStorage {
    pub fn new(
        config: TierConfig,
        hot_path: PathBuf,
        warm_path: PathBuf,
        cold_path: PathBuf,
    ) -> std::io::Result<Self> {
        std::fs::create_dir_all(&hot_path)?;
        std::fs::create_dir_all(&warm_path)?;
        std::fs::create_dir_all(&cold_path)?;

        Ok(MultiTierStorage {
            config,
            hot_tier: HashMap::new(),
            hot_used_bytes: 0,
            warm_tier: HashMap::new(),
            warm_used_bytes: 0,
            cold_tier: HashMap::new(),
            _hot_path: hot_path,
            _warm_path: warm_path,
            _cold_path: cold_path,
            stats: TierStats::default(),
        })
    }

    /// Find which tier contains a layer
    pub fn find_layer(&self, layer_id: &str) -> Option<(StorageTier, &TierEntry)> {
        if let Some(entry) = self.hot_tier.get(layer_id) {
            Some((StorageTier::Critical, entry))
        } else if let Some(entry) = self.warm_tier.get(layer_id) {
            Some((StorageTier::Important, entry))
        } else if let Some(entry) = self.cold_tier.get(layer_id) {
            Some((StorageTier::Robust, entry))
        } else {
            None
        }
    }

    /// Register a layer in the appropriate tier (initial placement)
    pub fn register_layer(
        &mut self,
        layer_id: String,
        size_bytes: u64,
        target_tier: StorageTier,
        path: PathBuf,
    ) -> Result<(), String> {
        let entry = TierEntry {
            layer_id: layer_id.clone(),
            tier: target_tier,
            size_bytes,
            access_count: 0,
            last_accessed: current_timestamp(),
            path,
        };

        match target_tier {
            StorageTier::Critical => {
                if self.hot_used_bytes + size_bytes <= self.config.hot_max_bytes {
                    self.hot_tier.insert(layer_id, entry);
                    self.hot_used_bytes += size_bytes;
                    Ok(())
                } else {
                    Err(format!(
                        "Hot tier full: {} + {} > {}",
                        self.hot_used_bytes, size_bytes, self.config.hot_max_bytes
                    ))
                }
            }
            StorageTier::Important => {
                if self.warm_used_bytes + size_bytes <= self.config.warm_max_bytes {
                    self.warm_tier.insert(layer_id, entry);
                    self.warm_used_bytes += size_bytes;
                    Ok(())
                } else {
                    Err(format!(
                        "Warm tier full: {} + {} > {}",
                        self.warm_used_bytes, size_bytes, self.config.warm_max_bytes
                    ))
                }
            }
            StorageTier::Robust | StorageTier::Redundant => {
                self.cold_tier.insert(layer_id, entry);
                Ok(())
            }
        }
    }

    /// Access a layer (record hit and potentially promote)
    pub fn access_layer(&mut self, layer_id: &str) -> Result<(), String> {
        let now = current_timestamp();
        self.stats.total_accessed += 1;

        // Try to find and update in existing tier
        if let Some(entry) = self.hot_tier.get_mut(layer_id) {
            entry.access_count += 1;
            entry.last_accessed = now;
            self.stats.hot_hits += 1;
            return Ok(());
        }

        if let Some(entry) = self.warm_tier.get_mut(layer_id) {
            entry.access_count += 1;
            entry.last_accessed = now;
            self.stats.warm_hits += 1;

            // Promote to hot tier if threshold reached
            if entry.access_count >= self.config.promotion_threshold as u64 {
                return self.promote_to_hot(layer_id);
            }
            return Ok(());
        }

        if let Some(entry) = self.cold_tier.get_mut(layer_id) {
            entry.access_count += 1;
            entry.last_accessed = now;
            self.stats.cold_hits += 1;
            return Ok(());
        }

        Err(format!("Layer {} not found in any tier", layer_id))
    }

    /// Promote layer from warm to hot
    fn promote_to_hot(&mut self, layer_id: &str) -> Result<(), String> {
        let entry = self
            .warm_tier
            .remove(layer_id)
            .ok_or_else(|| format!("Layer {} not in warm tier", layer_id))?;

        self.warm_used_bytes -= entry.size_bytes;

        // Make space in hot tier if needed
        while self.hot_used_bytes + entry.size_bytes > self.config.hot_max_bytes
            && !self.hot_tier.is_empty()
        {
            self.evict_from_hot()?;
        }

        self.hot_tier.insert(layer_id.to_string(), entry.clone());
        self.hot_used_bytes += entry.size_bytes;
        self.stats.promotions += 1;

        Ok(())
    }

    /// Evict least recently used layer from hot tier to warm
    fn evict_from_hot(&mut self) -> Result<(), String> {
        let lru_layer = self
            .hot_tier
            .values()
            .min_by_key(|e| e.last_accessed)
            .ok_or("Hot tier is empty")?
            .clone();

        let layer_id = lru_layer.layer_id.clone();
        let layer_size = lru_layer.size_bytes;
        self.hot_tier.remove(&layer_id);
        self.hot_used_bytes -= layer_size;

        // Try to place in warm tier
        if self.warm_used_bytes + layer_size <= self.config.warm_max_bytes {
            self.warm_tier.insert(layer_id, lru_layer);
            self.warm_used_bytes += layer_size;
        } else {
            // Overflow: move to cold tier
            self.cold_tier.insert(layer_id, lru_layer);
        }

        self.stats.evictions += 1;
        Ok(())
    }

    /// Demote inactive layers from warm to cold
    pub fn demote_inactive(&mut self) {
        let now = current_timestamp();
        let threshold_secs = self.config.demotion_threshold_secs;

        let to_demote: Vec<String> = self
            .warm_tier
            .iter()
            .filter(|(_, entry)| now.saturating_sub(entry.last_accessed) > threshold_secs)
            .map(|(id, _)| id.clone())
            .collect();

        for layer_id in to_demote {
            if let Some(entry) = self.warm_tier.remove(&layer_id) {
                self.warm_used_bytes -= entry.size_bytes;
                self.cold_tier.insert(layer_id, entry);
                self.stats.demotions += 1;
            }
        }
    }

    /// Prefetch layers expected to be accessed soon (background)
    pub fn prefetch_layers(&mut self, next_layer_ids: &[String]) -> Vec<String> {
        let mut prefetched = Vec::new();

        for layer_id in next_layer_ids.iter().take(self.config.prefetch_distance) {
            if self.find_layer(layer_id).is_none() {
                // Layer not currently loaded
                prefetched.push(layer_id.clone());
                self.stats.prefetch_requests += 1;
            }
        }

        prefetched
    }

    /// Get tier utilization statistics
    pub fn utilization(&self) -> TierUtilization {
        TierUtilization {
            hot_used: self.hot_used_bytes,
            hot_max: self.config.hot_max_bytes,
            warm_used: self.warm_used_bytes,
            warm_max: self.config.warm_max_bytes,
            cold_count: self.cold_tier.len(),
            hot_count: self.hot_tier.len(),
            warm_count: self.warm_tier.len(),
        }
    }

    /// Report tier statistics
    pub fn report(&self) {
        println!("\n📊 Multi-Tier Storage Report");
        println!("═══════════════════════════════════════════════════════════");

        let util = self.utilization();
        println!("\n🔥 Hot Tier (DRAM)");
        println!(
            "  Used: {:.2} MB / {:.2} MB ({:.1}%)",
            util.hot_used as f64 / 1024.0 / 1024.0,
            util.hot_max as f64 / 1024.0 / 1024.0,
            (util.hot_used as f64 / util.hot_max as f64) * 100.0
        );
        println!("  Layers: {}", util.hot_count);

        println!("\n🟡 Warm Tier (SSD)");
        println!(
            "  Used: {:.2} MB / {:.2} MB ({:.1}%)",
            util.warm_used as f64 / 1024.0 / 1024.0,
            util.warm_max as f64 / 1024.0 / 1024.0,
            (util.warm_used as f64 / util.warm_max as f64) * 100.0
        );
        println!("  Layers: {}", util.warm_count);

        println!("\n❄️  Cold Tier (HDD/Network)");
        println!("  Layers: {}", util.cold_count);

        println!("\n📈 Access Statistics");
        println!("  Hot tier hits: {}", self.stats.hot_hits);
        println!("  Warm tier hits: {}", self.stats.warm_hits);
        println!("  Cold tier hits: {}", self.stats.cold_hits);
        println!("  Total accesses: {}", self.stats.total_accessed);

        if self.stats.total_accessed > 0 {
            let hot_ratio = self.stats.hot_hits as f64 / self.stats.total_accessed as f64;
            let warm_ratio = self.stats.warm_hits as f64 / self.stats.total_accessed as f64;
            println!(
                "  Hit rate: Hot {:.1}%, Warm {:.1}%",
                hot_ratio * 100.0,
                warm_ratio * 100.0
            );
        }

        println!("\n🔄 Migration Statistics");
        println!("  Promotions: {}", self.stats.promotions);
        println!("  Demotions: {}", self.stats.demotions);
        println!("  Evictions: {}", self.stats.evictions);
        println!("  Prefetch requests: {}", self.stats.prefetch_requests);
    }
}

#[derive(Debug, Clone)]
pub struct TierUtilization {
    pub hot_used: u64,
    pub hot_max: u64,
    pub warm_used: u64,
    pub warm_max: u64,
    pub hot_count: usize,
    pub warm_count: usize,
    pub cold_count: usize,
}

fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_register_and_access() {
        let temp_dir = TempDir::new().unwrap();
        let config = TierConfig::default();
        let mut storage = MultiTierStorage::new(
            config,
            temp_dir.path().join("hot"),
            temp_dir.path().join("warm"),
            temp_dir.path().join("cold"),
        )
        .unwrap();

        storage
            .register_layer(
                "layer_0".to_string(),
                1024 * 1024,
                StorageTier::Critical,
                temp_dir.path().join("layer_0.bin"),
            )
            .unwrap();

        assert!(storage.access_layer("layer_0").is_ok());
        assert_eq!(storage.stats.hot_hits, 1);
    }

    #[test]
    fn test_tier_overflow() {
        let temp_dir = TempDir::new().unwrap();
        let mut config = TierConfig::default();
        config.hot_max_bytes = 100 * 1024; // 100 KB

        let mut storage = MultiTierStorage::new(
            config,
            temp_dir.path().join("hot"),
            temp_dir.path().join("warm"),
            temp_dir.path().join("cold"),
        )
        .unwrap();

        // Try to register too large layer
        let result = storage.register_layer(
            "large_layer".to_string(),
            1024 * 1024,
            StorageTier::Critical,
            temp_dir.path().join("large.bin"),
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_layer_promotion_and_lru_eviction() {
        let temp_dir = TempDir::new().unwrap();
        let mut config = TierConfig::default();
        config.hot_max_bytes = 100 * 1024; // 100 KB
        config.warm_max_bytes = 500 * 1024; // 500 KB
        config.promotion_threshold = 2; // Promote on second access

        let mut storage = MultiTierStorage::new(
            config,
            temp_dir.path().join("hot"),
            temp_dir.path().join("warm"),
            temp_dir.path().join("cold"),
        )
        .unwrap();

        // Register layer in warm tier (SSD)
        storage
            .register_layer(
                "layer_warm".to_string(),
                40 * 1024, // 40 KB
                StorageTier::Important,
                temp_dir.path().join("warm.bin"),
            )
            .unwrap();

        // Verify initial state
        assert!(storage.warm_tier.contains_key("layer_warm"));
        assert!(!storage.hot_tier.contains_key("layer_warm"));

        // Access 1: should increase access count but not promote
        storage.access_layer("layer_warm").unwrap();
        assert_eq!(storage.stats.warm_hits, 1);
        assert!(!storage.hot_tier.contains_key("layer_warm"));

        // Access 2: should promote to hot tier since promotion_threshold = 2
        storage.access_layer("layer_warm").unwrap();
        assert_eq!(storage.stats.warm_hits, 2);
        assert_eq!(storage.stats.promotions, 1);
        assert!(storage.hot_tier.contains_key("layer_warm"));
        assert!(!storage.warm_tier.contains_key("layer_warm"));
        assert_eq!(storage.hot_used_bytes, 40 * 1024);

        // Try to register another layer directly in Hot tier that overflows it
        storage
            .register_layer(
                "layer_hot_new".to_string(),
                70 * 1024, // 70 KB. Hot tier total capacity = 100 KB. 40 + 70 = 110 KB.
                StorageTier::Critical,
                temp_dir.path().join("hot_new.bin"),
            )
            .unwrap_err(); // Initial placement overflow is an error

        // But promotion can trigger eviction. Let's register "layer_lru" in Warm, promote it, and show eviction.
        storage
            .register_layer(
                "layer_lru".to_string(),
                70 * 1024, // 70 KB
                StorageTier::Important,
                temp_dir.path().join("lru.bin"),
            )
            .unwrap();

        // We want to promote "layer_lru" to Hot tier.
        // Hot tier currently has: "layer_warm" (40 KB).
        // Promoting "layer_lru" (70 KB) would result in 40 + 70 = 110 KB (overflow).
        // This should evict the LRU ("layer_warm") from Hot back to Warm.

        // Wait, to make sure "layer_warm" is indeed LRU, we update its last_accessed to be older.
        if let Some(entry) = storage.hot_tier.get_mut("layer_warm") {
            entry.last_accessed -= 100;
        }

        // Access layer_lru twice to promote it
        storage.access_layer("layer_lru").unwrap();
        storage.access_layer("layer_lru").unwrap();

        // Check if promoted
        assert!(storage.hot_tier.contains_key("layer_lru"));
        // Check if layer_warm got evicted from hot
        assert!(!storage.hot_tier.contains_key("layer_warm"));
        // And demoted back to warm
        assert!(storage.warm_tier.contains_key("layer_warm"));
        assert_eq!(storage.stats.evictions, 1);
    }

    #[test]
    fn test_demote_inactive() {
        let temp_dir = TempDir::new().unwrap();
        let mut config = TierConfig::default();
        config.demotion_threshold_secs = 10; // 10 seconds

        let mut storage = MultiTierStorage::new(
            config,
            temp_dir.path().join("hot"),
            temp_dir.path().join("warm"),
            temp_dir.path().join("cold"),
        )
        .unwrap();

        storage
            .register_layer(
                "layer_ssd".to_string(),
                1024,
                StorageTier::Important,
                temp_dir.path().join("ssd.bin"),
            )
            .unwrap();

        // Artificially age the entry
        if let Some(entry) = storage.warm_tier.get_mut("layer_ssd") {
            entry.last_accessed -= 20; // 20 seconds ago (exceeds threshold 10)
        }

        storage.demote_inactive();

        // Should be demoted from warm to cold
        assert!(!storage.warm_tier.contains_key("layer_ssd"));
        assert!(storage.cold_tier.contains_key("layer_ssd"));
        assert_eq!(storage.stats.demotions, 1);
    }

    #[test]
    fn test_prefetch_layers() {
        let temp_dir = TempDir::new().unwrap();
        let mut config = TierConfig::default();
        config.prefetch_distance = 2;

        let mut storage = MultiTierStorage::new(
            config,
            temp_dir.path().join("hot"),
            temp_dir.path().join("warm"),
            temp_dir.path().join("cold"),
        )
        .unwrap();

        // Prefetching non-existent layers should return them as candidate files to load
        let next_layers = vec![
            "layer_1".to_string(),
            "layer_2".to_string(),
            "layer_3".to_string(),
        ];
        let prefetch_reqs = storage.prefetch_layers(&next_layers);

        assert_eq!(prefetch_reqs.len(), 2); // limited by prefetch_distance
        assert_eq!(prefetch_reqs[0], "layer_1");
        assert_eq!(prefetch_reqs[1], "layer_2");
        assert_eq!(storage.stats.prefetch_requests, 2);
    }
}
