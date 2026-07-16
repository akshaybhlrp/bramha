use std::fs;
use std::path::Path;

/// Manages eviction and disk storage limits for prefix KV caches.
pub struct PrefixCacheDb;

impl PrefixCacheDb {
    /// Enforces disk capacity limits by evicting the oldest modified prefix cache files.
    pub fn enforce_limit(model_path: &Path, max_entries: usize) -> Result<(), String> {
        let cache_dir = model_path.join("prefix_kv_cache_data");
        if !cache_dir.exists() {
            return Ok(());
        }

        let mut entries = Vec::new();
        let paths = fs::read_dir(&cache_dir).map_err(|e| e.to_string())?;
        for path_res in paths {
            let entry = path_res.map_err(|e| e.to_string())?;
            let metadata = entry.metadata().map_err(|e| e.to_string())?;
            if metadata.is_file() {
                let p = entry.path();
                if let Ok(modified) = metadata.modified() {
                    entries.push((p, modified));
                }
            }
        }

        if entries.len() > max_entries {
            // Sort by modified time (oldest first)
            entries.sort_by_key(|e| e.1);
            let to_remove = entries.len() - max_entries;
            for i in 0..to_remove {
                let _ = fs::remove_file(&entries[i].0);
            }
            println!(
                "🗑️ [Prefix KV Cache] Evicted {} oldest cache files to remain under threshold.",
                to_remove
            );
        }
        Ok(())
    }
}

use serde::{Deserialize, Serialize};
use std::fs::{self as fs2, File};
use std::io::{Read, Write};

// ─── KV Cache Entry ──────────────────────────────────────────────────────────

/// A serialisable KV-cache snapshot for one session or prefill prefix.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct KvCacheEntry {
    pub session_id: String,
    pub tokens: Vec<u32>,
    /// Per-layer key vectors, flattened: [layers][seq_len * kv_dim]
    pub keys: Vec<Vec<f32>>,
    /// Per-layer value vectors, flattened: [layers][seq_len * kv_dim]
    pub values: Vec<Vec<f32>>,
    pub last_accessed: u64,
    pub ttl_expiry: u64,
}

// ─── KV Cache Manager ────────────────────────────────────────────────────────

/// Stores/retrieves serialised KV-cache entries from a local directory.
pub struct KvCacheManager {
    dir: std::path::PathBuf,
    ttl_secs: u64,
}

impl Default for KvCacheManager {
    fn default() -> Self {
        Self::new_with_dir(22, 3600, "kv_cache_data")
    }
}

impl KvCacheManager {
    pub fn new_with_dir(
        _max_layers: usize,
        ttl_secs: u64,
        dir: impl AsRef<std::path::Path>,
    ) -> Self {
        let dir = dir.as_ref().to_path_buf();
        let _ = fs2::create_dir_all(&dir);
        Self { dir, ttl_secs }
    }

    fn path_for(&self, session_id: &str) -> std::path::PathBuf {
        self.dir.join(format!("{}.bin", session_id))
    }

    pub fn store(
        &self,
        session_id: String,
        tokens: Vec<u32>,
        keys: Vec<Vec<f32>>,
        values: Vec<Vec<f32>>,
        _memory_pressure: bool,
    ) -> Result<(), String> {
        let values_stored = values;

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let entry = KvCacheEntry {
            session_id: session_id.clone(),
            tokens,
            keys,
            values: values_stored,
            last_accessed: now,
            ttl_expiry: now + self.ttl_secs,
        };

        let encoded = bincode::serde::encode_to_vec(&entry, bincode::config::standard())
            .map_err(|e| e.to_string())?;
        let mut file = File::create(self.path_for(&session_id)).map_err(|e| e.to_string())?;
        file.write_all(&encoded).map_err(|e| e.to_string())
    }

    pub fn retrieve(&self, session_id: &str) -> Result<Option<KvCacheEntry>, String> {
        let path = self.path_for(session_id);
        if !path.exists() {
            return Ok(None);
        }
        let mut buf = Vec::new();
        File::open(&path)
            .and_then(|mut f| f.read_to_end(&mut buf))
            .map_err(|e| e.to_string())?;
        let (entry, _): (KvCacheEntry, _) =
            bincode::serde::decode_from_slice(&buf, bincode::config::standard())
                .map_err(|e| e.to_string())?;

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        if entry.ttl_expiry > 0 && now > entry.ttl_expiry {
            let _ = fs2::remove_file(&path);
            return Ok(None);
        }
        Ok(Some(entry))
    }

    pub fn clear(&self) -> Result<(), String> {
        for entry in fs2::read_dir(&self.dir).map_err(|e| e.to_string())? {
            let entry = entry.map_err(|e| e.to_string())?;
            if entry.path().extension().and_then(|e| e.to_str()) == Some("bin") {
                let _ = fs2::remove_file(entry.path());
            }
        }
        Ok(())
    }
}
