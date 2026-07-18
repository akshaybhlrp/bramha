use crate::inference::paged_kv::prefix_cache::{
    PrefixCacheEntry, find_longest_prefix, save_prefix,
};
use std::path::{Path, PathBuf};

pub enum KVPersistenceMoment {
    Cold,
    Continued,
    Evict,
    Shutdown,
}

pub struct KVPersistenceManager {
    base_path: PathBuf,
}

impl KVPersistenceManager {
    pub fn new<P: AsRef<Path>>(base_path: P) -> Self {
        Self {
            base_path: base_path.as_ref().to_path_buf(),
        }
    }

    /// Trigger KV cache persistence at a specific moment in the session lifecycle.
    /// This delegates to boundary-aligned prefix cache saving.
    pub fn persist(
        &self,
        moment: KVPersistenceMoment,
        tokens: &[u32],
        keys: &Vec<Vec<f32>>,
        values: &Vec<Vec<f32>>,
    ) -> Result<(), String> {
        let moment_str = match moment {
            KVPersistenceMoment::Cold => "cold",
            KVPersistenceMoment::Continued => "continued",
            KVPersistenceMoment::Evict => "evict",
            KVPersistenceMoment::Shutdown => "shutdown",
        };

        println!(
            "💾 [KV Persistence] Saving KV cache on lifecycle moment: {}",
            moment_str
        );

        // This automatically handles boundary-aligned trimming and chunks.
        save_prefix(&self.base_path, tokens, keys, values)?;

        Ok(())
    }

    /// Load the longest matching KV cache prefix.
    pub fn load(&self, tokens: &[u32]) -> Option<(usize, PrefixCacheEntry)> {
        find_longest_prefix(&self.base_path, tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_kv_persistence_lifecycle() {
        let temp_dir = std::env::temp_dir().join("bramha_kv_persistence_lifecycle_test");
        let _ = fs::remove_dir_all(&temp_dir);
        fs::create_dir_all(&temp_dir).unwrap();

        // SAFETY: Manual invariants verified for performance/FFI
        unsafe {
            std::env::set_var("BRAMHA_PREFIX_CACHE", "true");
        }
        // SAFETY: Manual invariants verified for performance/FFI
        unsafe {
            std::env::set_var("BRAMHA_KV_TRIM_N", "0");
        }

        let manager = KVPersistenceManager::new(&temp_dir);

        let tokens = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let keys = vec![vec![0.5; 16]];
        let values = vec![vec![1.5; 16]];

        // Test Cold save
        manager
            .persist(KVPersistenceMoment::Cold, &tokens, &keys, &values)
            .unwrap();

        // Test Continued save
        manager
            .persist(KVPersistenceMoment::Continued, &tokens, &keys, &values)
            .unwrap();

        // Test Evict save
        manager
            .persist(KVPersistenceMoment::Evict, &tokens, &keys, &values)
            .unwrap();

        // Test Shutdown save
        manager
            .persist(KVPersistenceMoment::Shutdown, &tokens, &keys, &values)
            .unwrap();

        // Test Load
        let load_res = manager.load(&tokens);
        assert!(load_res.is_some());
        let (matched_len, entry) = load_res.unwrap();
        assert_eq!(matched_len, 16);
        assert_eq!(entry.tokens, tokens);

        let _ = fs::remove_dir_all(&temp_dir);
        // SAFETY: Manual invariants verified for performance/FFI
        unsafe {
            std::env::remove_var("BRAMHA_PREFIX_CACHE");
        }
        // SAFETY: Manual invariants verified for performance/FFI
        unsafe {
            std::env::remove_var("BRAMHA_KV_TRIM_N");
        }
    }
}
