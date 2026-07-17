use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedResponse {
    pub prompt: String,
    pub model_version: String,
    pub context_hash: String,
    pub completion: String,
    pub timestamp_ms: u64,
}

pub struct DeterministicAnswerCache {
    pub entries: Mutex<HashMap<String, CachedResponse>>,
    pub cache_path: PathBuf,
}

impl DeterministicAnswerCache {
    /// Computes a unique, deterministic hash of the query prompt, model, and active context chunks
    pub fn compute_context_hash(
        prompt: &str,
        model_version: &str,
        context_chunks: &[(String, String)],
    ) -> String {
        let mut hasher = DefaultHasher::new();
        prompt.hash(&mut hasher);
        model_version.hash(&mut hasher);
        for (id, text) in context_chunks {
            id.hash(&mut hasher);
            text.hash(&mut hasher);
        }
        format!("{:x}", hasher.finish())
    }

    /// Load or initialize the deterministic cache using the default path
    pub fn load() -> Self {
        let path = Path::new("cache").join("deterministic_answers.json");
        Self::load_from_path(&path)
    }

    /// Load or initialize the cache using a custom file path
    pub fn load_from_path(path: &Path) -> Self {
        let mut entries = HashMap::new();
        if path.exists()
            && let Ok(content) = fs::read_to_string(path)
                && let Ok(loaded) =
                    serde_json::from_str::<HashMap<String, CachedResponse>>(&content)
                {
                    entries = loaded;
                }
        Self {
            entries: Mutex::new(entries),
            cache_path: path.to_path_buf(),
        }
    }

    /// Retrieve a cached response if it exists, is structurally valid, and is within max_age limit
    pub fn get(
        &self,
        prompt: &str,
        model_version: &str,
        context_chunks: &[(String, String)],
        max_age_seconds: u64,
    ) -> Option<String> {
        let hash = Self::compute_context_hash(prompt, model_version, context_chunks);
        let mut guard = self.entries.lock().unwrap();

        if let Some(entry) = guard.get(&hash) {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;

            let age_ms = now.saturating_sub(entry.timestamp_ms);
            let max_age_ms = max_age_seconds * 1000;

            if age_ms <= max_age_ms {
                return Some(entry.completion.clone());
            } else {
                // Evict expired entry
                guard.remove(&hash);
                let _ = self.save_locked(&guard);
            }
        }
        None
    }

    /// Insert a new generated response into the cache and trigger automatic serialization
    pub fn insert(
        &self,
        prompt: &str,
        model_version: &str,
        context_chunks: &[(String, String)],
        completion: String,
    ) -> Result<(), String> {
        let hash = Self::compute_context_hash(prompt, model_version, context_chunks);
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let entry = CachedResponse {
            prompt: prompt.to_string(),
            model_version: model_version.to_string(),
            context_hash: hash.clone(),
            completion,
            timestamp_ms: now,
        };

        let mut guard = self.entries.lock().unwrap();
        guard.insert(hash, entry);
        self.save_locked(&guard)?;
        Ok(())
    }

    /// Save the cache state to disk thread-safely
    fn save_locked(&self, guard: &HashMap<String, CachedResponse>) -> Result<(), String> {
        let path = &self.cache_path;
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let serialized = serde_json::to_string_pretty(guard)
            .map_err(|e| format!("Failed to serialize cache: {}", e))?;
        fs::write(path, serialized).map_err(|e| format!("Failed to write cache file: {}", e))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_answer_cache_insert_retrieval_and_expiry() {
        let temp_dir = std::env::temp_dir().join("bramha_cache_test");
        let _ = fs::create_dir_all(&temp_dir);
        let test_file = temp_dir.join("test_deterministic_answers.json");
        let _ = fs::remove_file(&test_file);

        let cache = DeterministicAnswerCache::load_from_path(&test_file);

        let prompt = "Explain prefixed attention";
        let model = "llama-test";
        let context = vec![(
            "chunk1".to_string(),
            "Bramha runs local intelligence".to_string(),
        )];

        // 1. Initially empty
        assert!(cache.get(prompt, model, &context, 10).is_none());

        // 2. Insert and check retrieval
        cache
            .insert(prompt, model, &context, "cached reply".to_string())
            .unwrap();
        let hit = cache.get(prompt, model, &context, 10);
        assert_eq!(hit.unwrap(), "cached reply");

        // 3. Check cache mismatch on changed context
        let new_context = vec![(
            "chunk1".to_string(),
            "Bramha is accelerated via GPU".to_string(),
        )];
        assert!(cache.get(prompt, model, &new_context, 10).is_none());

        // 4. Check cache expiry (max_age = 2 seconds, but we manually subtract 5 seconds from timestamp)
        {
            let mut guard = cache.entries.lock().unwrap();
            let hash = DeterministicAnswerCache::compute_context_hash(prompt, model, &context);
            if let Some(entry) = guard.get_mut(&hash) {
                entry.timestamp_ms -= 5000;
            }
        }
        let expired = cache.get(prompt, model, &context, 2);
        assert!(expired.is_none());

        let _ = fs::remove_file(test_file);
    }
}
