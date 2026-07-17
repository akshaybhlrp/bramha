use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::Path;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PrefixCacheEntry {
    pub tokens: Vec<u32>,
    pub keys: Vec<Vec<f32>>, // keys[layer_idx] contains sliced flat f32 data
    pub values: Vec<Vec<f32>>, // values[layer_idx] contains sliced flat f32 data
    pub last_accessed: u64,
}

/// Compute SHA-256 hash of a token sequence.
pub fn compute_tokens_hash(tokens: &[u32]) -> String {
    let mut hasher = Sha256::new();
    for &t in tokens {
        hasher.update(t.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn load_entry(path: &Path) -> Result<PrefixCacheEntry, String> {
    let mut file = File::open(path).map_err(|e| e.to_string())?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).map_err(|e| e.to_string())?;
    let config = bincode::config::standard();
    let entry: PrefixCacheEntry = bincode::serde::decode_from_slice(&buffer, config)
        .map_err(|e| e.to_string())?
        .0;
    Ok(entry)
}

/// Searches the prefix cache directory for the longest matching prefix of the token sequence.
/// Checks lengths in multiples of PAGE_SIZE (16 tokens) from longest down to 16.
pub fn find_longest_prefix(base_path: &Path, tokens: &[u32]) -> Option<(usize, PrefixCacheEntry)> {
    if std::env::var("BRAMHA_PREFIX_CACHE").unwrap_or_default() == "false" {
        return None;
    }

    let len = tokens.len();
    let page_size = 16;
    let max_pages = len / page_size;

    let cache_dir = base_path.join("prefix_kv_cache_data");
    if !cache_dir.exists() {
        return None;
    }

    // Search from longest prefix down to shortest page
    for pages in (1..=max_pages).rev() {
        let prefix_len = pages * page_size;
        let prefix = &tokens[..prefix_len];
        let hash = compute_tokens_hash(prefix);
        let path = cache_dir.join(format!("{}.bin", hash));

        if path.exists()
            && let Ok(mut entry) = load_entry(&path)
                && entry.tokens == prefix {
                    // Update access time
                    entry.last_accessed = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_millis() as u64;
                    // Re-save asynchronously or ignore; simple in-memory access touch is fine,
                    // or re-save to keep modified time fresh for eviction
                    let _ = save_entry(&path, &entry);
                    return Some((prefix_len, entry));
                }
    }
    None
}

fn save_entry(path: &Path, entry: &PrefixCacheEntry) -> Result<(), String> {
    let config = bincode::config::standard();
    let encoded = bincode::serde::encode_to_vec(entry, config)
        .map_err(|e| format!("Failed to serialize prefix cache: {}", e))?;
    let mut file =
        File::create(path).map_err(|e| format!("Failed to create prefix cache: {}", e))?;
    file.write_all(&encoded)
        .map_err(|e| format!("Failed to write prefix cache: {}", e))?;
    Ok(())
}

/// Saves the KV cache state for prefixes of the token sequence up to the nearest page boundary.
pub fn save_prefix(
    base_path: &Path,
    tokens: &[u32],
    keys: &Vec<Vec<f32>>,
    values: &Vec<Vec<f32>>,
) -> Result<(), String> {
    if std::env::var("BRAMHA_PREFIX_CACHE").unwrap_or_default() == "false" {
        return Ok(());
    }

    let len = tokens.len();
    let trim_n = std::env::var("BRAMHA_KV_TRIM_N")
        .unwrap_or_default()
        .parse::<usize>()
        .unwrap_or(4);

    if len <= trim_n {
        return Ok(());
    }
    let active_len = len - trim_n;

    let page_size = 16;
    if active_len < page_size || keys.is_empty() || keys[0].is_empty() {
        return Ok(());
    }

    let cache_dir = base_path.join("prefix_kv_cache_data");
    fs::create_dir_all(&cache_dir).map_err(|e| e.to_string())?;

    // Determine key dimension per token
    let dim = keys[0].len() / len;
    if dim == 0 {
        return Ok(());
    }

    // Save prefixes at multiples of page_size (chunk boundaries)
    let max_pages = active_len / page_size;
    for pages in 1..=max_pages {
        let prefix_len = pages * page_size;
        let prefix = &tokens[..prefix_len];
        let hash = compute_tokens_hash(prefix);
        let path = cache_dir.join(format!("{}.bin", hash));

        // Skip if already cached on disk
        if path.exists() {
            continue;
        }

        let slice_len = prefix_len * dim;
        let mut prefix_keys = Vec::with_capacity(keys.len());
        let mut prefix_values = Vec::with_capacity(values.len());

        for layer_idx in 0..keys.len() {
            if keys[layer_idx].len() < slice_len || values[layer_idx].len() < slice_len {
                return Err("KV length is shorter than sliced prefix length".to_string());
            }
            prefix_keys.push(keys[layer_idx][..slice_len].to_vec());
            prefix_values.push(values[layer_idx][..slice_len].to_vec());
        }

        let entry = PrefixCacheEntry {
            tokens: prefix.to_vec(),
            keys: prefix_keys,
            values: prefix_values,
            last_accessed: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        };

        save_entry(&path, &entry)?;
    }

    // Enforce storage limits
    let _ = crate::storage::cache_db::PrefixCacheDb::enforce_limit(base_path, 20); // Keep max 20 prefixes

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prefix_tokens_hash() {
        let tokens1 = vec![1, 2, 3, 4];
        let tokens2 = vec![1, 2, 3, 4];
        let tokens3 = vec![1, 2, 3, 5];

        assert_eq!(compute_tokens_hash(&tokens1), compute_tokens_hash(&tokens2));
        assert_ne!(compute_tokens_hash(&tokens1), compute_tokens_hash(&tokens3));
    }

    #[test]
    fn test_rolling_prefix_matching_saving_eviction() {
        unsafe {
            std::env::set_var("BRAMHA_KV_TRIM_N", "0");
        }
        let temp_dir = std::env::temp_dir().join("bramha_prefix_cache_test");
        let _ = fs::remove_dir_all(&temp_dir);
        fs::create_dir_all(&temp_dir).unwrap();

        // 1. Create dummy tokens and layer keys/values (length 35, dim = 2)
        let tokens: Vec<u32> = (1..=35).collect();
        let keys = vec![vec![0.5f32; 35 * 2], vec![0.7f32; 35 * 2]];
        let values = vec![vec![1.5f32; 35 * 2], vec![1.7f32; 35 * 2]];

        // 2. Initially no prefix match exists
        let match_opt = find_longest_prefix(&temp_dir, &tokens);
        assert!(match_opt.is_none(), "Expected no prefix match initially");

        // 3. Save prefix caches (should save prefixes of length 16 and 32)
        save_prefix(&temp_dir, &tokens, &keys, &values).unwrap();

        // Verify files exist in directory
        let cache_dir = temp_dir.join("prefix_kv_cache_data");
        assert!(cache_dir.exists());

        // 4. Match longest prefix for identical tokens sequence (should hit page length 32)
        let match_hit = find_longest_prefix(&temp_dir, &tokens).unwrap();
        assert_eq!(match_hit.0, 32);
        assert_eq!(match_hit.1.tokens, tokens[..32].to_vec());
        assert_eq!(match_hit.1.keys.len(), 2);
        assert_eq!(match_hit.1.keys[0].len(), 32 * 2);
        assert_eq!(match_hit.1.keys[0], vec![0.5f32; 64]);

        // 5. Match longest prefix for partially matching sequence [1..=20] (should hit page length 16)
        let partial_tokens: Vec<u32> = (1..=20).collect();
        let match_hit_partial = find_longest_prefix(&temp_dir, &partial_tokens).unwrap();
        assert_eq!(match_hit_partial.0, 16);
        assert_eq!(match_hit_partial.1.tokens, tokens[..16].to_vec());

        // 6. Match for completely mismatched sequence (should miss fallback)
        let mismatch_tokens = vec![
            99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
        ];
        let mismatch_hit = find_longest_prefix(&temp_dir, &mismatch_tokens);
        assert!(
            mismatch_hit.is_none(),
            "Expected miss fallback for mismatched tokens"
        );

        // 7. Verify eviction limits (limit max_entries to 1)
        crate::storage::cache_db::PrefixCacheDb::enforce_limit(&temp_dir, 1).unwrap();
        let paths = fs::read_dir(&cache_dir).unwrap();
        let count = paths
            .filter_map(|p| p.ok())
            .filter(|e| e.metadata().map(|m| m.is_file()).unwrap_or(false))
            .count();
        assert_eq!(
            count, 1,
            "Eviction should bound cache files count to exactly 1"
        );

        // Cleanup
        let _ = fs::remove_dir_all(&temp_dir);
        unsafe {
            std::env::remove_var("BRAMHA_KV_TRIM_N");
        }
    }

    #[test]
    fn test_boundary_aligned_trimming() {
        unsafe {
            std::env::set_var("BRAMHA_KV_TRIM_N", "4");
        }
        let temp_dir = std::env::temp_dir().join("bramha_trim_test");
        let _ = fs::remove_dir_all(&temp_dir);
        fs::create_dir_all(&temp_dir).unwrap();

        // Tokens length 35: active_len = 31. Aligned to page boundary (16) = page 1 (length 16) only!
        let tokens: Vec<u32> = (1..=35).collect();
        let keys = vec![vec![0.1f32; 35 * 2]];
        let values = vec![vec![0.2f32; 35 * 2]];

        save_prefix(&temp_dir, &tokens, &keys, &values).unwrap();

        // Should find longest prefix at 16, NOT 32 because 32 was trimmed!
        let match_hit = find_longest_prefix(&temp_dir, &tokens).unwrap();
        assert_eq!(match_hit.0, 16);

        let _ = fs::remove_dir_all(&temp_dir);
        unsafe {
            std::env::remove_var("BRAMHA_KV_TRIM_N");
        }
    }
}
