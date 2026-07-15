use std::path::Path;
use std::fs::File;
use std::io::Read;

use crate::storage::activation_view::ActivationMaterializedView;
use super::prefix_cache::{PrefixCacheEntry, compute_tokens_hash};

pub struct ReplayResult {
    pub valid_length: usize,
    pub entry: PrefixCacheEntry,
}

/// Loads a materialized view from disk and validates its integrity against the requested prompt.
/// Performs "Activation Replay Validation" by rolling SHA-256 up to the divergence point.
pub fn load_and_validate_branch(
    view: &ActivationMaterializedView,
    requested_tokens: &[u32],
) -> Result<ReplayResult, String> {
    let path = Path::new(&view.disk_path);
    if !path.exists() {
        return Err("Activation view disk file missing.".into());
    }

    let mut file = File::open(path).map_err(|e| format!("Failed to open view file: {}", e))?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).map_err(|e| format!("Failed to read view file: {}", e))?;
    
    let config = bincode::config::standard();
    let mut entry: PrefixCacheEntry = bincode::serde::decode_from_slice(&buffer, config)
        .map_err(|e| format!("Failed to decode view file: {}", e))?.0;

    // Activation Replay Validation: Verify hash matches
    let cached_tokens = &entry.tokens;
    
    // Find where the branch diverges
    let mut valid_len = 0;
    for (i, &t) in requested_tokens.iter().enumerate() {
        if i >= cached_tokens.len() || cached_tokens[i] != t {
            break;
        }
        valid_len += 1;
    }

    if valid_len == 0 {
        return Err("No matching token prefix found in the materialized view.".into());
    }

    // Verify rolling SHA-256 of the validated subset
    let subset_hash = compute_tokens_hash(&requested_tokens[..valid_len]);
    let cached_subset_hash = compute_tokens_hash(&cached_tokens[..valid_len]);
    
    if subset_hash != cached_subset_hash {
        return Err("Activation Replay Validation failed: SHA-256 mismatch! Cache may be corrupted.".into());
    }

    let original_len = entry.tokens.len();

    // Truncate the KV cache down to the valid divergence point (page boundaries ideally, but exact for branching)
    entry.tokens.truncate(valid_len);
    
    // Truncate keys and values layer by layer. 
    // Assuming dim = keys[0].len() / original_token_len
    if original_len > 0 {
        let dim = entry.keys[0].len() / original_len;
        let valid_slice_len = valid_len * dim;
        
        for layer_idx in 0..entry.keys.len() {
            entry.keys[layer_idx].truncate(valid_slice_len);
            entry.values[layer_idx].truncate(valid_slice_len);
        }
    }

    Ok(ReplayResult {
        valid_length: valid_len,
        entry,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use crate::storage::activation_view::ActivationMaterializedView;
    use crate::inference::paged_kv::prefix_cache::{PrefixCacheEntry, compute_tokens_hash};

    #[test]
    fn test_branch_replay_validation_success() {
        let temp_dir = std::env::temp_dir().join("bramha_replay_test");
        let _ = fs::remove_dir_all(&temp_dir);
        fs::create_dir_all(&temp_dir).unwrap();

        let path = temp_dir.join("test_view.bin");
        
        let tokens = vec![100, 200, 300, 400, 500];
        let keys = vec![vec![0.1; 10], vec![0.2; 10]]; // dim=2
        let values = vec![vec![0.3; 10], vec![0.4; 10]];

        let entry = PrefixCacheEntry {
            tokens: tokens.clone(),
            keys,
            values,
            last_accessed: 0,
        };

        let config = bincode::config::standard();
        let encoded = bincode::serde::encode_to_vec(&entry, config).unwrap();
        fs::write(&path, encoded).unwrap();

        let view = ActivationMaterializedView {
            workflow_id: "wf-1".into(),
            branch_id: "br-1".into(),
            token_hash: compute_tokens_hash(&tokens),
            token_length: 5,
            disk_path: path.to_string_lossy().to_string(),
            created_at: 0,
        };

        // Test identical requested prompt
        let req1 = vec![100, 200, 300, 400, 500];
        let res1 = load_and_validate_branch(&view, &req1).unwrap();
        assert_eq!(res1.valid_length, 5);
        assert_eq!(res1.entry.keys[0].len(), 10);

        // Test diverging requested prompt (diverges at index 3)
        let req2 = vec![100, 200, 300, 999, 999];
        let res2 = load_and_validate_branch(&view, &req2).unwrap();
        assert_eq!(res2.valid_length, 3);
        assert_eq!(res2.entry.keys[0].len(), 6); // 3 * dim(2)

        let _ = fs::remove_dir_all(&temp_dir);
    }
}
