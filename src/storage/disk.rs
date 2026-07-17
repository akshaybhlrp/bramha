use serde::{Serialize, de::DeserializeOwned};
use std::fs::File;
use std::io::{Read, Write};

/// Saves any serializable data structure to a JSON file atomically and crash-safely.
pub fn save_to_file<T: Serialize>(data: &T, path: &str) -> Result<(), String> {
    let bytes =
        serde_json::to_vec_pretty(data).map_err(|e| format!("Failed to serialize: {}", e))?;
    let tmp_path = format!("{}.tmp", path);
    let mut file =
        File::create(&tmp_path).map_err(|e| format!("Failed to create temp file: {}", e))?;
    file.write_all(&bytes)
        .map_err(|e| format!("Failed to write to temp file: {}", e))?;
    file.sync_all()
        .map_err(|e| format!("Failed to sync temp file: {}", e))?;
    std::fs::rename(&tmp_path, path).map_err(|e| format!("Failed to atomic rename: {}", e))?;
    Ok(())
}

/// Loads a serialized data structure from a file, trying JSON first, then falling back to bincode.
pub fn load_from_file<T: DeserializeOwned>(path: &str) -> Result<T, String> {
    let mut file = File::open(path).map_err(|e| format!("Failed to open file: {}", e))?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .map_err(|e| format!("Failed to read file: {}", e))?;

    // 1. Try parsing as JSON first
    if let Ok(data) = serde_json::from_slice::<T>(&bytes) {
        return Ok(data);
    }

    // 1.5. Resilient fallback: If it looks like JSON but has trailing garbage/corruption, attempt self-healing recovery
    let trimmed = bytes.iter().position(|&b| !b.is_ascii_whitespace());
    if let Some(first_char_idx) = trimmed
        && (bytes[first_char_idx] == b'{' || bytes[first_char_idx] == b'[')
    {
        // Find the last closing brace/bracket
        let target_char = if bytes[first_char_idx] == b'{' {
            b'}'
        } else {
            b']'
        };
        if let Some(last_idx) = bytes.iter().rposition(|&b| b == target_char) {
            let stripped_bytes = &bytes[..=last_idx];
            if let Ok(data) = serde_json::from_slice::<T>(stripped_bytes) {
                println!(
                    "🛡️ Resilient Parser: Recovered JSON database from trailing corruption/tampering!"
                );
                return Ok(data);
            }
        }
    }

    // 2. Fallback to legacy bincode compatibility
    let config = bincode::config::standard();
    match bincode::serde::decode_from_slice::<T, _>(&bytes, config) {
        Ok((data, _)) => Ok(data),
        Err(e) => Err(format!("Failed to deserialize as JSON or bincode: {}", e)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Serialize, Deserialize, Debug, PartialEq, Eq)]
    struct DummyData {
        name: String,
        value: i32,
    }

    #[test]
    fn test_json_persistence_roundtrip() {
        let temp_dir = std::env::temp_dir();
        let test_path = temp_dir.join("test_bramha_db_json.json");
        let path_str = test_path.to_str().unwrap();

        let original = DummyData {
            name: "test_entity".to_string(),
            value: 42,
        };

        // Save as JSON
        save_to_file(&original, path_str).unwrap();

        // Check it was saved as JSON string
        let content = std::fs::read_to_string(path_str).unwrap();
        assert!(content.contains("\"name\": \"test_entity\""));
        assert!(content.contains("\"value\": 42"));

        // Load back
        let loaded: DummyData = load_from_file(path_str).unwrap();
        assert_eq!(original, loaded);

        let _ = std::fs::remove_file(path_str);
    }

    #[test]
    fn test_legacy_bincode_deserialization_fallback() {
        let temp_dir = std::env::temp_dir();
        let test_path = temp_dir.join("test_bramha_db_legacy.bin");
        let path_str = test_path.to_str().unwrap();

        let original = DummyData {
            name: "legacy_entity".to_string(),
            value: 99,
        };

        // Write as legacy bincode file
        let config = bincode::config::standard();
        let bytes = bincode::serde::encode_to_vec(&original, config).unwrap();
        std::fs::write(path_str, bytes).unwrap();

        // Load back using our resilient parser (should fall back to bincode)
        let loaded: DummyData = load_from_file(path_str).unwrap();
        assert_eq!(original, loaded);

        let _ = std::fs::remove_file(path_str);
    }
}
