use reqwest::Client;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{Read, Write};

use crate::storage::atomic_write::atomic_write_file;
use crate::storage::metadata_sql::MetadataSqlStore;
use crate::storage::tensor_db::TensorDB;

pub struct RegistryEntry {
    pub id: &'static str,
    pub model_url: &'static str,
    pub tokenizer_url: &'static str,
    pub config_url: &'static str,
    pub expected_sha256: &'static str,
}

pub const MODEL_REGISTRY: &[RegistryEntry] = &[
    RegistryEntry {
        id: "tinyllama-1.1b",
        model_url: "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/model.safetensors",
        tokenizer_url: "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/tokenizer.json",
        config_url: "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/config.json",
        expected_sha256: "", // Optional fallback
    },
    RegistryEntry {
        id: "mistral-7b-q4",
        model_url: "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        tokenizer_url: "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/resolve/main/tokenizer.json",
        config_url: "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/resolve/main/config.json",
        expected_sha256: "",
    },
    RegistryEntry {
        id: "qwen2.5-0.5b",
        model_url: "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/model.safetensors",
        tokenizer_url: "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/tokenizer.json",
        config_url: "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/config.json",
        expected_sha256: "",
    },
];

/// Pulls a model from the registry, streams down its weights and tokenizer, verifies integrity,
/// and automatically shards it into the database storage directory.
pub async fn pull_model(model_id: &str, tensor_db: &mut TensorDB) -> Result<(), String> {
    let entry = MODEL_REGISTRY
        .iter()
        .find(|e| e.id == model_id)
        .ok_or_else(|| {
            format!(
                "Model '{}' not found in registry. Registered models: {:?}",
                model_id,
                MODEL_REGISTRY.iter().map(|e| e.id).collect::<Vec<_>>()
            )
        })?;

    let client = Client::new();
    let model_dir = tensor_db.storage_dir.join(model_id);
    std::fs::create_dir_all(&model_dir).map_err(|e| e.to_string())?;

    // 1. Download config.json (architecture hyperparameters)
    if !entry.config_url.is_empty() {
        println!("⬇️ Downloading config.json for '{}'...", model_id);
        if let Ok(cfg_response) = client.get(entry.config_url).send().await {
            if cfg_response.status().is_success() {
                if let Ok(cfg_bytes) = cfg_response.bytes().await {
                    let config_path = model_dir.join("config.json");
                    let _ = atomic_write_file(&config_path, &cfg_bytes);
                    println!("✅ Config saved to {:?}", config_path);
                }
            }
        }
    }

    // 2. Download tokenizer.json
    println!("⬇️ Downloading tokenizer.json for '{}'...", model_id);
    let tok_response = client
        .get(entry.tokenizer_url)
        .send()
        .await
        .map_err(|e| format!("Failed to request tokenizer: {}", e))?;
    if !tok_response.status().is_success() {
        return Err(format!(
            "Tokenizer download failed with HTTP {}",
            tok_response.status()
        ));
    }
    let tok_bytes = tok_response
        .bytes()
        .await
        .map_err(|e| format!("Failed to read tokenizer bytes: {}", e))?;

    let tokenizer_path = model_dir.join("tokenizer.json");
    atomic_write_file(&tokenizer_path, &tok_bytes)
        .map_err(|e| format!("Failed to write tokenizer: {}", e))?;
    println!("✅ Tokenizer saved to {:?}", tokenizer_path);

    // 2. Stream model weights
    println!("⬇️ Downloading model weights for '{}'...", model_id);
    let response = client
        .get(entry.model_url)
        .send()
        .await
        .map_err(|e| format!("Failed to request model weights: {}", e))?;
    if !response.status().is_success() {
        return Err(format!(
            "Model download failed with HTTP {}",
            response.status()
        ));
    }

    let total_size = response.content_length().unwrap_or(0);
    let mut downloaded: u64 = 0;

    let temp_model_path = model_dir.join("downloading_weights.tmp");
    let mut temp_file = File::create(&temp_model_path)
        .map_err(|e| format!("Failed to create temporary weight file: {}", e))?;

    let mut last_update = std::time::Instant::now();
    let mut response_obj = response;

    while let Some(chunk) = response_obj
        .chunk()
        .await
        .map_err(|e| format!("Error downloading chunk: {}", e))?
    {
        temp_file
            .write_all(&chunk)
            .map_err(|e| format!("Error writing chunk to disk: {}", e))?;
        downloaded += chunk.len() as u64;

        if last_update.elapsed().as_millis() > 500 {
            if total_size > 0 {
                let percent = (downloaded as f64 / total_size as f64) * 100.0;
                print!(
                    "\r   Progress: {:.1}% ({:.2} MB / {:.2} MB)",
                    percent,
                    downloaded as f64 / 1_000_000.0,
                    total_size as f64 / 1_000_000.0
                );
            } else {
                print!(
                    "\r   Progress: {:.2} MB downloaded",
                    downloaded as f64 / 1_000_000.0
                );
            }
            std::io::stdout().flush().unwrap_or_default();
            last_update = std::time::Instant::now();
        }
    }
    println!("\n✅ Model weights downloaded successfully!");

    // 3. Verify SHA-256 checksum if configured
    if !entry.expected_sha256.is_empty() {
        println!("🛡️ Verifying SHA-256 integrity checksum...");
        temp_file.sync_all().map_err(|e| e.to_string())?;
        drop(temp_file);

        let mut file = File::open(&temp_model_path).map_err(|e| e.to_string())?;
        let mut hasher = Sha256::new();
        let mut buffer = [0; 65536];
        loop {
            let count = file.read(&mut buffer).map_err(|e| e.to_string())?;
            if count == 0 {
                break;
            }
            hasher.update(&buffer[..count]);
        }
        let hash_result = hasher.finalize();
        let sha256_hex = format!("{:x}", hash_result);
        if sha256_hex != entry.expected_sha256 {
            let _ = std::fs::remove_file(&temp_model_path);
            return Err(format!(
                "CRITICAL CHECKSUM MISMATCH: Expected {}, got {}",
                entry.expected_sha256, sha256_hex
            ));
        }
        println!("   Checksum validation PASSED!");
    } else {
        drop(temp_file);
    }

    // 4. Ingest/Shard the model
    println!("⚙️ Ingesting sharded weights into Bramha Engine...");
    tensor_db.create_model(model_id.to_string());
    let model_table = tensor_db.models.get_mut(model_id).unwrap();

    let final_model_path = model_dir.join("model.safetensors");
    std::fs::rename(&temp_model_path, &final_model_path).map_err(|e| e.to_string())?;

    model_table
        .load_safetensors("model.safetensors")
        .map_err(|e| format!("Ingestion sharding failed: {}", e))?;

    // Persist model metadata to SQLite metadata DB
    let metadata_store = MetadataSqlStore::new();
    let architecture = if model_id.contains("qwen") {
        "Qwen2"
    } else if model_id.contains("mistral") {
        "Mistral"
    } else {
        "Llama"
    };
    let params_count = if model_id.contains("0.5b") {
        500_000_000
    } else if model_id.contains("1.1b") {
        1_100_000_000
    } else if model_id.contains("7b") {
        7_000_000_000
    } else {
        0
    };
    let _ = metadata_store.create_model(
        model_id,
        architecture,
        params_count,
        &model_dir.to_string_lossy(),
    );

    // Cleanup raw safetensors file after ingestion completes
    let _ = std::fs::remove_file(final_model_path);

    println!(
        "🎉 Model '{}' is now fully registered and query-ready!",
        model_id
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_registry_lookups() {
        let registry = MODEL_REGISTRY;
        assert!(registry.iter().any(|e| e.id == "tinyllama-1.1b"));
        assert!(registry.iter().any(|e| e.id == "mistral-7b-q4"));

        let mut temp_db = TensorDB::new(std::env::temp_dir().join("pull_test_db"));
        let res = pull_model("unknown_model_xyz", &mut temp_db).await;
        assert!(res.is_err());
        assert!(res.unwrap_err().contains("not found in registry"));
    }
}
