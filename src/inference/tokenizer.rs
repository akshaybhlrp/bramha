use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

pub struct BramhaTokenizer {
    tokenizer: Tokenizer,
}

impl BramhaTokenizer {
    /// Resolves and loads the tokenizer config for a model in-process, scanning local folders
    /// and HuggingFace cache snapshots to guarantee a zero-IPC pure-Rust flow.
    pub fn load(model_name: &str, base_path: &Path) -> Result<Self, String> {
        let tokenizer_path = Self::resolve_path(model_name, base_path)?;

        // Port mistral.rs tokenizer fix: Ensure added_tokens are actually in the vocab
        // to prevent token_to_id from failing for special tokens.
        let raw = std::fs::read(&tokenizer_path)
            .map_err(|e| format!("Failed to read tokenizer: {}", e))?;
        let mut tokenizer_json: serde_json::Value =
            serde_json::from_slice(&raw).map_err(|e| format!("Failed to parse JSON: {}", e))?;

        let added_tokens_opt = tokenizer_json
            .get("added_tokens")
            .and_then(|v| v.as_array())
            .cloned();

        if let Some(added_tokens) = added_tokens_opt
            && let Some(vocab) = tokenizer_json
                .get_mut("model")
                .and_then(|m| m.get_mut("vocab"))
                .and_then(|v| v.as_object_mut())
        {
            for token in added_tokens {
                if let (Some(content), Some(id)) = (
                    token.get("content").and_then(|c| c.as_str()),
                    token.get("id"),
                ) && !vocab.contains_key(content)
                {
                    vocab.insert(content.to_string(), id.clone());
                }
            }
        }

        let raw_fixed = serde_json::to_vec(&tokenizer_json).map_err(|e| e.to_string())?;
        let tokenizer = Tokenizer::from_bytes(&raw_fixed)
            .map_err(|e| format!("Failed to parse fixed tokenizer.json: {}", e))?;

        Ok(BramhaTokenizer { tokenizer })
    }

    pub fn resolve_path(model_name: &str, base_path: &Path) -> Result<PathBuf, String> {
        // 1. FIRST: Check the Base directory of the model table
        let local_candidate = base_path.join("tokenizer.json");
        if local_candidate.is_file() {
            return Ok(local_candidate);
        }

        // 2. Check HuggingFace Cache Snapshots if not found locally
        let hf_home = std::env::var("HF_HOME").unwrap_or_else(|_| {
            format!(
                "{}/.cache/huggingface",
                std::env::var("HOME").unwrap_or_default()
            )
        });
        let hub_dir = format!("{}/hub", hf_home);
        let model_lower = model_name.to_lowercase();

        if let Ok(entries) = std::fs::read_dir(&hub_dir) {
            for entry in entries.flatten() {
                let dir_name = entry.file_name().to_string_lossy().to_lowercase();
                if dir_name.contains(&model_lower) {
                    let snapshots = entry.path().join("snapshots");
                    if snapshots.is_dir()
                        && let Ok(snap_entries) = std::fs::read_dir(&snapshots)
                    {
                        for snap in snap_entries.flatten() {
                            let candidate = snap.path().join("tokenizer.json");
                            if candidate.is_file() {
                                return Ok(candidate);
                            }
                        }
                    }
                }
            }
        }

        // 3. Fallback: Project root
        let root_candidate = PathBuf::from("tokenizer.json");
        if root_candidate.is_file() {
            return Ok(root_candidate);
        }

        Err(format!(
            "tokenizer.json not found for model '{}' in HF cache or base path {:?}",
            model_name, base_path
        ))
    }

    /// In-process encoding of textual prompt to tokens
    pub fn encode(&self, prompt: &str, add_special_tokens: bool) -> Result<Vec<u32>, String> {
        let special_tokens = [
            "<|system|>",
            "<|user|>",
            "<|assistant|>",
            "</s>",
            "<s>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|eot_id|>",
            "[INST]",
            "[/INST]",
            "<<SYS>>",
            "<</SYS>>",
            "<|begin_of_text|>",
            "<|im_start|>",
            "<|im_end|>",
        ];

        let mut final_ids = Vec::new();
        println!(
            "BramhaTokenizer::encode called with add_special_tokens={}",
            add_special_tokens
        );
        if add_special_tokens
            && !prompt.starts_with("<s>")
            && let Some(bos) = self.tokenizer.token_to_id("<s>")
        {
            final_ids.push(bos);
        }

        let mut current_pos = 0;
        let prompt_len = prompt.len();

        while current_pos < prompt_len {
            let mut earliest_match_pos = prompt_len;
            let mut matched_token_str = "";
            let mut matched_token_id = 0;

            for sp_str in &special_tokens {
                if let Some(pos) = prompt[current_pos..].find(sp_str)
                    && let Some(sp_id) = self.tokenizer.token_to_id(sp_str)
                {
                    let absolute_pos = current_pos + pos;
                    if absolute_pos < earliest_match_pos {
                        earliest_match_pos = absolute_pos;
                        matched_token_str = sp_str;
                        matched_token_id = sp_id;
                    }
                }
            }

            if earliest_match_pos < prompt_len {
                if earliest_match_pos > current_pos {
                    let chunk = &prompt[current_pos..earliest_match_pos];
                    if let Ok(encoding) = self.tokenizer.encode(chunk, false) {
                        final_ids.extend_from_slice(encoding.get_ids());
                    }
                }
                final_ids.push(matched_token_id);
                current_pos = earliest_match_pos + matched_token_str.len();
            } else {
                let chunk = &prompt[current_pos..];
                if let Ok(encoding) = self.tokenizer.encode(chunk, false) {
                    final_ids.extend_from_slice(encoding.get_ids());
                }
                break;
            }
        }

        Ok(final_ids)
    }

    /// In-process decoding of tokens back to human-readable string
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, String> {
        self.tokenizer
            .decode(ids, skip_special_tokens)
            .map_err(|e| format!("Tokenization decoding error: {}", e))
    }

    /// Access the underlying tokenizers::Tokenizer directly
    pub fn inner(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Read tokenizer_config.json and apply the model's native Jinja chat template.
    /// Returns the raw prompt if no template is found, enabling flawless cross-model compatibility.
    pub fn apply_chat_template(model_name: &str, base_path: &Path, prompt: &str) -> String {
        // Try to find tokenizer_config.json using the same resolution logic
        let config_path = Self::resolve_path(model_name, base_path)
            .unwrap_or_else(|_| PathBuf::from("."))
            .with_file_name("tokenizer_config.json");

        if config_path.is_file()
            && let Ok(raw) = std::fs::read(&config_path)
            && let Ok(config_json) = serde_json::from_slice::<serde_json::Value>(&raw)
            && let Some(chat_template) = config_json.get("chat_template").and_then(|v| v.as_str())
        {
            let mut env = minijinja::Environment::new();
            // Add some basic functions HF templates expect
            env.add_function(
                "raise_exception",
                |_msg: String| -> Result<String, minijinja::Error> {
                    Err(minijinja::Error::new(
                        minijinja::ErrorKind::InvalidOperation,
                        "HF Template exception",
                    ))
                },
            );

            if let Ok(template) = env.template_from_str(chat_template) {
                let context = minijinja::context! {
                    messages => vec![
                        minijinja::context! { role => "system", content => "You are a helpful AI assistant." },
                        minijinja::context! { role => "user", content => prompt }
                    ],
                    add_generation_prompt => true,
                    bos_token => config_json.get("bos_token").and_then(|v| v.as_str()).unwrap_or(""),
                    eos_token => config_json.get("eos_token").and_then(|v| v.as_str()).unwrap_or(""),
                };

                if let Ok(rendered) = template.render(context) {
                    return rendered;
                }
            }
        }

        // Fallback for missing template
        let model_name_lower = model_name.to_lowercase();
        if !prompt.contains("<|system|>")
            && !prompt.contains("<|user|>")
            && model_name_lower.contains("tinyllama")
        {
            format!(
                "<|system|>\nYou are a helpful AI assistant.</s>\n<|user|>\n{}</s>\n<|assistant|>\n",
                prompt
            )
        } else if !prompt.contains("<|im_start|>")
            && (model_name_lower.contains("llama") || model_name_lower.contains("qwen"))
        {
            format!(
                "<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                prompt
            )
        } else {
            prompt.to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_tokenizer_path_resolution_and_fallbacks() {
        let temp_dir = std::env::temp_dir();
        let model_dir = temp_dir.join("test_tokenizer_model");
        let _ = fs::remove_dir_all(&model_dir);
        fs::create_dir_all(&model_dir).unwrap();

        let res = BramhaTokenizer::resolve_path("nonexistent_model_dummy_xyz", &model_dir);
        let has_root = PathBuf::from("tokenizer.json").exists();
        if !has_root {
            assert!(res.is_err());
        }

        let local_tokenizer = model_dir.join("tokenizer.json");
        fs::write(&local_tokenizer, b"{}").unwrap();

        let path = BramhaTokenizer::resolve_path("test_model", &model_dir).unwrap();
        assert_eq!(path, local_tokenizer);

        let _ = fs::remove_dir_all(&model_dir);
    }
}
