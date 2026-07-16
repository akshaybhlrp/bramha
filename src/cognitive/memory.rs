use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryTier {
    Working,
    Episodic,
    Semantic,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MemoryEntry {
    pub id: String,
    pub content: String,
    pub tier: MemoryTier,
    pub confidence: f64,
    pub usage_count: usize,
    pub last_accessed_ms: u64,
    pub created_at_ms: u64,
    pub provenance: String,
    #[serde(default)]
    pub retracted: bool,
    #[serde(default)]
    pub retraction_reason: Option<String>,
}

pub struct MemoryManager {
    file_path: PathBuf,
}

impl MemoryManager {
    pub fn new() -> Self {
        let storage_dir = Path::new("storage");
        if !storage_dir.exists() {
            let _ = std::fs::create_dir_all(storage_dir);
        }
        MemoryManager {
            file_path: storage_dir.join("cognitive_memory.json"),
        }
    }

    /// Load memory entries from persistent storage
    pub fn load_memories(&self) -> HashMap<String, MemoryEntry> {
        if !self.file_path.exists() {
            return HashMap::new();
        }
        let data = std::fs::read_to_string(&self.file_path).unwrap_or_else(|_| "{}".to_string());
        serde_json::from_str(&data).unwrap_or_else(|_| HashMap::new())
    }

    /// Save memory entries atomically
    pub fn save_memories(&self, memories: &HashMap<String, MemoryEntry>) -> Result<(), String> {
        let serialized = serde_json::to_string_pretty(memories).map_err(|e| e.to_string())?;
        let temp_path = self.file_path.with_extension("tmp");
        {
            let mut file = File::create(&temp_path).map_err(|e| e.to_string())?;
            file.write_all(serialized.as_bytes())
                .map_err(|e| e.to_string())?;
            file.sync_all().map_err(|e| e.to_string())?;
        }
        std::fs::rename(temp_path, &self.file_path).map_err(|e| e.to_string())?;
        Ok(())
    }

    /// Insert or update a memory entry
    pub fn insert_memory(&self, entry: MemoryEntry) -> Result<(), String> {
        let mut memories = self.load_memories();
        memories.insert(entry.id.clone(), entry);
        self.save_memories(&memories)
    }

    /// Retrieve a memory entry and boost its usage
    pub fn retrieve_memory(&self, id: &str, now_ms: u64) -> Option<MemoryEntry> {
        let mut memories = self.load_memories();
        if let Some(entry) = memories.get_mut(id) {
            entry.usage_count += 1;
            entry.last_accessed_ms = now_ms;
            let result = entry.clone();
            let _ = self.save_memories(&memories);
            Some(result)
        } else {
            None
        }
    }

    /// Explicitly reinforce memory confidence
    pub fn reinforce_memory(&self, id: &str, amount: f64) -> Result<(), String> {
        let mut memories = self.load_memories();
        if let Some(entry) = memories.get_mut(id) {
            entry.confidence = (entry.confidence + amount).min(1.0);
            self.save_memories(&memories)?;
        }
        Ok(())
    }

    /// Explicitly decay memory confidence
    pub fn decay_memory(&self, id: &str, amount: f64) -> Result<(), String> {
        let mut memories = self.load_memories();
        if let Some(entry) = memories.get_mut(id) {
            entry.confidence = (entry.confidence - amount).max(0.0);
            self.save_memories(&memories)?;
        }
        Ok(())
    }

    /// Explicitly retract a memory entry and set its confidence to 0.0
    pub fn retract_memory(&self, id: &str, reason: &str) -> Result<(), String> {
        let mut memories = self.load_memories();
        if let Some(entry) = memories.get_mut(id) {
            entry.retracted = true;
            entry.retraction_reason = Some(reason.to_string());
            entry.confidence = 0.0;
            self.save_memories(&memories)?;
            println!("🛑 Memory '{}' retracted: {}", id, reason);
        }
        Ok(())
    }

    /// Search memory tiers automatically, score candidates by relevance/recency/confidence,
    /// and silently inject the top ones into the prompt. Logs injection decisions.
    pub fn proactive_inject(&self, prompt: &str, now_ms: u64) -> (String, Vec<String>) {
        let memories = self.load_memories();
        let mut candidates = Vec::new();
        let stopwords = [
            "the", "and", "a", "of", "to", "in", "is", "that", "it", "for", "on", "with", "as",
        ];
        let prompt_words: Vec<String> = prompt
            .to_lowercase()
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphabetic()).to_string())
            .filter(|w| !w.is_empty() && !stopwords.contains(&w.as_str()))
            .collect();

        for entry in memories.values() {
            if entry.retracted {
                continue;
            }
            // 1. Recency Decay calculation (without writing back to storage during inline search)
            let elapsed_sec = ((now_ms.saturating_sub(entry.last_accessed_ms)) as f64) / 1000.0;
            let decay_rate = match entry.tier {
                MemoryTier::Working => 0.05,
                MemoryTier::Episodic => 0.005,
                MemoryTier::Semantic => 0.0005,
            };
            let decayed_confidence =
                (entry.confidence * (-decay_rate * elapsed_sec).exp()).max(0.0);

            // 2. Keyword relevance
            let content_lower = entry.content.to_lowercase();
            let mut matches = 0;
            for word in &prompt_words {
                if content_lower.contains(word) {
                    matches += 1;
                }
            }

            let relevance = if !prompt_words.is_empty() {
                matches as f64 / prompt_words.len() as f64
            } else {
                0.0
            };

            // 3. Combined Score
            let score = relevance * decayed_confidence;
            if score >= 0.15 {
                candidates.push((entry.clone(), score));
            }
        }

        // Sort candidates by score descending
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut injected_prompts = Vec::new();
        let mut decision_logs = Vec::new();

        // Pick top 2 memories
        for (entry, score) in candidates.iter().take(2) {
            injected_prompts.push(format!(
                "[Silently Injected Memory Context: {} (provenance: {}, confidence: {:.2}, relevance score: {:.2})]",
                entry.content, entry.provenance, entry.confidence, score
            ));
            decision_logs.push(format!(
                "Injected memory '{}' with combined score {:.2}",
                entry.id, score
            ));
        }

        if injected_prompts.is_empty() {
            (
                prompt.to_string(),
                vec!["No relevant memories found to inject".to_string()],
            )
        } else {
            let merged_prompt = format!("{}\n{}", injected_prompts.join("\n"), prompt);
            (merged_prompt, decision_logs)
        }
    }

    /// Apply forgetting curves with different decay rates per tier
    /// score = confidence * exp(-decay_rate * (now - last_accessed))
    pub fn apply_forgetting_decay(&self, now_ms: u64) -> Result<(), String> {
        let mut memories = self.load_memories();
        for entry in memories.values_mut() {
            let elapsed_sec = ((now_ms.saturating_sub(entry.last_accessed_ms)) as f64) / 1000.0;

            // Tier-specific decay rates (Episodic decays faster than Semantic)
            let decay_rate = match entry.tier {
                MemoryTier::Working => 0.05,    // Fast decay
                MemoryTier::Episodic => 0.005,  // Medium decay
                MemoryTier::Semantic => 0.0005, // Slow/Stable
            };

            entry.confidence = (entry.confidence * (-decay_rate * elapsed_sec).exp()).max(0.0);
        }
        self.save_memories(&memories)
    }

    /// Prune memories with low confidence ( expired or forgotten )
    pub fn prune_memories(&self, min_confidence: f64) -> Result<usize, String> {
        let mut memories = self.load_memories();
        let initial_len = memories.len();
        memories.retain(|_, entry| entry.confidence >= min_confidence);
        let pruned_count = initial_len - memories.len();
        self.save_memories(&memories)?;
        Ok(pruned_count)
    }

    /// Distill Episodic memory from conversational history
    pub fn distill_episodic_memory(
        &self,
        session_id: &str,
        turns: &[crate::storage::session_store::ChatTurn],
        now_ms: u64,
    ) -> Result<MemoryEntry, String> {
        if turns.is_empty() {
            return Err("No turns available to distill memory".to_string());
        }

        // Aggregate prompt-response logs into a concise factual summary
        let summary = format!(
            "Session {}: User queried: '{}'. Assistant replied: '{}'.",
            session_id,
            turns[0].content,
            turns.last().map(|t| t.content.as_str()).unwrap_or("")
        );

        let entry = MemoryEntry {
            id: format!("mem_ep_{}", session_id),
            content: summary,
            tier: MemoryTier::Episodic,
            confidence: 0.8, // Initial confidence
            usage_count: 1,
            last_accessed_ms: now_ms,
            created_at_ms: now_ms,
            provenance: format!("session:{}", session_id),
            retracted: false,
            retraction_reason: None,
        };

        self.insert_memory(entry.clone())?;
        Ok(entry)
    }

    /// Spawns a background task to periodically consolidate high-usage Episodic memories into Semantic ones.
    pub fn spawn_consolidation_worker(manager: std::sync::Arc<Self>) {
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(3600)).await;
                println!("🧠 [Consolidation Worker] Running periodic memory promotion...");

                let mut memories = manager.load_memories();
                let mut promoted = 0;

                for entry in memories.values_mut() {
                    // Promote Episodic memories with high usage and high confidence
                    if entry.tier == MemoryTier::Episodic
                        && entry.usage_count >= 5
                        && entry.confidence > 0.85
                    {
                        entry.tier = MemoryTier::Semantic;
                        entry.confidence = 1.0; // Maximize confidence on promotion
                        promoted += 1;
                    }
                }

                if promoted > 0 {
                    if let Err(e) = manager.save_memories(&memories) {
                        eprintln!(
                            "⚠️ [Consolidation Worker] Failed to save promoted memories: {}",
                            e
                        );
                    } else {
                        println!(
                            "✅ [Consolidation Worker] Successfully promoted {} memories to Semantic Tier.",
                            promoted
                        );
                    }
                }
            }
        });
    }

    /// Detect contradictions against highly confident semantic memories
    pub fn detect_contradiction(&self, new_fact: &str) -> Option<MemoryEntry> {
        let memories = self.load_memories();
        let new_fact_clean = new_fact.to_lowercase();

        let stopwords = [
            "the", "and", "a", "of", "to", "in", "is", "that", "it", "for", "on", "with", "as",
        ];
        let new_fact_words: std::collections::HashSet<String> = new_fact_clean
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphabetic()).to_string())
            .filter(|w| !w.is_empty() && !stopwords.contains(&w.as_str()))
            .collect();

        if new_fact_words.is_empty() {
            return None;
        }

        for entry in memories.values() {
            if entry.retracted {
                continue;
            }
            if entry.tier == MemoryTier::Semantic && entry.confidence >= 0.8 {
                let entry_clean = entry.content.to_lowercase();
                let entry_words: std::collections::HashSet<String> = entry_clean
                    .split_whitespace()
                    .map(|w| w.trim_matches(|c: char| !c.is_alphabetic()).to_string())
                    .filter(|w| !w.is_empty() && !stopwords.contains(&w.as_str()))
                    .collect();

                // Calculate word overlap ratio
                let intersection: std::collections::HashSet<_> =
                    new_fact_words.intersection(&entry_words).cloned().collect();
                let overlap =
                    intersection.len() as f32 / new_fact_words.len().min(entry_words.len()) as f32;

                // If there's high overlap, check if there's a negation contradiction
                if overlap >= 0.4 {
                    // Check if one contains negation words and the other does not
                    let negations = [
                        "not",
                        "no",
                        "never",
                        "cannot",
                        "isn't",
                        "aren't",
                        "won't",
                        "don't",
                        "doesn't",
                        "false",
                        "incorrect",
                    ];
                    let new_has_neg = negations.iter().any(|&neg| new_fact_clean.contains(neg));
                    let entry_has_neg = negations.iter().any(|&neg| entry_clean.contains(neg));

                    if new_has_neg != entry_has_neg {
                        return Some(entry.clone());
                    }

                    // Direct antonym pairs check (e.g. true vs false, enable vs disable, hot vs cold)
                    let antonyms = [
                        ("true", "false"),
                        ("enable", "disable"),
                        ("enabled", "disabled"),
                        ("hot", "cold"),
                        ("warm", "cold"),
                        ("high", "low"),
                        ("active", "inactive"),
                        ("allow", "deny"),
                        ("allowed", "denied"),
                        ("success", "failure"),
                        ("successful", "failed"),
                    ];

                    for (a, b) in antonyms {
                        let has_a = new_fact_clean.contains(a) || entry_clean.contains(a);
                        let has_b = new_fact_clean.contains(b) || entry_clean.contains(b);

                        if has_a && has_b {
                            let new_has_a = new_fact_clean.contains(a);
                            let entry_has_a = entry_clean.contains(a);
                            if new_has_a != entry_has_a {
                                return Some(entry.clone());
                            }
                        }
                    }
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_contradiction() {
        let manager = MemoryManager {
            file_path: PathBuf::from("storage/test_contradiction_memory.json"),
        };
        let _ = std::fs::remove_file(&manager.file_path);

        let now = 1000000;
        let entry = MemoryEntry {
            id: "mem_fact".to_string(),
            content: "The sky is blue".to_string(),
            tier: MemoryTier::Semantic,
            confidence: 0.9,
            usage_count: 5,
            last_accessed_ms: now,
            created_at_ms: now,
            provenance: "test".to_string(),
            retracted: false,
            retraction_reason: None,
        };
        manager.insert_memory(entry).unwrap();

        // Detect contradiction via negation mismatch
        let detected = manager.detect_contradiction("The sky is not blue");
        assert!(detected.is_some());
        assert_eq!(detected.unwrap().id, "mem_fact");

        let detected_none = manager.detect_contradiction("The sky is blue today.");
        assert!(detected_none.is_none());

        let _ = std::fs::remove_file(&manager.file_path);
    }

    #[test]
    fn test_memory_tier_lifecycle_decay_reinforcement() {
        let manager = MemoryManager {
            file_path: PathBuf::from("storage/test_cognitive_memory.json"),
        };
        let _ = std::fs::remove_file(&manager.file_path);

        let now = 1000000;
        let entry = MemoryEntry {
            id: "mem_1".to_string(),
            content: "Bramha is a Rust RAG engine".to_string(),
            tier: MemoryTier::Semantic,
            confidence: 0.8,
            usage_count: 1,
            last_accessed_ms: now,
            created_at_ms: now,
            provenance: "test".to_string(),
            retracted: false,
            retraction_reason: None,
        };

        // 1. Insert
        manager.insert_memory(entry).unwrap();
        let loaded = manager.load_memories();
        assert!(loaded.contains_key("mem_1"));

        // 2. Retrieve & Reinforce manually
        let retrieved = manager.retrieve_memory("mem_1", now + 1000).unwrap();
        assert_eq!(retrieved.usage_count, 2);

        manager.reinforce_memory("mem_1", 0.15).unwrap();
        let reinforced = manager.load_memories().get("mem_1").unwrap().clone();
        assert!((reinforced.confidence - 0.95).abs() < 1e-9);

        // Decay manually
        manager.decay_memory("mem_1", 0.05).unwrap();
        let decayed_manual = manager.load_memories().get("mem_1").unwrap().clone();
        assert!((decayed_manual.confidence - 0.90).abs() < 1e-9);

        // 3. Forgetting Decay (Semantic tier decays very slowly)
        manager.apply_forgetting_decay(now + 200000).unwrap();
        let decayed = manager.load_memories().get("mem_1").unwrap().confidence;
        assert!(decayed > 0.8); // Semantic stays very high/stable!

        // 4. Pruning
        let pruned = manager.prune_memories(0.99).unwrap();
        assert_eq!(pruned, 1); // should prune since confidence is less than 0.99
        assert!(manager.load_memories().is_empty());

        let _ = std::fs::remove_file(&manager.file_path);
    }

    #[test]
    fn test_proactive_memory_injection() {
        let manager = MemoryManager {
            file_path: PathBuf::from("storage/test_proactive_inject.json"),
        };
        let _ = std::fs::remove_file(&manager.file_path);

        let now = 1000000;
        let entry = MemoryEntry {
            id: "mem_prompt_inject".to_string(),
            content: "The default sharding directory is storage/shards".to_string(),
            tier: MemoryTier::Semantic,
            confidence: 0.9,
            usage_count: 1,
            last_accessed_ms: now,
            created_at_ms: now,
            provenance: "docs".to_string(),
            retracted: false,
            retraction_reason: None,
        };

        manager.insert_memory(entry).unwrap();

        let prompt = "Where is the sharding directory stored?";
        let (merged, logs) = manager.proactive_inject(prompt, now);

        assert!(merged.contains("[Silently Injected Memory Context:"));
        assert!(merged.contains("The default sharding directory is storage/shards"));
        assert!(logs[0].contains("Injected memory"));

        let _ = std::fs::remove_file(&manager.file_path);
    }

    #[test]
    fn test_memory_retraction() {
        let manager = MemoryManager {
            file_path: PathBuf::from("storage/test_retraction_memory.json"),
        };
        let _ = std::fs::remove_file(&manager.file_path);

        let now = 1000000;
        let entry = MemoryEntry {
            id: "mem_retract_test".to_string(),
            content: "This is a fact to retract".to_string(),
            tier: MemoryTier::Semantic,
            confidence: 0.9,
            usage_count: 1,
            last_accessed_ms: now,
            created_at_ms: now,
            provenance: "user".to_string(),
            retracted: false,
            retraction_reason: None,
        };
        manager.insert_memory(entry).unwrap();

        // Perform retraction
        manager
            .retract_memory("mem_retract_test", "Fact proven false")
            .unwrap();

        let memories = manager.load_memories();
        let retracted_entry = memories.get("mem_retract_test").unwrap();
        assert!(retracted_entry.retracted);
        assert_eq!(retracted_entry.confidence, 0.0);
        assert_eq!(
            retracted_entry.retraction_reason.as_deref(),
            Some("Fact proven false")
        );

        // Verify proactive inject skips it
        let prompt = "Fact to retract";
        let (merged, _logs) = manager.proactive_inject(prompt, now);
        assert!(!merged.contains("This is a fact to retract"));

        let _ = std::fs::remove_file(&manager.file_path);
    }
}
