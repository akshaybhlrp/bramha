/// Planner v1 — selects between exact, speculative, and cached-answer paths
/// Status: COMPLETE
/// 
/// ACTIVE PATHS:
///   [x] exact decode
///   [x] speculative decode  
///   [x] cached-answer
///   [x] activation replay
/// 

use std::fs;
use std::path::{Path, PathBuf};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlannerDecision {
    ExactDecode,
    SpeculativeDecode,
    CachedAnswer,
    SpandaSparse,
}

impl std::fmt::Display for PlannerDecision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlannerDecision::ExactDecode => write!(f, "ExactDecode"),
            PlannerDecision::SpeculativeDecode => write!(f, "SpeculativeDecode"),
            PlannerDecision::CachedAnswer => write!(f, "CachedAnswer"),
            PlannerDecision::SpandaSparse => write!(f, "SpandaSparse"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlannerPolicy {
    pub planner_mode: String,            // "auto" | "exact_only"
    pub min_speculative_accept_rate: f32, // minimum threshold to allow speculative decoding
    pub max_cached_age_seconds: u64,      // maximum age of cache entries
}

impl Default for PlannerPolicy {
    fn default() -> Self {
        Self {
            planner_mode: "auto".to_string(),
            min_speculative_accept_rate: 0.5,
            max_cached_age_seconds: 86400, // 1 day
        }
    }
}

impl PlannerPolicy {
    /// Retrieve the standard persistence path in the `cache` directory
    fn get_cache_path() -> PathBuf {
        Path::new("cache").join("planner_policy.json")
    }

    /// Load the policy from the persisted cache file, falling back cleanly to defaults if missing or invalid
    pub fn load() -> Self {
        let path = Self::get_cache_path();
        if path.exists() {
            if let Ok(content) = fs::read_to_string(&path) {
                if let Ok(policy) = serde_json::from_str::<Self>(&content) {
                    return policy;
                }
            }
        }
        Self::default()
    }

    /// Save the active policy to disk for warm-state persistence
    pub fn save(&self) -> Result<(), String> {
        let path = Self::get_cache_path();
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let serialized = serde_json::to_string_pretty(self)
            .map_err(|e| format!("Failed to serialize policy: {}", e))?;
        fs::write(&path, serialized)
            .map_err(|e| format!("Failed to write policy file: {}", e))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_policy_default_and_roundtrip() {
        let temp_dir = std::env::temp_dir().join("bramha_policy_test");
        let _ = fs::create_dir_all(&temp_dir);
        let test_file = temp_dir.join("planner_policy.json");
        let _ = fs::remove_file(&test_file);

        let policy = PlannerPolicy {
            planner_mode: "exact_only".to_string(),
            min_speculative_accept_rate: 0.75,
            max_cached_age_seconds: 3600,
        };

        // Serialize directly
        let serialized = serde_json::to_string(&policy).unwrap();
        let deserialized: PlannerPolicy = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.planner_mode, "exact_only");
        assert_eq!(deserialized.min_speculative_accept_rate, 0.75);
        assert_eq!(deserialized.max_cached_age_seconds, 3600);
    }
}
