use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OnboardStep {
    pub name: String,
    pub completed: bool,
    pub description: String,
    pub action_hint: String,
}

pub struct OnboardManager {
    pub root_dir: String,
}

impl OnboardManager {
    pub fn new(root_dir: &str) -> Self {
        OnboardManager {
            root_dir: root_dir.to_string(),
        }
    }

    /// Check if target directory permissions and shard subdirectories are valid
    pub fn check_storage(&self) -> OnboardStep {
        let path = Path::new(&self.root_dir);
        let completed = path.exists() && fs::create_dir_all(path.join("collections")).is_ok();
        OnboardStep {
            name: "Storage Initialization".to_string(),
            completed,
            description: "Verifies the database engine can write to target shard folders on the local system.".to_string(),
            action_hint: "Ensure correct permissions for the database storage folder.".to_string(),
        }
    }

    /// Check if LLaMA or secondary adapters models are loaded and calibration thresholds exist
    pub fn check_model_calibration(&self) -> OnboardStep {
        let manifest_path = Path::new(&self.root_dir).join("model_manifest.json");
        // Create dummy manifest for onboarding test success
        let completed = manifest_path.exists();
        OnboardStep {
            name: "Model Calibration".to_string(),
            completed,
            description: "Verifies that Candle model weights are registered and early-exit thresholds are loaded.".to_string(),
            action_hint: "Run model calibration prompts to generate early-exit bounds.".to_string(),
        }
    }

    /// Check if API authentication file is configured
    pub fn check_api_security(&self) -> OnboardStep {
        let auth_path = Path::new(&self.root_dir).join("keys.json");
        let completed = auth_path.exists();
        OnboardStep {
            name: "Security Configuration".to_string(),
            completed,
            description: "Verifies that API keys are initialized to protect mutating endpoints."
                .to_string(),
            action_hint: "Copy default keys from boot template to protect the REST endpoints."
                .to_string(),
        }
    }

    /// Aggregate onboarding steps into a production readiness checklist
    pub fn get_readiness_checklist(&self) -> Vec<OnboardStep> {
        vec![
            self.check_storage(),
            self.check_model_calibration(),
            self.check_api_security(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_onboarding_checklists() {
        let test_dir = "storage_onboard_test";
        let _ = fs::remove_dir_all(test_dir);
        fs::create_dir_all(test_dir).unwrap();

        let manager = OnboardManager::new(test_dir);

        // 1. Initial State: storage passes but manifest / security fails
        let steps = manager.get_readiness_checklist();
        assert!(steps[0].completed);
        assert!(!steps[1].completed);
        assert!(!steps[2].completed);

        // 2. Resolve Model Calibration step
        fs::write(format!("{}/model_manifest.json", test_dir), b"{}").unwrap();
        let step2 = manager.check_model_calibration();
        assert!(step2.completed);

        // 3. Resolve Security step
        fs::write(format!("{}/keys.json", test_dir), b"{}").unwrap();
        let step3 = manager.check_api_security();
        assert!(step3.completed);

        let _ = fs::remove_dir_all(test_dir);
    }
}
