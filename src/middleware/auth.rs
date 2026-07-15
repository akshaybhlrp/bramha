use axum::{
    async_trait,
    extract::FromRequestParts,
    http::{StatusCode, request::Parts},
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Role {
    ReadOnly = 1,
    Write = 2,
    Admin = 3,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ApiKeyInfo {
    pub key: String,
    pub role: Role,
}

pub struct AuthManager {
    keys_file: &'static str,
}

impl AuthManager {
    pub fn new() -> Self {
        AuthManager {
            keys_file: "storage/keys.json",
        }
    }

    /// Load all keys from storage, writing default keys if file doesn't exist
    pub fn load_keys(&self) -> HashMap<String, Role> {
        let path = Path::new(self.keys_file);
        if !path.exists() {
            let mut default_keys = HashMap::new();
            default_keys.insert("admin_key".to_string(), Role::Admin);
            default_keys.insert("write_key".to_string(), Role::Write);
            default_keys.insert("read_key".to_string(), Role::ReadOnly);
            let _ = self.save_keys(&default_keys);
            return default_keys;
        }

        let data = std::fs::read_to_string(path).unwrap_or_else(|_| "{}".to_string());
        serde_json::from_str(&data).unwrap_or_else(|_| HashMap::new())
    }

    /// Save keys dynamically to allow runtime key rotation
    pub fn save_keys(&self, keys: &HashMap<String, Role>) -> Result<(), String> {
        let serialized = serde_json::to_string_pretty(keys).map_err(|e| e.to_string())?;
        let temp_path = Path::new(self.keys_file).with_extension("tmp");
        {
            let mut file = File::create(&temp_path).map_err(|e| e.to_string())?;
            file.write_all(serialized.as_bytes())
                .map_err(|e| e.to_string())?;
            file.sync_all().map_err(|e| e.to_string())?;
        }
        std::fs::rename(temp_path, self.keys_file).map_err(|e| e.to_string())?;
        Ok(())
    }

    /// Verify if a key matches or exceeds a required role
    pub fn authorize(&self, key: &str, required: Role) -> Result<ApiKeyInfo, String> {
        let keys = self.load_keys();
        if let Some(&role) = keys.get(key) {
            if role >= required {
                Ok(ApiKeyInfo {
                    key: key.to_string(),
                    role,
                })
            } else {
                Err(format!(
                    "Forbidden: Insufficient privileges. Required role: {:?}",
                    required
                ))
            }
        } else {
            Err("Unauthorized: Invalid API Key".to_string())
        }
    }
}

/// Extractor to require ReadOnly level access
pub struct RequireReadOnly(pub ApiKeyInfo);

#[async_trait]
impl<S> FromRequestParts<S> for RequireReadOnly
where
    S: Send + Sync,
{
    type Rejection = (StatusCode, String);

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        let key = extract_token_from_header(parts)?;
        let manager = AuthManager::new();
        let info = manager.authorize(&key, Role::ReadOnly).map_err(|e| {
            if e.starts_with("Forbidden") {
                (StatusCode::FORBIDDEN, e)
            } else {
                (StatusCode::UNAUTHORIZED, e)
            }
        })?;
        Ok(RequireReadOnly(info))
    }
}

/// Extractor to require Write level access
pub struct RequireWrite(pub ApiKeyInfo);

#[async_trait]
impl<S> FromRequestParts<S> for RequireWrite
where
    S: Send + Sync,
{
    type Rejection = (StatusCode, String);

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        let key = extract_token_from_header(parts)?;
        let manager = AuthManager::new();
        let info = manager.authorize(&key, Role::Write).map_err(|e| {
            if e.starts_with("Forbidden") {
                (StatusCode::FORBIDDEN, e)
            } else {
                (StatusCode::UNAUTHORIZED, e)
            }
        })?;
        Ok(RequireWrite(info))
    }
}

/// Extractor to require Admin level access
pub struct RequireAdmin(pub ApiKeyInfo);

#[async_trait]
impl<S> FromRequestParts<S> for RequireAdmin
where
    S: Send + Sync,
{
    type Rejection = (StatusCode, String);

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        let key = extract_token_from_header(parts)?;
        let manager = AuthManager::new();
        let info = manager.authorize(&key, Role::Admin).map_err(|e| {
            if e.starts_with("Forbidden") {
                (StatusCode::FORBIDDEN, e)
            } else {
                (StatusCode::UNAUTHORIZED, e)
            }
        })?;
        Ok(RequireAdmin(info))
    }
}

fn extract_token_from_header(parts: &Parts) -> Result<String, (StatusCode, String)> {
    if let Some(auth_header) = parts.headers.get("Authorization") {
        if let Ok(auth_str) = auth_header.to_str() {
            if auth_str.starts_with("Bearer ") {
                return Ok(auth_str["Bearer ".len()..].to_string());
            }
        }
    }
    Err((
        StatusCode::UNAUTHORIZED,
        "Unauthorized: Missing Authorization Bearer token header".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_role_hierarchy_comparisons() {
        assert!(Role::Admin > Role::Write);
        assert!(Role::Write > Role::ReadOnly);
        assert!(Role::Admin >= Role::Admin);
    }

    #[test]
    fn test_auth_manager_authorization_rules() {
        let _ = std::fs::create_dir_all("storage");
        let manager = AuthManager {
            keys_file: "storage/test_keys.json",
        };
        let _ = std::fs::remove_file(manager.keys_file);

        // 1. Load keys (which writes default keys)
        let keys = manager.load_keys();
        assert!(keys.contains_key("admin_key"));
        assert!(keys.contains_key("write_key"));
        assert!(keys.contains_key("read_key"));

        // 2. Authorize admin for Admin role
        let admin_info = manager.authorize("admin_key", Role::Admin).unwrap();
        assert_eq!(admin_info.role, Role::Admin);

        // 3. Authorize admin for ReadOnly role (hierarchy check)
        let read_info = manager.authorize("admin_key", Role::ReadOnly).unwrap();
        assert_eq!(read_info.role, Role::Admin);

        // 4. Authorize read_key for Write role (should fail!)
        let write_fail = manager.authorize("read_key", Role::Write);
        assert!(write_fail.is_err());
        assert!(write_fail.unwrap_err().contains("Forbidden"));

        // 5. Authorize invalid key (should fail!)
        let invalid_fail = manager.authorize("bad_key", Role::ReadOnly);
        assert!(invalid_fail.is_err());
        assert!(invalid_fail.unwrap_err().contains("Unauthorized"));

        // Cleanup
        let _ = std::fs::remove_file(manager.keys_file);
    }
}
