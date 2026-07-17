use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ChatTurn {
    pub role: String,
    pub content: String,
    pub timestamp_ms: u64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Session {
    pub id: String,
    pub title: String,
    pub kv_cache_id: Option<String>,
    pub turns: Vec<ChatTurn>,
    pub updated_at_ms: u64,
}

pub struct SessionStore {
    file_path: PathBuf,
}

impl Default for SessionStore {
    fn default() -> Self {
        Self::new()
    }
}

impl SessionStore {
    pub fn new() -> Self {
        let storage_dir = Path::new("storage");
        if !storage_dir.exists() {
            let _ = std::fs::create_dir_all(storage_dir);
        }
        SessionStore {
            file_path: storage_dir.join("sessions.json"),
        }
    }

    /// Loads all sessions from storage
    pub fn load_all(&self) -> HashMap<String, Session> {
        if !self.file_path.exists() {
            return HashMap::new();
        }
        let data = std::fs::read_to_string(&self.file_path).unwrap_or_else(|_| "{}".to_string());
        serde_json::from_str(&data).unwrap_or_else(|_| HashMap::new())
    }

    /// Saves all sessions to storage using an atomic crash-safe write
    pub fn save_all(&self, sessions: &HashMap<String, Session>) -> Result<(), String> {
        let serialized = serde_json::to_string_pretty(sessions).map_err(|e| e.to_string())?;

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

    /// Insert or update a session
    pub fn upsert(&self, session: Session) -> Result<(), String> {
        let mut sessions = self.load_all();
        sessions.insert(session.id.clone(), session);
        self.save_all(&sessions)
    }

    /// Retrieve a single session
    pub fn get(&self, id: &str) -> Option<Session> {
        let sessions = self.load_all();
        sessions.get(id).cloned()
    }

    /// Delete a session
    pub fn delete(&self, id: &str) -> Result<bool, String> {
        let mut sessions = self.load_all();
        let removed = sessions.remove(id).is_some();
        if removed {
            self.save_all(&sessions)?;
        }
        Ok(removed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_store_lifecycle_and_export() {
        let _ = std::fs::create_dir_all("storage");
        let store = SessionStore {
            file_path: Path::new("storage").join("test_sessions.json"),
        };
        let _ = std::fs::remove_file(&store.file_path);

        // 1. Create session turn Turn history
        let turn = ChatTurn {
            role: "user".to_string(),
            content: "Hello Bramha!".to_string(),
            timestamp_ms: 1000,
        };
        let session = Session {
            id: "sess_123".to_string(),
            title: "Bramha Conversational session".to_string(),
            kv_cache_id: Some("kv_123".to_string()),
            turns: vec![turn],
            updated_at_ms: 1005,
        };

        // 2. Save & Load
        store.upsert(session.clone()).unwrap();
        let retrieved = store.get("sess_123").unwrap();
        assert_eq!(retrieved.title, "Bramha Conversational session");
        assert_eq!(retrieved.turns[0].content, "Hello Bramha!");

        // 3. Export JSON Turn representation
        let exported_json = serde_json::to_value(&retrieved).unwrap();
        assert_eq!(exported_json["id"], "sess_123");

        // Cleanup
        let deleted = store.delete("sess_123").unwrap();
        assert!(deleted);
        let _ = std::fs::remove_file(&store.file_path);
    }
}
