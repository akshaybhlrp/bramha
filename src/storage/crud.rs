use super::metadata_sql::MetadataSqlStore;
use rusqlite::{params, Connection};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone)]
pub struct Collection {
    pub id: String,
    pub name: String,
    pub created_at: u64,
}

#[derive(Debug, Clone)]
pub struct Document {
    pub id: String,
    pub collection_id: String,
    pub name: String,
    pub content: String,
    pub created_at: u64,
}

#[derive(Debug, Clone)]
pub struct Chunk {
    pub id: String,
    pub document_id: String,
    pub chunk_index: i32,
    pub content: String,
    pub created_at: u64,
}

#[derive(Debug, Clone)]
pub struct Session {
    pub id: String,
    pub name: String,
    pub created_at: u64,
}

#[derive(Debug, Clone)]
pub struct Model {
    pub id: String,
    pub architecture: String,
    pub parameters: i64,
    pub path: String,
    pub created_at: u64,
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

impl MetadataSqlStore {
    // === Collection CRUD ===
    pub fn create_collection(&self, id: &str, name: &str) -> Result<Collection, String> {
        let conn = Connection::open(self.db_path()).map_err(|e| e.to_string())?;
        let created_at = now_ms();
        conn.execute(
            "INSERT INTO collections (id, name, created_at) VALUES (?1, ?2, ?3)",
            params![id, name, created_at],
        )
        .map_err(|e| format!("Error creating collection: {}", e))?;

        Ok(Collection {
            id: id.to_string(),
            name: name.to_string(),
            created_at,
        })
    }

    pub fn get_collection(&self, id: &str) -> Result<Option<Collection>, String> {
        let conn = Connection::open(self.db_path()).map_err(|e| e.to_string())?;
        let mut stmt = conn
            .prepare("SELECT id, name, created_at FROM collections WHERE id = ?1")
            .map_err(|e| e.to_string())?;
        let mut rows = stmt.query(params![id]).map_err(|e| e.to_string())?;

        if let Some(row) = rows.next().map_err(|e| e.to_string())? {
            Ok(Some(Collection {
                id: row.get(0).unwrap(),
                name: row.get(1).unwrap(),
                created_at: row.get(2).unwrap(),
            }))
        } else {
            Ok(None)
        }
    }

    pub fn update_collection(&self, id: &str, name: &str) -> Result<(), String> {
        let conn = Connection::open(self.db_path()).map_err(|e| e.to_string())?;
        conn.execute(
            "UPDATE collections SET name = ?1 WHERE id = ?2",
            params![name, id],
        )
        .map_err(|e| format!("Error updating collection: {}", e))?;
        Ok(())
    }

    pub fn delete_collection(&self, id: &str) -> Result<(), String> {
        let conn = Connection::open(self.db_path()).map_err(|e| e.to_string())?;
        // Enable foreign keys for this connection to cascade deletes
        conn.execute("PRAGMA foreign_keys=ON;", [])
            .map_err(|e| e.to_string())?;
        conn.execute("DELETE FROM collections WHERE id = ?1", params![id])
            .map_err(|e| format!("Error deleting collection: {}", e))?;
        Ok(())
    }

    // === Document CRUD ===
    pub fn create_document(
        &self,
        id: &str,
        collection_id: &str,
        name: &str,
        content: &str,
    ) -> Result<Document, String> {
        let conn = Connection::open(self.db_path()).map_err(|e| e.to_string())?;
        conn.execute("PRAGMA foreign_keys=ON;", [])
            .map_err(|e| e.to_string())?;
        let created_at = now_ms();
        conn.execute(
            "INSERT INTO documents (id, collection_id, name, content, created_at) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![id, collection_id, name, content, created_at],
        )
        .map_err(|e| format!("Error creating document: {}", e))?;

        Ok(Document {
            id: id.to_string(),
            collection_id: collection_id.to_string(),
            name: name.to_string(),
            content: content.to_string(),
            created_at,
        })
    }

    pub fn get_document(&self, id: &str) -> Result<Option<Document>, String> {
        let conn = Connection::open(self.db_path()).map_err(|e| e.to_string())?;
        let mut stmt = conn
            .prepare("SELECT id, collection_id, name, content, created_at FROM documents WHERE id = ?1")
            .map_err(|e| e.to_string())?;
        let mut rows = stmt.query(params![id]).map_err(|e| e.to_string())?;

        if let Some(row) = rows.next().map_err(|e| e.to_string())? {
            Ok(Some(Document {
                id: row.get(0).unwrap(),
                collection_id: row.get(1).unwrap(),
                name: row.get(2).unwrap(),
                content: row.get(3).unwrap(),
                created_at: row.get(4).unwrap(),
            }))
        } else {
            Ok(None)
        }
    }

    pub fn update_document(&self, id: &str, name: &str, content: &str) -> Result<(), String> {
        let conn = Connection::open(self.db_path()).map_err(|e| e.to_string())?;
        conn.execute(
            "UPDATE documents SET name = ?1, content = ?2 WHERE id = ?3",
            params![name, content, id],
        )
        .map_err(|e| format!("Error updating document: {}", e))?;
        Ok(())
    }

    pub fn delete_document(&self, id: &str) -> Result<(), String> {
        let conn = Connection::open(self.db_path()).map_err(|e| e.to_string())?;
        conn.execute("PRAGMA foreign_keys=ON;", [])
            .map_err(|e| e.to_string())?;
        conn.execute("DELETE FROM documents WHERE id = ?1", params![id])
            .map_err(|e| format!("Error deleting document: {}", e))?;
        Ok(())
    }

    // === Chunk CRUD ===
    pub fn create_chunk(
        &self,
        id: &str,
        document_id: &str,
        chunk_index: i32,
        content: &str,
    ) -> Result<Chunk, String> {
        let conn = Connection::open(self.db_path()).map_err(|e| e.to_string())?;
        conn.execute("PRAGMA foreign_keys=ON;", [])
            .map_err(|e| e.to_string())?;
        let created_at = now_ms();
        conn.execute(
            "INSERT INTO chunks (id, document_id, chunk_index, content, created_at) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![id, document_id, chunk_index, content, created_at],
        )
        .map_err(|e| format!("Error creating chunk: {}", e))?;

        Ok(Chunk {
            id: id.to_string(),
            document_id: document_id.to_string(),
            chunk_index,
            content: content.to_string(),
            created_at,
        })
    }

    pub fn get_chunk(&self, id: &str) -> Result<Option<Chunk>, String> {
        let conn = Connection::open(self.db_path()).map_err(|e| e.to_string())?;
        let mut stmt = conn
            .prepare("SELECT id, document_id, chunk_index, content, created_at FROM chunks WHERE id = ?1")
            .map_err(|e| e.to_string())?;
        let mut rows = stmt.query(params![id]).map_err(|e| e.to_string())?;

        if let Some(row) = rows.next().map_err(|e| e.to_string())? {
            Ok(Some(Chunk {
                id: row.get(0).unwrap(),
                document_id: row.get(1).unwrap(),
                chunk_index: row.get(2).unwrap(),
                content: row.get(3).unwrap(),
                created_at: row.get(4).unwrap(),
            }))
        } else {
            Ok(None)
        }
    }

    pub fn update_chunk(&self, id: &str, content: &str) -> Result<(), String> {
        let conn = Connection::open(self.db_path()).map_err(|e| e.to_string())?;
        conn.execute(
            "UPDATE chunks SET content = ?1 WHERE id = ?2",
            params![content, id],
        )
        .map_err(|e| format!("Error updating chunk: {}", e))?;
        Ok(())
    }

    pub fn delete_chunk(&self, id: &str) -> Result<(), String> {
        let conn = Connection::open(self.db_path()).map_err(|e| e.to_string())?;
        conn.execute("DELETE FROM chunks WHERE id = ?1", params![id])
            .map_err(|e| format!("Error deleting chunk: {}", e))?;
        Ok(())
    }

    // === Session CRUD ===
    pub fn create_session(&self, id: &str, name: &str) -> Result<Session, String> {
        let conn = Connection::open(self.db_path()).map_err(|e| e.to_string())?;
        let created_at = now_ms();
        conn.execute(
            "INSERT INTO sessions (id, name, created_at) VALUES (?1, ?2, ?3)",
            params![id, name, created_at],
        )
        .map_err(|e| format!("Error creating session: {}", e))?;

        Ok(Session {
            id: id.to_string(),
            name: name.to_string(),
            created_at,
        })
    }

    pub fn get_session(&self, id: &str) -> Result<Option<Session>, String> {
        let conn = Connection::open(self.db_path()).map_err(|e| e.to_string())?;
        let mut stmt = conn
            .prepare("SELECT id, name, created_at FROM sessions WHERE id = ?1")
            .map_err(|e| e.to_string())?;
        let mut rows = stmt.query(params![id]).map_err(|e| e.to_string())?;

        if let Some(row) = rows.next().map_err(|e| e.to_string())? {
            Ok(Some(Session {
                id: row.get(0).unwrap(),
                name: row.get(1).unwrap(),
                created_at: row.get(2).unwrap(),
            }))
        } else {
            Ok(None)
        }
    }

    pub fn update_session(&self, id: &str, name: &str) -> Result<(), String> {
        let conn = Connection::open(self.db_path()).map_err(|e| e.to_string())?;
        conn.execute(
            "UPDATE sessions SET name = ?1 WHERE id = ?2",
            params![name, id],
        )
        .map_err(|e| format!("Error updating session: {}", e))?;
        Ok(())
    }

    pub fn delete_session(&self, id: &str) -> Result<(), String> {
        let conn = Connection::open(self.db_path()).map_err(|e| e.to_string())?;
        conn.execute("DELETE FROM sessions WHERE id = ?1", params![id])
            .map_err(|e| format!("Error deleting session: {}", e))?;
        Ok(())
    }

    // === Model CRUD ===
    pub fn create_model(
        &self,
        id: &str,
        architecture: &str,
        parameters: i64,
        path: &str,
    ) -> Result<Model, String> {
        let conn = Connection::open(self.db_path()).map_err(|e| e.to_string())?;
        let created_at = now_ms();
        conn.execute(
            "INSERT INTO models (id, architecture, parameters, path, created_at) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![id, architecture, parameters, path, created_at],
        )
        .map_err(|e| format!("Error creating model: {}", e))?;

        Ok(Model {
            id: id.to_string(),
            architecture: architecture.to_string(),
            parameters,
            path: path.to_string(),
            created_at,
        })
    }

    pub fn get_model(&self, id: &str) -> Result<Option<Model>, String> {
        let conn = Connection::open(self.db_path()).map_err(|e| e.to_string())?;
        let mut stmt = conn
            .prepare("SELECT id, architecture, parameters, path, created_at FROM models WHERE id = ?1")
            .map_err(|e| e.to_string())?;
        let mut rows = stmt.query(params![id]).map_err(|e| e.to_string())?;

        if let Some(row) = rows.next().map_err(|e| e.to_string())? {
            Ok(Some(Model {
                id: row.get(0).unwrap(),
                architecture: row.get(1).unwrap(),
                parameters: row.get(2).unwrap(),
                path: row.get(3).unwrap(),
                created_at: row.get(4).unwrap(),
            }))
        } else {
            Ok(None)
        }
    }

    pub fn update_model(&self, id: &str, architecture: &str, parameters: i64, path: &str) -> Result<(), String> {
        let conn = Connection::open(self.db_path()).map_err(|e| e.to_string())?;
        conn.execute(
            "UPDATE models SET architecture = ?1, parameters = ?2, path = ?3 WHERE id = ?4",
            params![architecture, parameters, path, id],
        )
        .map_err(|e| format!("Error updating model: {}", e))?;
        Ok(())
    }

    pub fn delete_model(&self, id: &str) -> Result<(), String> {
        let conn = Connection::open(self.db_path()).map_err(|e| e.to_string())?;
        conn.execute("DELETE FROM models WHERE id = ?1", params![id])
            .map_err(|e| format!("Error deleting model: {}", e))?;
        Ok(())
    }
}
