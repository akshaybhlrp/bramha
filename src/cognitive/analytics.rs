use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct QueryTrace {
    pub id: Option<i64>,
    pub query_string: String,
    pub retrieval_ms: f64,
    pub rerank_ms: f64,
    pub inference_ms: f64,
    pub cache_hit: bool,
    pub exit_layer: usize,
    pub timestamp_ms: u64,
}

pub struct AnalyticsStore {
    db_path: PathBuf,
}

impl Default for AnalyticsStore {
    fn default() -> Self {
        Self::new()
    }
}

impl AnalyticsStore {
    pub fn new() -> Self {
        let storage_dir = Path::new("storage");
        if !storage_dir.exists() {
            let _ = std::fs::create_dir_all(storage_dir);
        }
        let db_path = storage_dir.join("query_analytics.db");
        let store = AnalyticsStore { db_path };
        store.initialize_db().unwrap();
        store
    }

    pub fn new_with_path(custom_path: &str) -> Self {
        let db_path = PathBuf::from(custom_path);
        if let Some(parent) = db_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let store = AnalyticsStore { db_path };
        store.initialize_db().unwrap();
        store
    }

    fn initialize_db(&self) -> Result<(), String> {
        let conn = Connection::open(&self.db_path).map_err(|e| e.to_string())?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;")
            .map_err(|e| e.to_string())?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS query_traces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_string TEXT NOT NULL,
                retrieval_ms REAL NOT NULL,
                rerank_ms REAL NOT NULL,
                inference_ms REAL NOT NULL,
                cache_hit INTEGER NOT NULL,
                exit_layer INTEGER NOT NULL,
                timestamp_ms INTEGER NOT NULL
            )",
            [],
        )
        .map_err(|e| e.to_string())?;
        Ok(())
    }

    /// Persist a single query trace to SQLite
    pub fn log_trace(&self, trace: QueryTrace) -> Result<(), String> {
        let conn = Connection::open(&self.db_path).map_err(|e| e.to_string())?;
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        conn.execute(
            "INSERT INTO query_traces (query_string, retrieval_ms, rerank_ms, inference_ms, cache_hit, exit_layer, timestamp_ms)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                trace.query_string,
                trace.retrieval_ms,
                trace.rerank_ms,
                trace.inference_ms,
                if trace.cache_hit { 1 } else { 0 },
                trace.exit_layer,
                now
            ],
        ).map_err(|e| e.to_string())?;
        Ok(())
    }

    /// Retrieve the most recent N query traces
    pub fn get_recent_traces(&self, limit: usize) -> Result<Vec<QueryTrace>, String> {
        let conn = Connection::open(&self.db_path).map_err(|e| e.to_string())?;
        let mut stmt = conn.prepare(
            "SELECT id, query_string, retrieval_ms, rerank_ms, inference_ms, cache_hit, exit_layer, timestamp_ms
             FROM query_traces ORDER BY id DESC LIMIT ?1"
        ).map_err(|e| e.to_string())?;

        let rows = stmt
            .query_map(params![limit], |row| {
                Ok(QueryTrace {
                    id: Some(row.get(0)?),
                    query_string: row.get(1)?,
                    retrieval_ms: row.get(2)?,
                    rerank_ms: row.get(3)?,
                    inference_ms: row.get(4)?,
                    cache_hit: row.get::<_, i32>(5)? != 0,
                    exit_layer: row.get(6)?,
                    timestamp_ms: row.get(7)?,
                })
            })
            .map_err(|e| e.to_string())?;

        let mut traces = Vec::new();
        for trace in rows.flatten() {
            traces.push(trace);
        }
        Ok(traces)
    }

    /// Compute running statistics for query analytics feedback loop
    pub fn get_average_latency_ms(&self) -> Result<f64, String> {
        let conn = Connection::open(&self.db_path).map_err(|e| e.to_string())?;
        let mut stmt = conn
            .prepare("SELECT AVG(retrieval_ms + rerank_ms + inference_ms) FROM query_traces")
            .map_err(|e| e.to_string())?;
        let mut rows = stmt.query([]).map_err(|e| e.to_string())?;
        if let Some(row) = rows.next().map_err(|e| e.to_string())? {
            let avg: Option<f64> = row.get(0).map_err(|e| e.to_string())?;
            Ok(avg.unwrap_or(0.0))
        } else {
            Ok(0.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analytics_persistence_and_stats() {
        let db_file = "storage/test_query_analytics.db";
        let _ = std::fs::remove_file(db_file);

        let store = AnalyticsStore::new_with_path(db_file);
        let trace = QueryTrace {
            id: None,
            query_string: "test prompt".to_string(),
            retrieval_ms: 12.5,
            rerank_ms: 5.0,
            inference_ms: 120.0,
            cache_hit: true,
            exit_layer: 4,
            timestamp_ms: 0,
        };

        store.log_trace(trace).unwrap();
        let traces = store.get_recent_traces(10).unwrap();
        assert_eq!(traces.len(), 1);
        assert_eq!(traces[0].query_string, "test prompt");
        assert!(traces[0].cache_hit);

        let avg = store.get_average_latency_ms().unwrap();
        assert_eq!(avg, 137.5); // 12.5 + 5.0 + 120.0

        let _ = std::fs::remove_file(db_file);
    }
}
