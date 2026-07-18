use super::activation_view::ActivationMaterializedView;
use rusqlite::{Connection, params};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PlannerTrace {
    pub id: Option<i64>,
    pub prompt: String,
    pub decision: String,
    pub latency_ms: f64,
    pub spec_accept_rate: f32, // actual speculative acceptance rate if speculative path was run
    pub timestamp_ms: u64,
}

/// Singleton for the default storage/query_analytics.db path. Avoids re-opening
/// the file + re-running CREATE TABLE IF NOT EXISTS on every MetadataSqlStore::new().
static DEFAULT_STORE: OnceLock<MetadataSqlStore> = OnceLock::new();

/// Prune threshold: keep at most this many rows in append-only tables.
const PLANNER_TRACE_MAX_ROWS: i64 = 10_000;
const SHADOW_SCAN_MAX_ROWS: i64 = 1_000;

pub struct MetadataSqlStore {
    db_path: PathBuf,
    /// Single connection per store instance, protected by Mutex. Eliminates the
    /// per-method Connection::open() syscall + WAL lock acquisition overhead that
    /// previously happened on every call.
    conn: Mutex<Connection>,
}

impl MetadataSqlStore {
    pub fn db_path(&self) -> &Path {
        &self.db_path
    }

    /// Return a reference to the process-global default store (storage/query_analytics.db).
    /// First call initializes the connection + tables; subsequent calls are free.
    pub fn global() -> &'static MetadataSqlStore {
        DEFAULT_STORE.get_or_init(|| {
            let storage_dir = Path::new("storage");
            if !storage_dir.exists() {
                let _ = std::fs::create_dir_all(storage_dir);
            }
            let db_path = storage_dir.join("query_analytics.db");
            Self::open(db_path).expect("Failed to open default MetadataSqlStore")
        })
    }

    /// Initialize a new SQLite connection to the shared query analytics database.
    /// Prefer `MetadataSqlStore::global()` for the default path to avoid redundant opens.
    pub fn new() -> Self {
        let storage_dir = Path::new("storage");
        if !storage_dir.exists() {
            let _ = std::fs::create_dir_all(storage_dir);
        }
        let db_path = storage_dir.join("query_analytics.db");
        Self::open(db_path).expect("Failed to open MetadataSqlStore")
    }

    pub fn new_with_path(custom_path: &str) -> Self {
        let db_path = PathBuf::from(custom_path);
        Self::open(db_path).expect("Failed to open MetadataSqlStore")
    }

    /// Open a connection and initialize all tables.
    fn open(db_path: PathBuf) -> Result<Self, String> {
        let conn = Connection::open(&db_path).map_err(|e| e.to_string())?;
        let store = MetadataSqlStore {
            db_path,
            conn: Mutex::new(conn),
        };
        store.initialize_db()?;
        Ok(store)
    }

    fn initialize_db(&self) -> Result<(), String> {
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        // Enable high-concurrency WAL mode and SQLite index creations
        conn.execute_batch(
            "PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA foreign_keys=ON;",
        )
        .map_err(|e| format!("SQLite pragma err: {}", e))?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS planner_traces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt TEXT NOT NULL,
                decision TEXT NOT NULL,
                latency_ms REAL NOT NULL,
                spec_accept_rate REAL NOT NULL,
                timestamp_ms INTEGER NOT NULL
            )",
            [],
        )
        .map_err(|e| format!("SQLite table creation err: {}", e))?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS activation_views (
                workflow_id TEXT NOT NULL,
                branch_id TEXT NOT NULL,
                token_hash TEXT NOT NULL,
                token_length INTEGER NOT NULL,
                disk_path TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                PRIMARY KEY(workflow_id, branch_id)
            )",
            [],
        )
        .map_err(|e| format!("SQLite activation_views table err: {}", e))?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS route_quality_stats (
                decision TEXT PRIMARY KEY,
                avg_latency_ms REAL NOT NULL,
                confidence_score REAL NOT NULL,
                success_count INTEGER NOT NULL,
                last_updated INTEGER NOT NULL
            )",
            [],
        )
        .map_err(|e| format!("SQLite route_quality_stats table err: {}", e))?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS collections (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                created_at INTEGER NOT NULL
            )",
            [],
        )
        .map_err(|e| format!("SQLite collections table err: {}", e))?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                collection_id TEXT NOT NULL,
                name TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                FOREIGN KEY(collection_id) REFERENCES collections(id) ON DELETE CASCADE
            )",
            [],
        )
        .map_err(|e| format!("SQLite documents table err: {}", e))?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
            )",
            [],
        )
        .map_err(|e| format!("SQLite chunks table err: {}", e))?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                created_at INTEGER NOT NULL
            )",
            [],
        )
        .map_err(|e| format!("SQLite sessions table err: {}", e))?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS models (
                id TEXT PRIMARY KEY,
                architecture TEXT NOT NULL,
                parameters INTEGER NOT NULL,
                path TEXT NOT NULL,
                created_at INTEGER NOT NULL
            )",
            [],
        )
        .map_err(|e| format!("SQLite models table err: {}", e))?;

        // Create shadow_scan_stats table so the shadow scan path in cpu_engine.rs
        // doesn't have to CREATE TABLE IF NOT EXISTS on every event.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS shadow_scan_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt TEXT NOT NULL,
                cosine_similarity REAL NOT NULL,
                timestamp_ms INTEGER NOT NULL
            )",
            [],
        )
        .map_err(|e| format!("SQLite shadow_scan_stats table err: {}", e))?;

        Ok(())
    }

    /// Persist a query plan trace into SQLite
    pub fn log_planner_trace(&self, trace: PlannerTrace) -> Result<(), String> {
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        conn.execute(
            "INSERT INTO planner_traces (prompt, decision, latency_ms, spec_accept_rate, timestamp_ms)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                trace.prompt,
                trace.decision,
                trace.latency_ms,
                trace.spec_accept_rate,
                now
            ],
        ).map_err(|e| format!("SQLite insert trace err: {}", e))?;

        // Periodic pruning: keep only the most recent N rows
        conn.execute(
            "DELETE FROM planner_traces WHERE id <= (SELECT MAX(id) - ?1 FROM planner_traces)",
            params![PLANNER_TRACE_MAX_ROWS],
        ).ok(); // best-effort

        Ok(())
    }

    /// Retrieve the most recent N planner traces for telemetry and dashboards
    pub fn get_recent_planner_traces(&self, limit: usize) -> Result<Vec<PlannerTrace>, String> {
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        let mut stmt = conn
            .prepare(
                "SELECT id, prompt, decision, latency_ms, spec_accept_rate, timestamp_ms
             FROM planner_traces ORDER BY id DESC LIMIT ?1",
            )
            .map_err(|e| e.to_string())?;

        let rows = stmt
            .query_map(params![limit], |row| {
                Ok(PlannerTrace {
                    id: Some(row.get(0)?),
                    prompt: row.get(1)?,
                    decision: row.get(2)?,
                    latency_ms: row.get(3)?,
                    spec_accept_rate: row.get(4)?,
                    timestamp_ms: row.get(5)?,
                })
            })
            .map_err(|e| e.to_string())?;

        let mut traces = Vec::new();
        for trace in rows.flatten() {
            traces.push(trace);
        }
        Ok(traces)
    }

    /// Computes the running average speculative accept rate from the last N speculative traces.
    /// Defaults to 0.7 (optimal baseline) if no history is present.
    pub fn get_historical_accept_rate(&self, limit: usize) -> Result<f32, String> {
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        let mut stmt = conn
            .prepare(
                "SELECT AVG(spec_accept_rate) 
             FROM (
                 SELECT spec_accept_rate FROM planner_traces 
                 WHERE decision = 'SpeculativeDecode' 
                 ORDER BY id DESC LIMIT ?1
             )",
            )
            .map_err(|e| e.to_string())?;

        let mut rows = stmt.query(params![limit]).map_err(|e| e.to_string())?;
        if let Some(row) = rows.next().map_err(|e| e.to_string())? {
            let avg: Option<f32> = row.get(0).map_err(|e| e.to_string())?;
            // Return 0.7 (healthy speculative baseline) if no history exists yet
            Ok(avg.unwrap_or(0.7))
        } else {
            Ok(0.7)
        }
    }

    /// Sprint 11: Adaptive Learning - Update Route Quality (Reinforcement)
    pub fn update_route_quality(
        &self,
        decision: &str,
        latency_ms: f64,
        success: bool,
    ) -> Result<(), String> {
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        // Exponential Weighted Average (EWA) for latency, and +/- adjustment for confidence
        let alpha = 0.1;
        let mut current_latency = latency_ms;
        let mut current_confidence = if success { 1.0f32 } else { 0.0f32 };
        let mut current_success_count = if success { 1 } else { 0 };

        let mut stmt = conn.prepare("SELECT avg_latency_ms, confidence_score, success_count FROM route_quality_stats WHERE decision = ?1").unwrap();
        let mut rows = stmt.query(params![decision]).unwrap();
        if let Some(row) = rows.next().unwrap() {
            let prev_lat: f64 = row.get(0).unwrap();
            let prev_conf: f32 = row.get(1).unwrap();
            let prev_succ: i64 = row.get(2).unwrap();

            current_latency = (alpha * latency_ms) + ((1.0 - alpha) * prev_lat);

            // Adjust confidence: +0.05 on success, -0.15 on failure, clamped [0.0, 1.0]
            let conf_delta = if success { 0.05 } else { -0.15 };
            current_confidence = (prev_conf + conf_delta).clamp(0.0, 1.0);

            current_success_count = prev_succ + if success { 1 } else { 0 };
        }

        conn.execute(
            "INSERT INTO route_quality_stats (decision, avg_latency_ms, confidence_score, success_count, last_updated)
             VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(decision) DO UPDATE SET
             avg_latency_ms=excluded.avg_latency_ms,
             confidence_score=excluded.confidence_score,
             success_count=excluded.success_count,
             last_updated=excluded.last_updated",
            params![decision, current_latency, current_confidence, current_success_count, now],
        ).map_err(|e| format!("SQLite route_quality err: {}", e))?;

        Ok(())
    }

    /// Retrieve the learned confidence score for a specific planner route [0.0, 1.0]
    pub fn get_route_confidence(&self, decision: &str) -> Result<f32, String> {
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        let mut stmt = conn
            .prepare("SELECT confidence_score FROM route_quality_stats WHERE decision = ?1")
            .map_err(|e| e.to_string())?;
        let mut rows = stmt.query(params![decision]).map_err(|e| e.to_string())?;
        if let Some(row) = rows.next().map_err(|e| e.to_string())? {
            Ok(row.get(0).unwrap_or(0.5))
        } else {
            Ok(0.5) // Default neutral confidence
        }
    }

    /// Persist an Activation Materialized View to track KV checkpoints
    pub fn log_activation_view(&self, view: ActivationMaterializedView) -> Result<(), String> {
        let conn = self.conn.lock().map_err(|e| e.to_string())?;

        // Use UPSERT to replace old branches
        conn.execute(
            "INSERT INTO activation_views (workflow_id, branch_id, token_hash, token_length, disk_path, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)
             ON CONFLICT(workflow_id, branch_id) DO UPDATE SET
             token_hash=excluded.token_hash,
             token_length=excluded.token_length,
             disk_path=excluded.disk_path,
             created_at=excluded.created_at",
            params![
                view.workflow_id,
                view.branch_id,
                view.token_hash,
                view.token_length,
                view.disk_path,
                view.created_at
            ],
        ).map_err(|e| format!("SQLite insert view err: {}", e))?;
        Ok(())
    }

    /// Fetch a valid activation view for a workflow
    pub fn get_activation_view(
        &self,
        workflow_id: &str,
        branch_id: &str,
    ) -> Result<Option<ActivationMaterializedView>, String> {
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        let mut stmt = conn
            .prepare(
                "SELECT workflow_id, branch_id, token_hash, token_length, disk_path, created_at
             FROM activation_views WHERE workflow_id = ?1 AND branch_id = ?2",
            )
            .map_err(|e| e.to_string())?;

        let mut rows = stmt
            .query(params![workflow_id, branch_id])
            .map_err(|e| e.to_string())?;
        if let Some(row) = rows.next().map_err(|e| e.to_string())? {
            Ok(Some(ActivationMaterializedView {
                workflow_id: row.get(0).map_err(|e| e.to_string())?,
                branch_id: row.get(1).map_err(|e| e.to_string())?,
                token_hash: row.get(2).map_err(|e| e.to_string())?,
                token_length: row.get(3).map_err(|e| e.to_string())?,
                disk_path: row.get(4).map_err(|e| e.to_string())?,
                created_at: row.get(5).map_err(|e| e.to_string())?,
            }))
        } else {
            Ok(None)
        }
    }

    /// Retrieve the average latency for a specific planner route in milliseconds
    pub fn get_route_average_latency(&self, decision: &str) -> Result<Option<f64>, String> {
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        let mut stmt = conn
            .prepare("SELECT avg_latency_ms FROM route_quality_stats WHERE decision = ?1")
            .map_err(|e| e.to_string())?;
        let mut rows = stmt.query(params![decision]).map_err(|e| e.to_string())?;
        if let Some(row) = rows.next().map_err(|e| e.to_string())? {
            let avg: Option<f64> = row.get(0).ok();
            Ok(avg)
        } else {
            Ok(None)
        }
    }

    /// Log a shadow scan result and prune old rows. Called from cpu_engine.rs shadow path.
    pub fn log_shadow_scan(
        &self,
        prompt: &str,
        cosine_similarity: f64,
    ) -> Result<(), String> {
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        conn.execute(
            "INSERT INTO shadow_scan_stats (prompt, cosine_similarity, timestamp_ms) VALUES (?1, ?2, ?3)",
            params![prompt, cosine_similarity, now],
        ).map_err(|e| format!("SQLite shadow_scan insert err: {}", e))?;

        // Prune: keep only the most recent N rows
        conn.execute(
            "DELETE FROM shadow_scan_stats WHERE id <= (SELECT MAX(id) - ?1 FROM shadow_scan_stats)",
            params![SHADOW_SCAN_MAX_ROWS],
        ).ok(); // best-effort

        Ok(())
    }

    /// Check if the shadow scan gate has tripped (>5% of recent queries below 0.999 cosine sim).
    /// Returns Some(ratio) if tripped, None if not enough data or healthy.
    pub fn check_shadow_gate(&self) -> Result<Option<f64>, String> {
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        let mut stmt = conn
            .prepare("SELECT cosine_similarity FROM shadow_scan_stats ORDER BY id DESC LIMIT 100")
            .map_err(|e| e.to_string())?;
        let mut rows = stmt.query([]).map_err(|e| e.to_string())?;

        let mut total = 0;
        let mut low_similarity_count = 0;
        while let Ok(Some(row)) = rows.next() {
            if let Ok(sim) = row.get::<_, f64>(0) {
                total += 1;
                if sim < 0.999 {
                    low_similarity_count += 1;
                }
            }
        }
        if total >= 20 {
            let ratio = (low_similarity_count as f64) / (total as f64);
            if ratio > 0.05 {
                return Ok(Some(ratio));
            }
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_planner_sql_persistence_and_stats() {
        let db_file = "storage/test_planner_metadata.db";
        let _ = fs::remove_file(db_file);

        let store = MetadataSqlStore::new_with_path(db_file);

        // Assert initial default accept rate is 0.7
        let initial_rate = store.get_historical_accept_rate(10).unwrap();
        assert_eq!(initial_rate, 0.7f32);

        // Log one exact trace and one speculative trace
        store
            .log_planner_trace(PlannerTrace {
                id: None,
                prompt: "exact path".to_string(),
                decision: "ExactDecode".to_string(),
                latency_ms: 150.0,
                spec_accept_rate: 0.0,
                timestamp_ms: 0,
            })
            .unwrap();

        store
            .log_planner_trace(PlannerTrace {
                id: None,
                prompt: "spec path".to_string(),
                decision: "SpeculativeDecode".to_string(),
                latency_ms: 60.0,
                spec_accept_rate: 0.85,
                timestamp_ms: 0,
            })
            .unwrap();

        // Check retrieval
        let traces = store.get_recent_planner_traces(5).unwrap();
        assert_eq!(traces.len(), 2);
        assert_eq!(traces[0].prompt, "spec path");
        assert_eq!(traces[0].decision, "SpeculativeDecode");

        // Check accept rate calculation from history
        let rate = store.get_historical_accept_rate(10).unwrap();
        assert!((rate - 0.85).abs() < 1e-4);

        let _ = fs::remove_file(db_file);
    }

    #[test]
    fn test_activation_view_persistence() {
        let db_file = "storage/test_activation_view.db";
        let _ = fs::remove_file(db_file);

        let store = MetadataSqlStore::new_with_path(db_file);

        let view = ActivationMaterializedView {
            workflow_id: "wf-123".to_string(),
            branch_id: "branch-abc".to_string(),
            token_hash: "hash-456".to_string(),
            token_length: 50,
            disk_path: "/tmp/bramha/kv_wf-123_branch-abc.bin".to_string(),
            created_at: 1000000,
        };

        store.log_activation_view(view.clone()).unwrap();

        let fetched = store.get_activation_view("wf-123", "branch-abc").unwrap();
        assert!(fetched.is_some());
        let fetched_view = fetched.unwrap();
        assert_eq!(fetched_view.token_hash, "hash-456");
        assert_eq!(fetched_view.token_length, 50);
        assert_eq!(
            fetched_view.disk_path,
            "/tmp/bramha/kv_wf-123_branch-abc.bin"
        );

        // Test upsert (replace on conflict)
        let mut view_updated = view.clone();
        view_updated.token_length = 75;
        store.log_activation_view(view_updated).unwrap();

        let fetched_updated = store
            .get_activation_view("wf-123", "branch-abc")
            .unwrap()
            .unwrap();
        assert_eq!(fetched_updated.token_length, 75);

        let _ = fs::remove_file(db_file);
    }
}
