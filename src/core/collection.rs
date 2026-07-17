use crate::core::filter::Filter;
use crate::core::vector::{Metric, Vector};
use crate::index::{HnswIndex, IvfFlatIndex};
use rayon::prelude::*;
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SearchResult {
    pub id: String,
    pub score: f32,
    pub metadata: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector_score: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bm25_score: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub alpha: Option<f32>,
}

/// Helper function to estimate query lexical vs semantic complexity
pub fn estimate_hybrid_alpha(query: &str) -> f64 {
    let lower = query.to_lowercase();

    // Exact matching symbols or error codes
    let is_highly_lexical = query.contains('{')
        || query.contains('}')
        || query.contains("::")
        || query.contains('"')
        || query.contains('\'')
        || query.contains('(')
        || query.contains(')')
        || query.contains("ERROR_")
        || query.contains("ERR_")
        || query.contains("0x");

    let is_semantic = lower.starts_with("how")
        || lower.starts_with("what")
        || lower.starts_with("why")
        || lower.starts_with("explain")
        || lower.starts_with("describe");

    if is_highly_lexical {
        0.3 // Trust BM25 heavily
    } else if is_semantic {
        0.8 // Trust Vector heavily
    } else {
        0.5 // Balanced fusion
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum CollectionStatus {
    READY,
    BUILDING,
    CORRUPT,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
pub enum TuningProfile {
    #[default]
    Balanced,
    LowLatency,
    HighRecall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Collection {
    pub name: String,
    pub dimension: usize,
    pub metric: Metric,
    pub vectors: HashMap<String, Vector>,
    pub index: Option<IvfFlatIndex>,
    #[serde(default = "default_status")]
    pub status: CollectionStatus,
    #[serde(default)]
    pub bm25_index: Option<crate::index::bm25::BM25Index>,
    #[serde(skip, default = "default_sqlite")]
    pub sqlite: Option<Arc<Mutex<Connection>>>,
    #[serde(default)]
    pub hnsw_index: Option<HnswIndex>,
    #[serde(default)]
    pub tuning_profile: TuningProfile,
}

fn default_status() -> CollectionStatus {
    CollectionStatus::READY
}

fn default_sqlite() -> Option<Arc<Mutex<Connection>>> {
    None
}

impl Collection {
    pub fn new(name: String, dimension: usize, metric: Metric) -> Self {
        let mut c = Collection {
            name,
            dimension,
            metric,
            vectors: HashMap::new(),
            index: None,
            status: CollectionStatus::READY,
            bm25_index: Some(crate::index::bm25::BM25Index::new()),
            sqlite: None,
            hnsw_index: None,
            tuning_profile: TuningProfile::Balanced,
        };
        let _ = c.init_sqlite_index();
        c
    }

    /// Rebuilds the in-memory SQLite metadata index from the current vector set.
    pub fn init_sqlite_index(&mut self) -> Result<(), String> {
        let conn = Connection::open_in_memory().map_err(|e| format!("SQLite open err: {}", e))?;
        let _ = conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;");
        conn.execute(
            "CREATE TABLE IF NOT EXISTS metadata_index (id TEXT PRIMARY KEY, metadata JSON)",
            [],
        )
        .map_err(|e| format!("SQLite create err: {}", e))?;

        for (id, vector) in &self.vectors {
            let meta_str = vector
                .metadata
                .as_ref()
                .map(|m| m.to_string())
                .unwrap_or_else(|| "{}".to_string());
            conn.execute(
                "INSERT INTO metadata_index (id, metadata) VALUES (?1, ?2)",
                rusqlite::params![id, meta_str],
            )
            .map_err(|e| format!("SQLite insert err: {}", e))?;
        }

        self.sqlite = Some(Arc::new(Mutex::new(conn)));
        Ok(())
    }

    /// Inserts a vector into the collection, validating its dimension.
    pub fn insert(&mut self, vector: Vector) -> Result<(), String> {
        if vector.values.len() != self.dimension {
            return Err(format!(
                "Vector dimension mismatch. Expected {}, got {}",
                self.dimension,
                vector.values.len()
            ));
        }

        // S3.3: Automatically index document text into lexical BM25 index on insert
        if let Some(ref meta) = vector.metadata
            && let Some(text_val) = meta.get("text").or_else(|| meta.get("content"))
            && let Some(text_str) = text_val.as_str()
        {
            if let Some(ref mut bm25) = self.bm25_index {
                bm25.add_document(vector.id.clone(), text_str);
            } else {
                let mut bm25 = crate::index::bm25::BM25Index::new();
                bm25.add_document(vector.id.clone(), text_str);
                self.bm25_index = Some(bm25);
            }
        }

        self.vectors.insert(vector.id.clone(), vector.clone());

        if let Some(ref db) = self.sqlite
            && let Ok(conn) = db.lock()
        {
            let meta_str = vector
                .metadata
                .as_ref()
                .map(|m| m.to_string())
                .unwrap_or_else(|| "{}".to_string());
            let _ = conn.execute(
                "INSERT OR REPLACE INTO metadata_index (id, metadata) VALUES (?1, ?2)",
                rusqlite::params![vector.id, meta_str],
            );
        }

        Ok(())
    }

    /// Deletes a vector by ID. Returns true if the vector existed.
    pub fn delete(&mut self, id: &str) -> bool {
        let removed = self.vectors.remove(id).is_some();
        if removed
            && let Some(ref db) = self.sqlite
            && let Ok(conn) = db.lock()
        {
            let _ = conn.execute(
                "DELETE FROM metadata_index WHERE id = ?1",
                rusqlite::params![id],
            );
        }
        removed
    }

    /// Searches for top-k similar vectors using exact search or IVF-Flat approximate search, applying optional filters.
    pub fn search(
        &self,
        query_vec: &[f32],
        k: usize,
        filter: Option<&Filter>,
        use_index: bool,
    ) -> Vec<SearchResult> {
        if self.status == CollectionStatus::CORRUPT {
            println!(
                "⚠️ Cannot search in degraded/corrupt collection '{}'",
                self.name
            );
            return vec![];
        }
        if query_vec.len() != self.dimension {
            return vec![];
        }

        // S3.7: Use SQL pre-filter before ANN search
        let mut allowed_ids: Option<HashSet<String>> = None;
        if let Some(f) = filter
            && let Some(ref db) = self.sqlite
            && let Ok(conn) = db.lock()
        {
            let (sql_where, params_json) = f.to_sql_query();
            let sql = format!("SELECT id FROM metadata_index WHERE {}", sql_where);

            let params_str: Vec<String> = params_json
                .iter()
                .map(|v| match v {
                    serde_json::Value::String(s) => s.clone(),
                    _ => v.to_string(),
                })
                .collect();

            let params_dyn: Vec<&dyn rusqlite::ToSql> = params_str
                .iter()
                .map(|s| s as &dyn rusqlite::ToSql)
                .collect();

            if let Ok(mut stmt) = conn.prepare(&sql)
                && let Ok(rows) = stmt.query_map(&params_dyn[..], |row| row.get::<_, String>(0))
            {
                let mut ids = HashSet::new();
                for r in rows.flatten() {
                    ids.insert(r);
                }
                allowed_ids = Some(ids);
            }
        }

        // Try approximate search using the index if requested and available
        if use_index {
            if let Some(ref hnsw) = self.hnsw_index {
                let hnsw_results = hnsw.search(self, query_vec, k, allowed_ids.as_ref());
                if !hnsw_results.is_empty() {
                    return hnsw_results;
                }
            }
            if let Some(ref idx) = self.index {
                let ann_results = idx.search(self, query_vec, k, filter, allowed_ids.as_ref());
                if !ann_results.is_empty() {
                    return ann_results;
                }
            }
        }

        // Fallback: Exact Flat Scan (Multi-threaded using Rayon)
        let mut results: Vec<SearchResult> = self
            .vectors
            .par_iter()
            .map(|(_, v)| v)
            .filter(|v| {
                if let Some(ref allowed) = allowed_ids {
                    allowed.contains(&v.id)
                } else if let Some(f) = filter {
                    f.matches(&v.metadata)
                } else {
                    true
                }
            })
            .map(|v| {
                let score = self.metric.distance(query_vec, &v.values);
                SearchResult {
                    id: v.id.clone(),
                    score,
                    metadata: v.metadata.clone(),
                    ..Default::default()
                }
            })
            .collect();

        // Sort: L2 is ascending (smaller distance is better), Cosine/DotProduct is descending (larger similarity is better)
        if self.metric.is_ascending() {
            results.sort_by(|a, b| {
                a.score
                    .partial_cmp(&b.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        } else {
            results.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        results.truncate(k);
        results
    }

    /// Performs a high-fidelity hybrid search (Vector ANN + BM25 keyword lexical match)
    /// merged using Reciprocal Rank Fusion (RRF).
    pub fn hybrid_search(
        &self,
        query_text: &str,
        query_vec: &[f32],
        k: usize,
        filter: Option<&Filter>,
        use_index: bool,
    ) -> Vec<SearchResult> {
        let alpha = estimate_hybrid_alpha(query_text);

        // 1. Execute vector similarity search
        let vec_results = self.search(query_vec, k * 2, filter, use_index);

        // 2. Execute BM25 keyword lexical search if BM25 index is built
        let bm25_results = if let Some(ref bm25) = self.bm25_index {
            bm25.search(query_text, k * 2)
        } else {
            Vec::new()
        };

        if bm25_results.is_empty() {
            let mut final_results = vec_results;
            for res in &mut final_results {
                res.vector_score = Some(res.score);
                res.alpha = Some(alpha as f32);
            }
            final_results.truncate(k);
            return final_results;
        }

        // 3. Perform Reciprocal Rank Fusion (RRF)
        // RRF score smoothing constant
        let rrf_k = 60.0;
        struct HybridScores {
            vec_rrf: f64,
            bm25_rrf: f64,
        }
        let mut rrf_scores: HashMap<String, HybridScores> = HashMap::new();

        // Accumulate vector ranks
        for (rank, res) in vec_results.iter().enumerate() {
            let score = 1.0 / (rrf_k + rank as f64);
            rrf_scores
                .entry(res.id.clone())
                .or_insert(HybridScores {
                    vec_rrf: 0.0,
                    bm25_rrf: 0.0,
                })
                .vec_rrf += score;
        }

        // Accumulate BM25 ranks
        for (rank, (doc_id, _)) in bm25_results.iter().enumerate() {
            let score = 1.0 / (rrf_k + rank as f64);
            rrf_scores
                .entry(doc_id.clone())
                .or_insert(HybridScores {
                    vec_rrf: 0.0,
                    bm25_rrf: 0.0,
                })
                .bm25_rrf += score;
        }

        // 4. Map back to SearchResult format
        let mut hybrid_results: Vec<SearchResult> = rrf_scores
            .into_iter()
            .map(|(id, scores)| {
                // Find original metadata if available in self.vectors
                let metadata = self.vectors.get(&id).and_then(|v| v.metadata.clone());

                let final_score = (alpha * scores.vec_rrf) + ((1.0 - alpha) * scores.bm25_rrf);

                SearchResult {
                    id,
                    score: final_score as f32,
                    metadata,
                    vector_score: Some(scores.vec_rrf as f32),
                    bm25_score: Some(scores.bm25_rrf as f32),
                    alpha: Some(alpha as f32),
                }
            })
            .collect();

        // Sort descending by RRF score (larger score is better)
        hybrid_results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hybrid_results.truncate(k);
        hybrid_results
    }

    /// Rebuilds the IVF-Flat index for this collection.
    pub fn rebuild_index(&mut self, num_clusters: usize, n_probe: usize) {
        self.status = CollectionStatus::BUILDING;
        let idx = IvfFlatIndex::build(self, num_clusters, n_probe);
        if idx.centroids.is_empty() && !self.vectors.is_empty() {
            self.status = CollectionStatus::CORRUPT;
        } else {
            self.index = Some(idx);
            self.status = CollectionStatus::READY;
        }
    }

    /// Rebuilds the HNSW proximity graph index for this collection.
    pub fn rebuild_hnsw_index(&mut self, m: usize, ef_construction: usize, ef_search: usize) {
        self.status = CollectionStatus::BUILDING;
        let hnsw = HnswIndex::build(self, m, ef_construction, ef_search);
        self.hnsw_index = Some(hnsw);
        self.status = CollectionStatus::READY;
    }
}
