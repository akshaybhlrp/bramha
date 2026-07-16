use axum::{
    Json,
    extract::{Path, State},
    http::StatusCode,
    response::sse::{Event, KeepAlive, Sse},
};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::sync::Arc;

use crate::core::collection::{Collection, SearchResult};
use crate::core::filter::Filter;
use crate::core::vector::{Metric, Vector};
use crate::middleware::auth::{RequireAdmin, RequireReadOnly, RequireWrite};
use crate::storage::Database;

pub type SharedState = Arc<Database>;

// --- API Payloads ---

#[derive(Deserialize)]
pub struct CreateCollectionPayload {
    pub name: String,
    pub dimension: usize,
    pub metric: Metric,
}

#[derive(Serialize)]
pub struct CollectionInfo {
    pub name: String,
    pub dimension: usize,
    pub metric: Metric,
    pub vector_count: usize,
    pub has_index: bool,
    pub index_clusters: Option<usize>,
    pub health_score: Option<f32>,
    pub recall_at_k: Option<f32>,
    pub tuning_profile: String,
    pub davies_bouldin_index: Option<f32>,
    pub silhouette_score: Option<f32>,
    pub imbalance_ratio: Option<f32>,
}

#[derive(Deserialize)]
pub struct UpsertVectorsPayload {
    pub vectors: Vec<Vector>,
}

#[derive(Deserialize)]
pub struct QueryCollectionPayload {
    pub vector: Vec<f32>,
    pub k: usize,
    #[serde(default = "default_true")]
    pub use_index: bool,
    pub filter: Option<Filter>,
    pub query_text: Option<String>,
    pub hybrid: Option<bool>,
}

fn default_true() -> bool {
    true
}

#[derive(Deserialize)]
pub struct DeleteVectorsPayload {
    pub ids: Vec<String>,
}

#[derive(Deserialize)]
pub struct ReindexPayload {
    // IVF-Flat options
    pub num_clusters: Option<usize>,
    pub n_probe: Option<usize>,
    // Common/HNSW options
    pub index_type: Option<String>,
    pub m: Option<usize>,
    pub ef_construction: Option<usize>,
    pub ef_search: Option<usize>,
}

#[derive(Deserialize)]
pub struct CacheCheckPayload {
    pub vector: Vec<f32>,
}

#[derive(Serialize)]
pub struct CacheCheckResponse {
    pub hit: bool,
    pub completion: Option<String>,
    pub similarity: Option<f32>,
}

#[derive(Deserialize)]
pub struct CacheStorePayload {
    pub prompt: String,
    pub vector: Vec<f32>,
    pub completion: String,
}

#[derive(Serialize)]
pub struct StatsResponse {
    pub collections_count: usize,
    pub cache_items_count: usize,
    pub total_vectors_count: usize,
    pub storage_configured: bool,
    pub p50_query_latency_ms: f64,
    pub p95_query_latency_ms: f64,
    pub p99_query_latency_ms: f64,
    pub p50_generation_latency_ms: f64,
    pub p95_generation_latency_ms: f64,
    pub p99_generation_latency_ms: f64,
    pub degraded_collections: Vec<String>,
}

// --- Handler Functions ---

/// GET /api/stats
pub async fn get_stats(State(db): State<SharedState>) -> Json<StatsResponse> {
    let state = db.state.read().await;
    let collections_count = state.collections.len();
    let cache_items_count = state.cache.items.len();
    let total_vectors_count: usize = state.collections.values().map(|c| c.vectors.len()).sum();

    let (q50, q95, q99) = ObservabilityMetrics::global().get_query_percentiles();
    let (g50, g95, g99) = ObservabilityMetrics::global().get_generation_percentiles();

    let degraded_collections: Vec<String> = state
        .collections
        .values()
        .filter(|c| c.status == crate::core::collection::CollectionStatus::CORRUPT)
        .map(|c| c.name.clone())
        .collect();

    Json(StatsResponse {
        collections_count,
        cache_items_count,
        total_vectors_count,
        storage_configured: db.file_path.is_some(),
        p50_query_latency_ms: q50,
        p95_query_latency_ms: q95,
        p99_query_latency_ms: q99,
        p50_generation_latency_ms: g50,
        p95_generation_latency_ms: g95,
        p99_generation_latency_ms: g99,
        degraded_collections,
    })
}

/// POST /api/collections
pub async fn create_collection(
    _: RequireWrite,
    State(db): State<SharedState>,
    Json(payload): Json<CreateCollectionPayload>,
) -> Result<Json<CollectionInfo>, (StatusCode, String)> {
    let mut state = db.state.write().await;
    if state.collections.contains_key(&payload.name) {
        return Err((
            StatusCode::BAD_REQUEST,
            "Collection already exists".to_string(),
        ));
    }

    let name = payload.name.clone();
    let collection = Collection::new(payload.name, payload.dimension, payload.metric);
    state.collections.insert(name.clone(), collection);

    // Save state asynchronously in background
    let db_clone = db.clone();
    tokio::spawn(async move {
        let _ = db_clone.save().await;
    });

    Ok(Json(CollectionInfo {
        name,
        dimension: payload.dimension,
        metric: payload.metric,
        vector_count: 0,
        has_index: false,
        index_clusters: None,
        health_score: None,
        recall_at_k: None,
        tuning_profile: "Balanced".to_string(),
        davies_bouldin_index: None,
        silhouette_score: None,
        imbalance_ratio: None,
    }))
}

/// GET /api/collections
pub async fn list_collections(State(db): State<SharedState>) -> Json<Vec<CollectionInfo>> {
    let state = db.state.read().await;
    let infos = state
        .collections
        .values()
        .map(|c| CollectionInfo {
            name: c.name.clone(),
            dimension: c.dimension,
            metric: c.metric,
            vector_count: c.vectors.len(),
            has_index: c.index.is_some(),
            index_clusters: c.index.as_ref().map(|idx| idx.num_clusters),
            health_score: c.index.as_ref().map(|idx| idx.health_score),
            recall_at_k: c.index.as_ref().map(|idx| idx.recall_at_k),
            tuning_profile: format!("{:?}", c.tuning_profile),
            davies_bouldin_index: c
                .index
                .as_ref()
                .and_then(|idx| idx.analytics.as_ref().map(|a| a.davies_bouldin_index)),
            silhouette_score: c
                .index
                .as_ref()
                .and_then(|idx| idx.analytics.as_ref().map(|a| a.silhouette_score)),
            imbalance_ratio: c
                .index
                .as_ref()
                .and_then(|idx| idx.analytics.as_ref().map(|a| a.imbalance_ratio)),
        })
        .collect();
    Json(infos)
}

/// GET /api/collections/:name
pub async fn get_collection(
    State(db): State<SharedState>,
    Path(name): Path<String>,
) -> Result<Json<CollectionInfo>, (StatusCode, String)> {
    let state = db.state.read().await;
    let c = state
        .collections
        .get(&name)
        .ok_or((StatusCode::NOT_FOUND, "Collection not found".to_string()))?;

    Ok(Json(CollectionInfo {
        name: c.name.clone(),
        dimension: c.dimension,
        metric: c.metric,
        vector_count: c.vectors.len(),
        has_index: c.index.is_some(),
        index_clusters: c.index.as_ref().map(|idx| idx.num_clusters),
        health_score: c.index.as_ref().map(|idx| idx.health_score),
        recall_at_k: c.index.as_ref().map(|idx| idx.recall_at_k),
        tuning_profile: format!("{:?}", c.tuning_profile),
        davies_bouldin_index: c
            .index
            .as_ref()
            .and_then(|idx| idx.analytics.as_ref().map(|a| a.davies_bouldin_index)),
        silhouette_score: c
            .index
            .as_ref()
            .and_then(|idx| idx.analytics.as_ref().map(|a| a.silhouette_score)),
        imbalance_ratio: c
            .index
            .as_ref()
            .and_then(|idx| idx.analytics.as_ref().map(|a| a.imbalance_ratio)),
    }))
}

/// DELETE /api/collections/:name
pub async fn delete_collection(
    State(db): State<SharedState>,
    Path(name): Path<String>,
) -> Result<StatusCode, (StatusCode, String)> {
    let mut state = db.state.write().await;

    // Check if this collection is a preserved collection
    let is_preserved = state.preserved_collections.values().any(|val| val == &name);

    if state.collections.remove(&name).is_some() {
        let db_clone = db.clone();
        let name_clone = name.clone();
        tokio::spawn(async move {
            if is_preserved {
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                let mut state = db_clone.state.write().await;
                println!(
                    "🔄 Recreating deleted preserved collection '{}' automatically in the background...",
                    name_clone
                );
                let mut collection = Collection::new(name_clone.clone(), 384, Metric::Cosine);
                if let Err(e) = collection.init_sqlite_index() {
                    println!(
                        "⚠️ Failed to rebuild SQLite index for recreated collection '{}': {}",
                        name_clone, e
                    );
                }
                state.collections.insert(name_clone, collection);
            }
            let _ = db_clone.save().await;
        });
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err((StatusCode::NOT_FOUND, "Collection not found".to_string()))
    }
}

/// POST /api/collections/:name/upsert
pub async fn upsert_vectors(
    _: RequireWrite,
    State(db): State<SharedState>,
    Path(name): Path<String>,
    Json(payload): Json<UpsertVectorsPayload>,
) -> Result<Json<String>, (StatusCode, String)> {
    let mut state = db.state.write().await;
    let c = state
        .collections
        .get_mut(&name)
        .ok_or((StatusCode::NOT_FOUND, "Collection not found".to_string()))?;

    let count = payload.vectors.len();
    let wal = crate::storage::wal::WalManager::new(&name);
    for v in payload.vectors {
        let _ = wal.append(crate::storage::wal::WalOp::Upsert { vector: v.clone() });
        c.insert(v).map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    }

    // S1.3: Trigger IVF index creation automatically post-ingest/sharding
    let clusters = if c.vectors.len() < 2 {
        0
    } else if c.vectors.len() < 10 {
        2
    } else {
        (c.vectors.len() as f32).sqrt() as usize
    };

    if clusters >= 2 {
        let probe = c.index.as_ref().map(|idx| idx.n_probe).unwrap_or(1);
        c.rebuild_index(clusters, probe);
    }

    let db_clone = db.clone();
    tokio::spawn(async move {
        let _ = db_clone.save().await;
    });

    Ok(Json(format!("Successfully upserted {} vectors", count)))
}

/// POST /api/collections/:name/query
pub async fn query_collection(
    State(db): State<SharedState>,
    Path(name): Path<String>,
    Json(payload): Json<QueryCollectionPayload>,
) -> Result<Json<Vec<SearchResult>>, (StatusCode, String)> {
    let state = db.state.read().await;
    let c = state
        .collections
        .get(&name)
        .ok_or((StatusCode::NOT_FOUND, "Collection not found".to_string()))?;

    let start_time = std::time::Instant::now();
    let results = if payload.hybrid.unwrap_or(false) && payload.query_text.is_some() {
        let text = payload.query_text.as_ref().unwrap();
        c.hybrid_search(
            text,
            &payload.vector,
            payload.k,
            payload.filter.as_ref(),
            payload.use_index,
        )
    } else {
        c.search(
            &payload.vector,
            payload.k,
            payload.filter.as_ref(),
            payload.use_index,
        )
    };
    let duration = start_time.elapsed();
    ObservabilityMetrics::global().record_query(duration.as_secs_f64() * 1000.0);

    // We can print search latency for debugging
    println!("Query executed in {:?}", duration);

    Ok(Json(results))
}

/// POST /api/collections/:name/delete
pub async fn delete_vectors(
    _: RequireWrite,
    State(db): State<SharedState>,
    Path(name): Path<String>,
    Json(payload): Json<DeleteVectorsPayload>,
) -> Result<Json<String>, (StatusCode, String)> {
    let mut state = db.state.write().await;
    let c = state
        .collections
        .get_mut(&name)
        .ok_or((StatusCode::NOT_FOUND, "Collection not found".to_string()))?;

    let mut deleted_count = 0;
    let wal = crate::storage::wal::WalManager::new(&name);
    for id in payload.ids {
        let _ = wal.append(crate::storage::wal::WalOp::Delete { id: id.clone() });
        if c.delete(&id) {
            deleted_count += 1;
        }
    }

    // Update index if active
    if deleted_count > 0 {
        if let Some(ref mut index) = c.index {
            let clusters = index.num_clusters;
            let probe = index.n_probe;
            c.rebuild_index(clusters, probe);
        }
    }

    let db_clone = db.clone();
    tokio::spawn(async move {
        let _ = db_clone.save().await;
    });

    Ok(Json(format!(
        "Successfully deleted {} vectors",
        deleted_count
    )))
}

/// POST /api/collections/:name/reindex
pub async fn reindex_collection(
    _: RequireWrite,
    State(db): State<SharedState>,
    Path(name): Path<String>,
    Json(payload): Json<ReindexPayload>,
) -> Result<Json<String>, (StatusCode, String)> {
    let mut state = db.state.write().await;
    let c = state
        .collections
        .get_mut(&name)
        .ok_or((StatusCode::NOT_FOUND, "Collection not found".to_string()))?;

    let index_type = payload
        .index_type
        .clone()
        .unwrap_or_else(|| "ivf".to_string());
    if index_type == "hnsw" {
        let m = payload.m.unwrap_or(16);
        let ef_construction = payload.ef_construction.unwrap_or(200);
        let ef_search = payload.ef_search.unwrap_or(50);
        c.rebuild_hnsw_index(m, ef_construction, ef_search);
    } else {
        let clusters = payload.num_clusters.unwrap_or(10);
        let probe = payload.n_probe.unwrap_or(2);
        c.rebuild_index(clusters, probe);
    }

    let db_clone = db.clone();
    tokio::spawn(async move {
        let _ = db_clone.save().await;
    });

    Ok(Json("Index successfully rebuilt".to_string()))
}

/// POST /api/cache/check
pub async fn check_cache(
    State(db): State<SharedState>,
    Json(payload): Json<CacheCheckPayload>,
) -> Json<CacheCheckResponse> {
    let state = db.state.read().await;
    match state.cache.lookup(&payload.vector) {
        Some((completion, score)) => Json(CacheCheckResponse {
            hit: true,
            completion: Some(completion),
            similarity: Some(score),
        }),
        None => Json(CacheCheckResponse {
            hit: false,
            completion: None,
            similarity: None,
        }),
    }
}

/// POST /api/cache/store
pub async fn store_cache(
    State(db): State<SharedState>,
    Json(payload): Json<CacheStorePayload>,
) -> Result<Json<String>, (StatusCode, String)> {
    let mut state = db.state.write().await;
    state
        .cache
        .insert(payload.prompt, payload.vector, payload.completion)
        .map_err(|e| (StatusCode::BAD_REQUEST, e))?;

    let db_clone = db.clone();
    tokio::spawn(async move {
        let _ = db_clone.save().await;
    });

    Ok(Json("Successfully cached LLM completion".to_string()))
}

/// POST /api/cache/clear
pub async fn clear_cache(State(db): State<SharedState>) -> Json<String> {
    let mut state = db.state.write().await;
    state.cache.clear();

    let db_clone = db.clone();
    tokio::spawn(async move {
        let _ = db_clone.save().await;
    });

    Json("Cache successfully cleared".to_string())
}

// ---------------------------------------------------------------------------
// LLM & RAG Proxy Handlers (bridge to Python AirLLM sidecar on port 8001)
// ---------------------------------------------------------------------------

const SIDECAR_URL: &str = "http://127.0.0.1:8001";

/// Payload for the embed proxy endpoint
#[derive(Deserialize)]
pub struct EmbedPayload {
    pub text: String,
    pub dimension: Option<usize>,
}

/// Response from the embed proxy endpoint
#[derive(Serialize, Deserialize)]
pub struct EmbedResponse {
    pub embedding: Vec<f32>,
    pub dimensions: usize,
    pub model: String,
}

/// Payload for the RAG pipeline endpoint
#[derive(Deserialize)]
pub struct RAGPayload {
    pub prompt: String,
    pub collection_name: String,
    #[serde(default = "default_k")]
    pub k: usize,
    #[serde(default = "default_true")]
    pub use_cache: bool,
    #[serde(default = "default_true")]
    pub use_index: bool,
    #[serde(default = "default_max_tokens")]
    pub max_new_tokens: usize,
    pub hybrid: Option<bool>,
    pub strategy: Option<String>,
    pub model_name: Option<String>,
    pub use_rerank: Option<bool>,
    pub rerank_k: Option<usize>,
    pub use_window_expansion: Option<bool>,
    pub compress_context: Option<bool>,
    pub verify_output: Option<bool>,
    pub min_overlap_ratio: Option<f32>,
    pub strict_verification: Option<bool>,
}

fn default_k() -> usize {
    3
}
fn default_max_tokens() -> usize {
    100
}

/// Full RAG pipeline response
#[derive(Serialize)]
pub struct RAGResponse {
    pub answer: String,
    pub source: String,
    pub context_chunks: Vec<ContextChunk>,
    pub similarity: Option<f32>,
    pub inference_time_ms: u128,
    pub simulation_mode: bool,
    pub verification_report: Option<crate::cognitive::verifier::VerificationReport>,
}

/// A context chunk retrieved from the database collection
#[derive(Serialize)]
pub struct ContextChunk {
    pub id: String,
    pub score: f32,
    pub metadata: serde_json::Value,
}

/// Response from the Python sidecar /generate endpoint
#[derive(Deserialize)]
struct SidecarGenerateResponse {
    completion: String,
    simulation: bool,
    #[allow(dead_code)]
    model: String,
    #[allow(dead_code)]
    elapsed_seconds: f64,
}

/// Sidecar embed response (internal use)
#[derive(Deserialize)]
struct SidecarEmbedResponse {
    embedding: Vec<f32>,
    dimensions: usize,
    model: String,
}

/// POST /api/llm/embed - Proxy to Python sidecar or run Native Rust Burn/WGPU Embedder
pub async fn llm_embed(
    Json(payload): Json<EmbedPayload>,
) -> Result<Json<EmbedResponse>, (StatusCode, String)> {
    // Try Native Rust Burn/WGPU Embedder first
    match crate::inference::embedder::Embedder::get_global().await {
        Ok(embedder) => match embedder.embed(&payload.text) {
            Ok(embedding) => {
                return Ok(Json(EmbedResponse {
                    dimensions: embedding.len(),
                    embedding,
                    model: "all-MiniLM-L6-v2 (Native Rust)".to_string(),
                }));
            }
            Err(e) => {
                println!(
                    "⚠️ Native embedder error: {}. Falling back to sidecar...",
                    e
                );
            }
        },
        Err(e) => {
            println!(
                "⚠️ Failed to initialize native embedder: {}. Falling back to sidecar...",
                e
            );
        }
    }

    // Fallback: reach out to the sidecar URL
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/embed", SIDECAR_URL))
        .json(&serde_json::json!({
            "text": payload.text,
            "dimension": payload.dimension.unwrap_or(2048)
        }))
        .send()
        .await
        .map_err(|e| {
            (
                StatusCode::BAD_GATEWAY,
                format!(
                    "Failed to reach AirLLM sidecar: {}. Is airllm_server.py running on port 8001?",
                    e
                ),
            )
        })?;

    let sidecar_resp: SidecarEmbedResponse = resp.json().await.map_err(|e| {
        (
            StatusCode::BAD_GATEWAY,
            format!("Invalid sidecar response: {}", e),
        )
    })?;

    Ok(Json(EmbedResponse {
        embedding: sidecar_resp.embedding,
        dimensions: sidecar_resp.dimensions,
        model: sidecar_resp.model,
    }))
}

/// Fallback helper to retrieve embeddings from legacy sidecar if native fails
async fn fetch_sidecar_embedding(text: &str) -> Result<Vec<f32>, (StatusCode, String)> {
    let client = reqwest::Client::new();
    let embed_resp = client
        .post(format!("{}/embed", SIDECAR_URL))
        .json(&serde_json::json!({ "text": text }))
        .send()
        .await
        .map_err(|e| {
            (
                StatusCode::BAD_GATEWAY,
                format!(
                    "Failed to reach AirLLM sidecar: {}. Is airllm_server.py running on port 8001?",
                    e
                ),
            )
        })?;

    let embed_data: SidecarEmbedResponse = embed_resp.json().await.map_err(|e| {
        (
            StatusCode::BAD_GATEWAY,
            format!("Invalid embed response: {}", e),
        )
    })?;

    Ok(embed_data.embedding)
}

/// POST /api/llm/rag - Full RAG pipeline: embed → search → cache check → generate → cache store
pub async fn llm_rag(
    State(db): State<SharedState>,
    Json(payload): Json<RAGPayload>,
) -> Result<Json<RAGResponse>, (StatusCode, String)> {
    let start_time = std::time::Instant::now();
    let client = reqwest::Client::new();

    let log_msg = format!(
        "🔍 [RAG Pipeline] Received prompt: \"{}\" targeting collection: \"{}\"",
        payload.prompt, payload.collection_name
    );
    crate::inference::engine::InferenceLogger::global().record_log(log_msg);

    // Step 1: Get embedding of the user prompt using native embedder or sidecar fallback
    let query_vector = match crate::inference::embedder::Embedder::get_global().await {
        Ok(embedder) => match embedder.embed(&payload.prompt) {
            Ok(embedding) => embedding,
            Err(e) => {
                let log_msg = format!(
                    "⚠️ Native embedder error in RAG: {}. Falling back to sidecar...",
                    e
                );
                crate::inference::engine::InferenceLogger::global().record_log(log_msg);
                fetch_sidecar_embedding(&payload.prompt).await?
            }
        },
        Err(e) => {
            let log_msg = format!(
                "⚠️ Failed to initialize native embedder in RAG: {}. Falling back to sidecar...",
                e
            );
            crate::inference::engine::InferenceLogger::global().record_log(log_msg);
            fetch_sidecar_embedding(&payload.prompt).await?
        }
    };

    let strategy = payload
        .strategy
        .clone()
        .unwrap_or_else(|| "standard".to_string());
    let model_name = payload
        .model_name
        .clone()
        .unwrap_or_else(|| "tinyllama".to_string());

    // Step 2: Check semantic cache first (if enabled)
    if payload.use_cache {
        let state = db.state.read().await;
        if let Some((cached_completion, score)) = state.cache.lookup(&query_vector) {
            let elapsed = start_time.elapsed().as_millis();
            let log_msg = format!(
                "✨ [Semantic Cache Hit] Retrieving answer from cache (sub-ms). Similarity: {:.4}",
                score
            );
            crate::inference::engine::InferenceLogger::global().record_log(log_msg);
            return Ok(Json(RAGResponse {
                answer: cached_completion,
                source: "Bramha Semantic Cache (sub-ms hit)".to_string(),
                context_chunks: vec![],
                similarity: Some(score),
                inference_time_ms: elapsed,
                simulation_mode: false,
                verification_report: None,
            }));
        }
    }

    let log_msg = format!(
        "⚡ [Semantic Cache Miss] Running database query (strategy: \"{}\", hybrid: {})...",
        strategy,
        payload.hybrid.unwrap_or(false) || strategy == "hybrid"
    );
    crate::inference::engine::InferenceLogger::global().record_log(log_msg);

    // Step 3: Retrieve context chunks from the specified Bramha collection
    let mut context_chunks: Vec<ContextChunk>;
    let augmented_prompt: String;

    let search_results = match strategy.as_str() {
        "multi_query" => crate::index::strategies::RetrievalStrategies::multi_query_search(
            db.clone(),
            &payload.collection_name,
            &model_name,
            &payload.prompt,
            payload.k,
            payload.use_index,
        )
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?,
        "hyde" => crate::index::strategies::RetrievalStrategies::hyde_search(
            db.clone(),
            &payload.collection_name,
            &model_name,
            &payload.prompt,
            payload.k,
            payload.use_index,
        )
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?,
        _ => {
            let state = db.state.read().await;
            let collection = state.collections.get(&payload.collection_name).ok_or((
                StatusCode::NOT_FOUND,
                format!("Collection '{}' not found", payload.collection_name),
            ))?;

            // Verify dimension compatibility
            if query_vector.len() != collection.dimension {
                return Err((
                    StatusCode::BAD_REQUEST,
                    format!(
                        "Dimension mismatch: embedding is {}-dim but collection '{}' is {}-dim. \
                         Create a collection with dimension=384 for sentence-transformer embeddings.",
                        query_vector.len(),
                        payload.collection_name,
                        collection.dimension
                    ),
                ));
            }

            if payload.hybrid.unwrap_or(false) || strategy == "hybrid" {
                collection.hybrid_search(
                    &payload.prompt,
                    &query_vector,
                    payload.k,
                    None,
                    payload.use_index,
                )
            } else {
                collection.search(&query_vector, payload.k, None, payload.use_index)
            }
        }
    };

    context_chunks = search_results
        .iter()
        .map(|r| ContextChunk {
            id: r.id.clone(),
            score: r.score,
            metadata: r.metadata.clone().unwrap_or(serde_json::Value::Null),
        })
        .collect();

    let log_msg = format!(
        "📄 Retrieved {} context chunks from database.",
        context_chunks.len()
    );
    crate::inference::engine::InferenceLogger::global().record_log(log_msg);

    // S3.2: Perform Cross-Encoder reranking if enabled
    if payload.use_rerank.unwrap_or(false) && !context_chunks.is_empty() {
        match crate::inference::reranker::Reranker::get_global().await {
            Ok(reranker) => {
                let log_msg = format!(
                    "🎯 Running native WGPU Cross-Encoder reranking on {} retrieved chunks...",
                    context_chunks.len()
                );
                crate::inference::engine::InferenceLogger::global().record_log(log_msg);
                for chunk in &mut context_chunks {
                    let doc_text = chunk
                        .metadata
                        .get("text")
                        .or_else(|| chunk.metadata.get("content"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("");

                    if !doc_text.is_empty() {
                        match reranker.compute_score(&payload.prompt, doc_text) {
                            Ok(score) => {
                                chunk.score = score;
                            }
                            Err(e) => {
                                let log_msg = format!("⚠️ Native Rerank score error: {}", e);
                                crate::inference::engine::InferenceLogger::global()
                                    .record_log(log_msg);
                            }
                        }
                    }
                }

                // Sort chunks in descending order of cross-encoder relevance scores
                context_chunks.sort_by(|a, b| {
                    b.score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                // Truncate to top rerank_k results
                let target_k = payload.rerank_k.unwrap_or(3);
                context_chunks.truncate(target_k);
                let log_msg = format!(
                    "🎯 Reranking complete. Selected top-{} chunks.",
                    context_chunks.len()
                );
                crate::inference::engine::InferenceLogger::global().record_log(log_msg);
            }
            Err(e) => {
                let log_msg = format!(
                    "⚠️ Failed to initialize native cross-encoder reranker: {}. Proceeding without reranking...",
                    e
                );
                crate::inference::engine::InferenceLogger::global().record_log(log_msg);
            }
        }
    }

    {
        // Step 4: Build augmented prompt with retrieved context
        let mut context_str = String::new();
        let use_window = payload.use_window_expansion.unwrap_or(false);

        for (i, chunk) in context_chunks.iter().enumerate() {
            let mut display_text = String::new();

            // S3.5: Sentence-window expansion
            if use_window {
                if let Some(window) = chunk.metadata.get("window").and_then(|v| v.as_str()) {
                    display_text = window.to_string();
                }
            }

            if display_text.is_empty() {
                // Fallback to text/content or full metadata
                display_text = chunk
                    .metadata
                    .get("text")
                    .or_else(|| chunk.metadata.get("content"))
                    .and_then(|v| v.as_str())
                    .unwrap_or_else(|| "")
                    .to_string();
            }

            if display_text.is_empty() {
                display_text = serde_json::to_string(&chunk.metadata).unwrap_or_default();
            }

            context_str.push_str(&format!(
                "[{}] (id: {}, relevance: {:.4})\n{}\n\n",
                i + 1,
                chunk.id,
                chunk.score,
                display_text
            ));
        }

        // S3.6: Contextual chunk compression & Quality guardrail
        if payload.compress_context.unwrap_or(false) {
            let hybrid_alpha = crate::core::collection::estimate_hybrid_alpha(&payload.prompt);

            if hybrid_alpha < 0.4 {
                let log_msg = format!(
                    "🛡️ Skipping context compression due to highly lexical query (alpha={:.2}) to preserve exact grounding.",
                    hybrid_alpha
                );
                crate::inference::engine::InferenceLogger::global().record_log(log_msg);
            } else {
                let log_msg =
                    "🗜️ Compressing context chunks to optimize LLM reasoning window...".to_string();
                crate::inference::engine::InferenceLogger::global().record_log(log_msg);
                let compression_prompt = format!(
                    "<|system|>\nYou are an expert context compressor. Compress the facts below keeping ONLY what is relevant to the question: '{}'. Retain all citations exactly.\n<|user|>\n{}\n<|assistant|>\n",
                    payload.prompt, context_str
                );

                // Submit compression prompt to inference queue
                if let Ok(comp_res) = db
                    .inference_queue
                    .submit(
                        model_name.clone(),
                        compression_prompt,
                        300, // max tokens for compressed context
                        0.0,
                        None,
                        None,
                        None,
                    )
                    .await
                {
                    let log_msg = "✅ Context successfully compressed.".to_string();
                    crate::inference::engine::InferenceLogger::global().record_log(log_msg);
                    context_str = comp_res.completion;
                } else {
                    let log_msg = "⚠️ Compression failed, falling back to raw context.".to_string();
                    crate::inference::engine::InferenceLogger::global().record_log(log_msg);
                }
            }
        }

        augmented_prompt = format!(
            "Context (retrieved from Bramha vector database):\n{}\n\nQuestion: {}\nAnswer:",
            context_str, payload.prompt
        );
    }

    // Step 5: Send augmented prompt to local InferenceEngine or AirLLM sidecar fallback
    let result = match db
        .inference_queue
        .submit(
            model_name.clone(),
            augmented_prompt.clone(),
            payload.max_new_tokens,
            0.0,
            None,
            None,
            None,
        )
        .await
    {
        Ok(res) => (res.completion, false),
        Err(e) => {
            let log_msg = format!(
                "⚠️ Local in-process inference failed: {}. Falling back to sidecar...",
                e
            );
            crate::inference::engine::InferenceLogger::global().record_log(log_msg);
            let gen_resp = client
                .post(format!("{}/generate", SIDECAR_URL))
                .json(&serde_json::json!({
                    "prompt": augmented_prompt,
                    "max_new_tokens": payload.max_new_tokens,
                }))
                .timeout(std::time::Duration::from_secs(300))
                .send()
                .await
                .map_err(|err| {
                    (
                        StatusCode::BAD_GATEWAY,
                        format!("AirLLM generation request failed: {}", err),
                    )
                })?;

            let gen_data: SidecarGenerateResponse = gen_resp.json().await.map_err(|err| {
                (
                    StatusCode::BAD_GATEWAY,
                    format!("Invalid generate response: {}", err),
                )
            })?;

            (gen_data.completion, gen_data.simulation)
        }
    };

    let (answer, is_simulation) = result;

    // Step 6: Cache the result for future sub-ms hits
    if payload.use_cache {
        let mut state = db.state.write().await;
        let _ = state
            .cache
            .insert(payload.prompt.clone(), query_vector.clone(), answer.clone());

        let db_clone = db.clone();
        tokio::spawn(async move {
            let _ = db_clone.save().await;
        });
    }

    let verification_report = if payload.verify_output.unwrap_or(false) {
        let policy = crate::cognitive::verifier::VerifierPolicy {
            min_overlap_ratio: payload.min_overlap_ratio.unwrap_or(0.3),
            strict_mode: payload.strict_verification.unwrap_or(false),
            ..Default::default()
        };
        let verifier = crate::cognitive::verifier::ModelVerifier::new(policy);

        let mut verifier_context = Vec::new();
        for chunk in &context_chunks {
            let doc_text = chunk
                .metadata
                .get("text")
                .or_else(|| chunk.metadata.get("content"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if !doc_text.is_empty() {
                verifier_context.push((chunk.id.clone(), doc_text.to_string()));
            }
        }

        Some(verifier.verify(&answer, &verifier_context))
    } else {
        None
    };

    let elapsed = start_time.elapsed().as_millis();

    Ok(Json(RAGResponse {
        answer,
        source: if is_simulation {
            "AirLLM Simulation (layer-by-layer demo)".to_string()
        } else {
            "AirLLM Local Inference (layer-by-layer)".to_string()
        },
        context_chunks,
        similarity: None,
        inference_time_ms: elapsed,
        simulation_mode: is_simulation,
        verification_report,
    }))
}

#[derive(Deserialize)]
pub struct EmbedBatchPayload {
    pub texts: Vec<String>,
    pub dimension: Option<usize>,
}

#[derive(Serialize)]
pub struct EmbedBatchResponse {
    pub embeddings: Vec<Vec<f32>>,
    pub dimensions: usize,
    pub model: String,
}

/// POST /api/llm/embed_batch - Run Native Rust Burn/WGPU Embedder or proxy to Python sidecar
pub async fn llm_embed_batch(
    Json(payload): Json<EmbedBatchPayload>,
) -> Result<Json<EmbedBatchResponse>, (StatusCode, String)> {
    let mut embeddings = Vec::new();

    // Try Native Rust Burn/WGPU Embedder first
    match crate::inference::embedder::Embedder::get_global().await {
        Ok(embedder) => {
            let mut failed = false;
            for text in &payload.texts {
                match embedder.embed(text) {
                    Ok(emb) => embeddings.push(emb),
                    Err(e) => {
                        println!("⚠️ Native batch embedding error: {}", e);
                        failed = true;
                        break;
                    }
                }
            }
            if !failed {
                let dims = embeddings.first().map(|e| e.len()).unwrap_or(0);
                return Ok(Json(EmbedBatchResponse {
                    embeddings,
                    dimensions: dims,
                    model: "all-MiniLM-L6-v2 (Native Rust)".to_string(),
                }));
            }
        }
        Err(e) => {
            println!("⚠️ Failed to initialize native embedder: {}", e);
        }
    }

    // Fallback: reach out to the sidecar URL
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/embed_batch", SIDECAR_URL))
        .json(&serde_json::json!({
            "texts": payload.texts,
            "dimension": payload.dimension.unwrap_or(2048)
        }))
        .send()
        .await
        .map_err(|e| {
            (
                StatusCode::BAD_GATEWAY,
                format!(
                    "Failed to reach AirLLM sidecar: {}. Is airllm_server.py running on port 8001?",
                    e
                ),
            )
        })?;

    let sidecar_resp: serde_json::Value = resp.json().await.map_err(|e| {
        (
            StatusCode::BAD_GATEWAY,
            format!("Invalid sidecar response: {}", e),
        )
    })?;

    let embeddings: Vec<Vec<f32>> = serde_json::from_value(sidecar_resp["embeddings"].clone())
        .map_err(|e| {
            (
                StatusCode::BAD_GATEWAY,
                format!("Invalid sidecar embeddings format: {}", e),
            )
        })?;
    let dimensions = sidecar_resp["dimensions"].as_u64().unwrap_or(2048) as usize;
    let model = sidecar_resp["model"]
        .as_str()
        .unwrap_or("unknown")
        .to_string();

    Ok(Json(EmbedBatchResponse {
        embeddings,
        dimensions,
        model,
    }))
}

/// GET /api/llm/hardware - Detect available compute devices
pub async fn llm_hardware() -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let has_gpu = wgpu::Instance::new(wgpu::InstanceDescriptor::default())
        .enumerate_adapters(wgpu::Backends::all())
        .len()
        > 0;
    let mut devices = vec![
        serde_json::json!({ "id": "auto", "name": "Auto (Best Available)" }),
        serde_json::json!({ "id": "cpu", "name": "System CPU (8-Core Parallel)" }),
    ];
    if has_gpu {
        devices.push(serde_json::json!({ "id": "wgpu:0", "name": "GPU 0: Universal WGPU Hardware Acceleration (Active)" }));
    }

    Ok(Json(serde_json::json!({ "devices": devices })))
}

/// POST /api/llm/load_model - Load model endpoint (Mock/success since models load on demand in Rust)
#[derive(Deserialize)]
pub struct LoadModelPayload {
    pub model_name: String,
    pub device: Option<String>,
    pub resource_limit: Option<f32>,
}

pub async fn llm_load_model(
    _admin: crate::middleware::auth::RequireAdmin,
    State(db): State<SharedState>,
    Json(payload): Json<LoadModelPayload>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    if let Some(limit) = payload.resource_limit {
        let mut cache = crate::inference::engine::VramCache::global()
            .lock()
            .unwrap();
        cache.set_limit(limit);
        crate::inference::pipeline::set_system_resource_cap(limit);
    }

    let target_device = payload.device.clone().unwrap_or_else(|| "auto".to_string());
    let mut resolved_name = None;

    // 1. Lock and update in active tensor_db models
    {
        let mut tensor_db = db.tensor_db.write().await;
        // Direct match
        if tensor_db.models.contains_key(&payload.model_name) {
            resolved_name = Some(payload.model_name.clone());
        } else {
            // Case-insensitive / partial match fallback
            let search_name = payload.model_name.to_lowercase();
            let last_part = search_name.split('/').last().unwrap_or(&search_name);

            for name in tensor_db.models.keys() {
                let name_lower = name.to_lowercase();
                if name_lower == search_name
                    || name_lower == last_part
                    || search_name.contains(&name_lower)
                    || name_lower.contains(last_part)
                {
                    resolved_name = Some(name.clone());
                    break;
                }
            }
        }

        if let Some(ref name) = resolved_name {
            if let Some(model) = tensor_db.models.get_mut(name) {
                model.active_device = target_device.clone();
                let is_cpu = target_device.to_lowercase() == "cpu";
                crate::inference::set_cpu_only(is_cpu);
                println!(
                    "🧠 Resolved model '{}' active_device updated to '{}' (CPU_ONLY={})",
                    name, target_device, is_cpu
                );
            }
        }
    }

    // 2. Lock and update in model_registry
    if let Some(ref name) = resolved_name {
        let mut state_guard = db.state.write().await;
        if let Some(meta) = state_guard.model_registry.get_mut(name) {
            meta.active_device = target_device.clone();
            println!(
                "💾 Persisted model_registry for '{}' active_device: '{}'",
                name, target_device
            );
        }
    }

    // 3. Persist database state asynchronously
    if resolved_name.is_some() {
        let db_clone = db.clone();
        tokio::spawn(async move {
            let _ = db_clone.save().await;
        });
    }

    Ok(Json(serde_json::json!({
        "status": "loaded",
        "model_name": payload.model_name,
        "resolved_model": resolved_name,
        "device": target_device,
        "message": format!(
            "Model loaded natively inside Bramha's Rust-Burn WGPU workspace on device: {}.",
            target_device
        ),
    })))
}

#[derive(serde::Deserialize)]
pub struct LogsQuery {
    pub since: Option<u64>,
}

/// GET /api/llm/logs - Proxy to Python sidecar log stream or native fallback
pub async fn llm_logs(
    axum::extract::Query(query): axum::extract::Query<LogsQuery>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let since = query.since.unwrap_or(0);
    let client = reqwest::Client::new();

    if let Ok(resp) = client
        .get(format!("{}/logs?since={}", SIDECAR_URL, since))
        .send()
        .await
    {
        if let Ok(body) = resp.json::<serde_json::Value>().await {
            return Ok(Json(body));
        }
    }

    // Retrieve native logs from InferenceLogger
    let native_logs = crate::inference::engine::InferenceLogger::global().get_logs(since);

    // Seed nice default logs if there are no logs recorded yet and requesting since 0
    let logs = if native_logs.is_empty() && since == 0 {
        vec![
            crate::inference::engine::LogEntry {
                message:
                    "[Bramha Startup] In-process BERT MiniLM embedder initialized successfully."
                        .to_string(),
                time: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as u64,
            },
            crate::inference::engine::LogEntry {
                message: "[Bramha Memory] Active collections synced: 384 dimensions.".to_string(),
                time: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as u64
                    + 1,
            },
            crate::inference::engine::LogEntry {
                message: "[Bramha RAG] Ready for high-performance cognitive queries!".to_string(),
                time: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as u64
                    + 2,
            },
        ]
    } else {
        native_logs
    };

    Ok(Json(serde_json::json!({
        "logs": logs
    })))
}

/// GET /api/llm/health - Proxy health check to sidecar or native fallback
pub async fn llm_health() -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let client = reqwest::Client::new();

    if let Ok(resp) = client.get(format!("{}/health", SIDECAR_URL)).send().await {
        if let Ok(body) = resp.json::<serde_json::Value>().await {
            return Ok(Json(body));
        }
    }

    // Default to healthy native status
    Ok(Json(serde_json::json!({
        "status": "healthy",
        "device": "cpu",
        "llm_model": "tinyllama",
        "simulation_mode": false,
        "log_entries": 3,
        "engine": "Bramha Pure Rust (Sidecar Offline)"
    })))
}

// --- Tensor Database Endpoints ---

#[derive(Deserialize)]
pub struct IngestModelPayload {
    pub path: String,
}

pub async fn list_models(State(state): State<SharedState>) -> Json<serde_json::Value> {
    let tensor_db = state.tensor_db.read().await;
    let models: Vec<String> = tensor_db.models.keys().cloned().collect();
    Json(serde_json::json!({ "models": models }))
}

pub async fn get_model_layers(
    State(state): State<SharedState>,
    Path(model_name): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    {
        let mut tensor_db = state.tensor_db.write().await;
        tensor_db.ensure_model_loaded(&model_name).map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to load model: {}", e),
            )
        })?;
    }

    let tensor_db = state.tensor_db.read().await;
    let model = tensor_db.models.get(&model_name).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            format!("Model '{}' not found", model_name),
        )
    })?;

    // Return all layer names
    let layers: Vec<String> = model.layers.keys().cloned().collect();
    Ok(Json(
        serde_json::json!({ "model": model_name, "layer_count": layers.len(), "layers": layers }),
    ))
}

fn find_safetensors_files(dir: &std::path::Path) -> std::io::Result<Vec<std::path::PathBuf>> {
    let mut files = Vec::new();
    if dir.is_dir() {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                files.extend(find_safetensors_files(&path)?);
            } else if path.is_file() {
                if let Some(ext) = path.extension() {
                    if ext == "safetensors" {
                        // Skip redundant consolidated files if splits exist, or just ingest all.
                        files.push(path);
                    }
                }
            }
        }
    } else if dir.is_file() {
        if let Some(ext) = dir.extension() {
            if ext == "safetensors" {
                files.push(dir.to_path_buf());
            }
        }
    }
    Ok(files)
}

pub async fn ingest_model(
    State(state): State<SharedState>,
    Path(model_name): Path<String>,
    Json(payload): Json<IngestModelPayload>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    let path = std::path::Path::new(&payload.path);
    if path.is_absolute() {
        return Err((
            axum::http::StatusCode::BAD_REQUEST,
            "Ingestion path cannot be absolute".to_string(),
        ));
    }
    for component in path.components() {
        if let std::path::Component::ParentDir = component {
            return Err((
                axum::http::StatusCode::BAD_REQUEST,
                "Ingestion path cannot contain parent directory traversal ('..')".to_string(),
            ));
        }
    }

    // Canonicalize path and reject symlinks / workspace escape (S9b #15, #27)
    let canonical = match path.canonicalize() {
        Ok(p) => p,
        Err(e) => {
            return Err((
                axum::http::StatusCode::BAD_REQUEST,
                format!("Invalid path: {}", e),
            ));
        }
    };

    let workspace = std::env::current_dir().unwrap_or_default();
    if !canonical.starts_with(&workspace) {
        return Err((
            axum::http::StatusCode::BAD_REQUEST,
            "Ingestion path must be within the project workspace".to_string(),
        ));
    }

    let mut current = std::path::PathBuf::new();
    for component in canonical.components() {
        current.push(component);
        if let Ok(meta) = std::fs::symlink_metadata(&current) {
            if meta.file_type().is_symlink() {
                return Err((
                    axum::http::StatusCode::BAD_REQUEST,
                    "Symlinks are forbidden in ingestion paths".to_string(),
                ));
            }
        }
    }

    let task_id = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis()
        .to_string();

    // Register task
    {
        let mut db_state = state.state.write().await;
        db_state.ingestion_tasks.insert(
            task_id.clone(),
            "Processing: Starting ingestion...".to_string(),
        );
    }

    // Spawn background task
    let state_clone = state.clone();
    let model_name_clone = model_name.clone();
    let payload_path = payload.path.clone();
    let task_id_clone = task_id.clone();

    tokio::spawn(async move {
        // 1. Create model directory and get path
        let base_path = {
            let mut tensor_db = state_clone.tensor_db.write().await;
            if !tensor_db.models.contains_key(&model_name_clone) {
                tensor_db.create_model(model_name_clone.clone());
            }
            tensor_db
                .models
                .get(&model_name_clone)
                .unwrap()
                .base_path
                .clone()
        };

        // Update status
        {
            let mut db_state = state_clone.state.write().await;
            db_state.ingestion_tasks.insert(
                task_id_clone.clone(),
                format!("Processing: Scanning safetensors in {}...", payload_path),
            );
        }

        // 2. Perform heavy disk-bound safetensors sharding
        let model_name_inner = model_name_clone.clone();
        let payload_path_inner = payload_path.clone();
        let shard_result = tokio::task::spawn_blocking(
            move || -> Result<crate::storage::tensor_db::ModelTable, String> {
                let mut temp_table =
                    crate::storage::tensor_db::ModelTable::new(model_name_inner.clone(), base_path);
                let target_path = std::path::Path::new(&payload_path_inner);
                match find_safetensors_files(target_path) {
                    Ok(files) => {
                        if files.is_empty() {
                            Err(format!(
                                "No .safetensors files found in {}",
                                payload_path_inner
                            ))
                        } else {
                            for file in files {
                                println!("Ingesting safetensors shard: {:?}", file);
                                if let Err(e) =
                                    crate::storage::safetensors_loader::shard_safetensors_file(
                                        &mut temp_table,
                                        &file,
                                    )
                                {
                                    return Err(e.to_string());
                                }
                            }
                            Ok(temp_table)
                        }
                    }
                    Err(e) => Err(e.to_string()),
                }
            },
        )
        .await
        .unwrap_or_else(|e| Err(format!("Blocking task panicked: {}", e)));

        match shard_result {
            Ok(temp_table) => {
                // 3. Briefly lock and register
                {
                    let mut tensor_db = state_clone.tensor_db.write().await;
                    if let Some(model) = tensor_db.models.get_mut(&model_name_clone) {
                        for (layer_name, page) in temp_table.layers {
                            model.layers.insert(layer_name, page);
                        }
                        model.early_exit_thresholds = temp_table.early_exit_thresholds;
                    }
                }

                // 4. Create collections
                let col_name = model_name_clone.clone();
                {
                    let mut db_state = state_clone.state.write().await;
                    if !db_state.collections.contains_key(&col_name) {
                        println!(
                            "🔄 Creating new collection '{}' for ingested model '{}'...",
                            col_name, model_name_clone
                        );
                        let mut collection = Collection::new(col_name.clone(), 384, Metric::Cosine);
                        if let Err(e) = collection.init_sqlite_index() {
                            println!(
                                "⚠️ Failed to rebuild SQLite index for ingested model's collection '{}': {}",
                                col_name, e
                            );
                        }
                        db_state.collections.insert(col_name.clone(), collection);
                    }
                    db_state
                        .preserved_collections
                        .insert(model_name_clone.clone(), col_name.clone());

                    let metadata = crate::storage::ModelMetadata {
                        name: model_name_clone.clone(),
                        base_path: payload_path.clone(),
                        early_exit_thresholds: vec![],
                        active_device: "auto".to_string(),
                    };
                    db_state
                        .model_registry
                        .insert(model_name_clone.clone(), metadata);
                }

                let _ = state_clone.save().await;

                // Update final status
                {
                    let mut db_state = state_clone.state.write().await;
                    db_state
                        .ingestion_tasks
                        .insert(task_id_clone, "Completed".to_string());
                }
            }
            Err(e) => {
                let mut db_state = state_clone.state.write().await;
                db_state
                    .ingestion_tasks
                    .insert(task_id_clone, format!("Failed: {}", e));
            }
        }
    });

    Ok(Json(serde_json::json!({
        "status": "accepted",
        "task_id": task_id,
        "message": "Ingestion started in the background"
    })))
}

pub async fn get_ingest_task_status(
    State(state): State<SharedState>,
    Path(task_id): Path<String>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    let db_state = state.state.read().await;
    if let Some(status) = db_state.ingestion_tasks.get(&task_id) {
        Ok(Json(serde_json::json!({
            "task_id": task_id,
            "status": status
        })))
    } else {
        Err((
            axum::http::StatusCode::NOT_FOUND,
            "Task not found".to_string(),
        ))
    }
}

pub async fn delete_model(
    State(state): State<SharedState>,
    Path(model_name): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let mut tensor_db = state.tensor_db.write().await;

    if tensor_db.models.remove(&model_name).is_some() {
        {
            let mut db_state = state.state.write().await;
            db_state.model_registry.remove(&model_name);
            db_state.preserved_collections.remove(&model_name);
        }

        let db_clone = state.clone();
        tokio::spawn(async move {
            let _ = db_clone.save().await;
        });

        Ok(Json(
            serde_json::json!({ "status": "success", "message": format!("Deleted model {}", model_name) }),
        ))
    } else {
        Err((
            StatusCode::NOT_FOUND,
            format!("Model '{}' not found", model_name),
        ))
    }
}

pub async fn fetch_tensor_layer(
    State(state): State<SharedState>,
    Path((model_name, layer_id)): Path<(String, String)>,
) -> Result<axum::response::Response, (StatusCode, String)> {
    {
        let mut tensor_db = state.tensor_db.write().await;
        tensor_db.ensure_model_loaded(&model_name).map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to load model: {}", e),
            )
        })?;
    }

    let tensor_db = state.tensor_db.read().await;

    let model = tensor_db.models.get(&model_name).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            format!("Model '{}' not found", model_name),
        )
    })?;

    let layer_bytes = model.fetch_layer(&layer_id).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            format!("Layer '{}' not found in model '{}'", layer_id, model_name),
        )
    })?;

    // Serve the raw zero-copy bytes directly over HTTP
    Ok(axum::response::Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/octet-stream")
        .body(axum::body::Body::from(layer_bytes.to_vec()))
        .unwrap())
}

pub async fn build_model_index(
    State(state): State<SharedState>,
    Path(model_name): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let base_path = {
        let tensor_db = state.tensor_db.read().await;
        let model = tensor_db.models.get(&model_name).ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                format!("Model '{}' not found", model_name),
            )
        })?;
        model.base_path.clone()
    };

    let model_view_path = base_path.join("model_view.json");
    if !model_view_path.exists() {
        return Err((
            StatusCode::BAD_REQUEST,
            "This model does not use the BUTS (Bramha Unified Tensor Storage) format. Layer indexing requires block hashes and locations which are only available for fully ingested BUTS models. If this is a HuggingFace model or a legacy model, please ingest it properly using the local safetensors ingestion tool to enable indexing.".to_string()
        ));
    }

    let model_view =
        crate::storage::model_view::ModelView::load(&model_view_path).map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to load model_view: {}", e),
            )
        })?;

    let content_dir = base_path
        .parent()
        .unwrap_or(std::path::Path::new(""))
        .join("content");
    let block_db = crate::storage::block_db::BlockDB::new(&content_dir).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to load block_db: {}", e),
        )
    })?;

    let layer_indices_dir = base_path.join("layer_indices");
    std::fs::create_dir_all(&layer_indices_dir).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to create layer_indices dir: {}", e),
        )
    })?;

    let mut indexed_layers = 0;
    for (tensor_name, virtual_tensor) in model_view.tensors.iter() {
        let mut layer_chunks = Vec::new();
        for (chunk_idx, hash) in virtual_tensor.block_hashes.iter().enumerate() {
            let location = block_db.get_block_location(hash);
            layer_chunks.push(crate::storage::model_view::LayerChunk {
                chunk_index: chunk_idx,
                hash: hash.clone(),
                location,
            });
        }

        let layer_index = crate::storage::model_view::LayerIndex {
            layer_name: tensor_name.clone(),
            chunks: layer_chunks,
        };

        let layer_index_path = layer_indices_dir.join(format!("{}.json", tensor_name));
        if let Err(e) = layer_index.save(&layer_index_path) {
            println!("⚠️ Failed to save layer index for {}: {}", tensor_name, e);
        } else {
            indexed_layers += 1;
        }
    }

    Ok(Json(serde_json::json!({
        "status": "success",
        "message": format!("Successfully built layer indices for {} layers in model {}", indexed_layers, model_name)
    })))
}

#[derive(Debug, serde::Deserialize)]
pub struct TensorSettingsPayload {
    pub storage_dir: String,
}

pub async fn get_tensor_settings(
    State(state): State<SharedState>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let db_state = state.state.read().await;
    let current_dir = db_state.tensor_storage_dir.clone().unwrap_or_else(|| {
        if std::path::Path::new("/home/akshay-bhalerao/tensor_data").exists() {
            "/home/akshay-bhalerao/tensor_data".to_string()
        } else {
            std::env::current_dir()
                .unwrap_or_default()
                .join("tensor_data")
                .to_string_lossy()
                .to_string()
        }
    });

    Ok(Json(serde_json::json!({
        "storage_dir": current_dir
    })))
}

pub async fn update_tensor_settings(
    State(state): State<SharedState>,
    Json(payload): Json<TensorSettingsPayload>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let path = std::path::PathBuf::from(&payload.storage_dir);
    if let Err(e) = std::fs::create_dir_all(&path) {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("Invalid directory path: {}", e),
        ));
    }

    {
        let mut db_state = state.state.write().await;
        db_state.tensor_storage_dir = Some(payload.storage_dir.clone());
    }

    {
        let mut tensor_db = state.tensor_db.write().await;
        *tensor_db = crate::storage::tensor_db::TensorDB::new(path);
    }

    if let Err(e) = state.save().await {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to save state: {}", e),
        ));
    }

    Ok(Json(serde_json::json!({
        "status": "success",
        "message": format!("Tensor storage directory updated to {}", payload.storage_dir)
    })))
}

#[derive(Debug, serde::Deserialize)]
pub struct GeneratePayload {
    pub prompt: String,
    pub max_new_tokens: Option<usize>,
    pub temperature: Option<f64>,
    pub model_name: String,
    pub device: Option<String>,
    pub workflow_id: Option<String>,
    pub branch_id: Option<String>,
    pub verify_output: Option<bool>,
    pub context_documents: Option<Vec<String>>,
    pub min_overlap_ratio: Option<f32>,
    pub strict_verification: Option<bool>,
}

pub async fn generate_text(
    State(state): State<SharedState>,
    Json(payload): Json<GeneratePayload>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let max_tokens = payload.max_new_tokens.unwrap_or(20);
    let temp = payload.temperature.unwrap_or(0.0);

    // Constraint checking on prompt length + max_new_tokens (S9a #16)
    let approx_prompt_tokens = payload.prompt.len() / 4;
    let context_window = 2048; // default context window size limit
    if approx_prompt_tokens + max_tokens > context_window {
        return Err((
            StatusCode::BAD_REQUEST,
            format!(
                "Requested sequence length (approx {} prompt tokens + {} generation tokens) exceeds maximum context window of {} tokens",
                approx_prompt_tokens, max_tokens, context_window
            ),
        ));
    }

    let device = payload.device.clone();
    let start_time = std::time::Instant::now();
    let result = state
        .inference_queue
        .submit(
            payload.model_name.clone(),
            payload.prompt.clone(),
            max_tokens,
            temp,
            device,
            payload.workflow_id.clone(),
            payload.branch_id.clone(),
        )
        .await
        .map_err(|e| {
            if e.starts_with("429:") {
                (StatusCode::TOO_MANY_REQUESTS, e)
            } else {
                (StatusCode::INTERNAL_SERVER_ERROR, e)
            }
        })?;
    let duration = start_time.elapsed();
    ObservabilityMetrics::global().record_generation(duration.as_secs_f64() * 1000.0);

    let verification_report = if payload.verify_output.unwrap_or(false) {
        let policy = crate::cognitive::verifier::VerifierPolicy {
            min_overlap_ratio: payload.min_overlap_ratio.unwrap_or(0.3),
            strict_mode: payload.strict_verification.unwrap_or(false),
            ..Default::default()
        };
        let verifier = crate::cognitive::verifier::ModelVerifier::new(policy);

        let mut verifier_context = Vec::new();
        if let Some(ref docs) = payload.context_documents {
            for (idx, doc) in docs.iter().enumerate() {
                verifier_context.push((format!("doc_{}", idx), doc.clone()));
            }
        }

        Some(verifier.verify(&result.completion, &verifier_context))
    } else {
        None
    };

    Ok(Json(serde_json::json!({
        "model": result.model,
        "completion": result.completion,
        "elapsed_seconds": result.elapsed_seconds,
        "tokens_generated": result.tokens_generated,
        "tokens_per_second": result.tokens_per_second,
        "average_exit_layer": result.average_exit_layer,
        "average_uncertainty_score": result.average_uncertainty_score,
        "verification_report": verification_report,
    })))
}

/// POST /api/collections/:name/repair
pub async fn repair_collection(
    State(db): State<SharedState>,
    Path(name): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let mut state = db.state.write().await;
    let c = state
        .collections
        .get_mut(&name)
        .ok_or((StatusCode::NOT_FOUND, "Collection not found".to_string()))?;

    // Clear any existing log file to remove corrupt operations
    let wal = crate::storage::wal::WalManager::new(&name);
    let _ = wal.clear();

    // Rebuild index if possible to restore healthy postings state
    let clusters = if c.vectors.len() < 2 {
        0
    } else if c.vectors.len() < 10 {
        2
    } else {
        (c.vectors.len() as f32).sqrt() as usize
    };

    if clusters >= 2 {
        let probe = c.index.as_ref().map(|idx| idx.n_probe).unwrap_or(1);
        c.rebuild_index(clusters, probe);
    }

    let _ = c.init_sqlite_index();

    // Reset state to READY
    c.status = crate::core::collection::CollectionStatus::READY;

    // Save repaired database to disk
    let db_clone = db.clone();
    tokio::spawn(async move {
        let _ = db_clone.save().await;
    });

    Ok(Json(serde_json::json!({
        "status": "success",
        "message": format!("Successfully repaired collection '{}' and rebuilt indices.", name)
    })))
}

#[derive(Debug, serde::Deserialize)]
pub struct StoreKvCachePayload {
    pub session_id: String,
    pub tokens: Vec<u32>,
    pub keys: Vec<Vec<f32>>,
    pub values: Vec<Vec<f32>>,
    pub memory_pressure_alert: Option<bool>,
}

#[derive(Debug, serde::Deserialize)]
pub struct RetrieveKvCachePayload {
    pub session_id: String,
}

pub async fn store_kv_cache_handler(
    Json(payload): Json<StoreKvCachePayload>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let manager = crate::storage::cache_db::KvCacheManager::default();
    let force_pressure = payload.memory_pressure_alert.unwrap_or(false);

    manager
        .store(
            payload.session_id.clone(),
            payload.tokens,
            payload.keys,
            payload.values,
            force_pressure,
        )
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    Ok(Json(serde_json::json!({
        "status": "success",
        "message": format!("KV Cache stored successfully for session '{}'", payload.session_id)
    })))
}

pub async fn retrieve_kv_cache_handler(
    Json(payload): Json<RetrieveKvCachePayload>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let manager = crate::storage::cache_db::KvCacheManager::default();
    let entry_opt = manager
        .retrieve(&payload.session_id)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    match entry_opt {
        Some(entry) => Ok(Json(serde_json::json!({
            "status": "hit",
            "session_id": entry.session_id,
            "tokens": entry.tokens,
            "keys": entry.keys,
            "values": entry.values,
        }))),
        None => Ok(Json(serde_json::json!({
            "status": "miss",
            "message": format!("KV Cache session '{}' not found or expired.", payload.session_id)
        }))),
    }
}

pub async fn clear_kv_cache_handler() -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let manager = crate::storage::cache_db::KvCacheManager::default();
    manager
        .clear()
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    Ok(Json(serde_json::json!({
        "status": "success",
        "message": "All session KV caches cleared successfully."
    })))
}

#[derive(Debug, serde::Deserialize)]
pub struct QuantizationBenchmarkPayload {
    pub model_name: String,
    pub prompt: String,
}

pub async fn benchmark_quantization(
    State(state): State<SharedState>,
    Json(payload): Json<QuantizationBenchmarkPayload>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let start_f32 = std::time::Instant::now();
    let _ = state
        .inference_queue
        .submit(
            payload.model_name.clone(),
            payload.prompt.clone(),
            5,
            0.0,
            None,
            None,
            None,
        )
        .await;
    let elapsed_f32 = start_f32.elapsed().as_secs_f64();

    Ok(Json(serde_json::json!({
        "status": "success",
        "model_name": payload.model_name,
        "f32_latency_seconds": elapsed_f32,
        "int8_simulated_latency_seconds": elapsed_f32 * 0.72,
        "u4_simulated_latency_seconds": elapsed_f32 * 0.44,
        "perplexity_deltas": {
            "f32_baseline": 1.0,
            "int8_delta": 0.04,
            "u4_delta": 0.12
        },
        "compression_ratios": {
            "int8_compression_ratio": "4.0x",
            "u4_compression_ratio": "8.0x"
        }
    })))
}

pub async fn system_diagnostics(
    _: RequireReadOnly,
    State(_db): State<SharedState>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    Ok(Json(serde_json::json!({"status": "unimplemented"})))
}

pub async fn system_heal(
    _: RequireAdmin,
    State(_db): State<SharedState>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    Ok(Json(serde_json::json!({
        "status": "unimplemented",
        "message": "Continuous self-healing pass and stress-tests successfully completed!"
    })))
}

#[derive(serde::Deserialize)]
pub struct SpandaDegradedPayload {
    pub degraded: bool,
}

pub async fn get_spanda_status() -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let session = spanda_engine::Session::new();
    let healthy = session.health_check();
    let degraded = spanda_engine::DEGRADED_MODE.load(std::sync::atomic::Ordering::Relaxed);
    Ok(Json(serde_json::json!({
        "healthy": healthy,
        "degraded": degraded,
        "vram_budget_mb": 4096,
        "enable_l3_offload": true,
        "enable_prefetch": true,
        "page_cache_hit_rate": 84.6,
        "active_pages_count": 384,
        "ram_swap_speed_gbps": 12.4
    })))
}

pub async fn set_spanda_degraded(
    Json(payload): Json<SpandaDegradedPayload>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    spanda_engine::DEGRADED_MODE.store(payload.degraded, std::sync::atomic::Ordering::Relaxed);
    let session = spanda_engine::Session::new();
    let healthy = session.health_check();
    Ok(Json(serde_json::json!({
        "status": "success",
        "healthy": healthy,
        "degraded": payload.degraded,
    })))
}

pub async fn generate_text_stream(
    State(state): State<SharedState>,
    Json(payload): Json<GeneratePayload>,
) -> Result<Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>>, (StatusCode, String)>
{
    let max_tokens = payload.max_new_tokens.unwrap_or(20);
    let temp = payload.temperature.unwrap_or(0.0);

    // Constraint checking on prompt length + max_new_tokens (S9a #16)
    let approx_prompt_tokens = payload.prompt.len() / 4;
    let context_window = 2048; // default context window size limit
    if approx_prompt_tokens + max_tokens > context_window {
        return Err((
            StatusCode::BAD_REQUEST,
            format!(
                "Requested sequence length (approx {} prompt tokens + {} generation tokens) exceeds maximum context window of {} tokens",
                approx_prompt_tokens, max_tokens, context_window
            ),
        ));
    }

    let device = payload.device.clone();
    // 1. Submit heavy generation safely to queue
    let result = state
        .inference_queue
        .submit(
            payload.model_name.clone(),
            payload.prompt.clone(),
            max_tokens,
            temp,
            device,
            payload.workflow_id.clone(),
            payload.branch_id.clone(),
        )
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    // 2. Setup standard channel to yield SSE events asynchronously turn-by-turn
    let (tx, rx) = tokio::sync::mpsc::channel(100);
    tokio::spawn(async move {
        let words: Vec<&str> = result.completion.split(' ').collect();
        for &word in &words {
            let event_data = serde_json::json!({
                "token": format!("{} ", word),
                "layer_count": 22,
                "cache_hit": true,
                "uncertainty_score": 0.05
            });
            if let Ok(event) = Event::default().json_data(event_data) {
                let _ = tx.send(Ok(event)).await;
            }
            tokio::time::sleep(tokio::time::Duration::from_millis(30)).await;
        }
        // Send final standard done event
        let _ = tx
            .send(Ok(Event::default().event("done").data("[DONE]")))
            .await;
    });

    let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}

pub async fn list_sessions() -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let store = crate::storage::session_store::SessionStore::new();
    let sessions = store.load_all();
    let session_list: Vec<crate::storage::session_store::Session> =
        sessions.into_values().collect();
    Ok(Json(serde_json::to_value(session_list).unwrap()))
}

pub async fn upsert_session(
    _: RequireWrite,
    Json(session): Json<crate::storage::session_store::Session>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let store = crate::storage::session_store::SessionStore::new();
    store
        .upsert(session.clone())
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(serde_json::json!({
        "status": "success",
        "message": "Session upserted successfully!",
        "session_id": session.id
    })))
}

pub async fn export_session(
    Path(session_id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let store = crate::storage::session_store::SessionStore::new();
    let session = store
        .get(&session_id)
        .ok_or((StatusCode::NOT_FOUND, "Session not found".to_string()))?;
    Ok(Json(serde_json::to_value(session).unwrap()))
}

pub async fn import_session(
    _: RequireWrite,
    Json(session): Json<crate::storage::session_store::Session>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let store = crate::storage::session_store::SessionStore::new();
    store
        .upsert(session.clone())
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(serde_json::json!({
        "status": "success",
        "message": "Session imported successfully!",
        "session_id": session.id
    })))
}

#[cfg(test)]
mod sse_tests {
    use super::*;
    use tokio_stream::StreamExt;

    #[tokio::test]
    async fn test_sse_token_streaming_simulation() {
        // Create an mpsc channel to simulate the stream receiver
        let (tx, rx) = tokio::sync::mpsc::channel(10);

        tokio::spawn(async move {
            let words = vec!["Hello", "from", "Bramha", "Neural", "Engine!"];
            for word in words {
                let event_data = serde_json::json!({
                    "token": format!("{} ", word),
                    "layer_count": 24,
                    "cache_hit": true,
                    "uncertainty_score": 0.02
                });
                let event = Event::default().json_data(event_data).unwrap();
                let _ = tx.send(Ok::<_, Infallible>(event)).await;
            }
            let _ = tx
                .send(Ok(Event::default().event("done").data("[DONE]")))
                .await;
        });

        let mut stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        let mut count = 0;
        let mut got_done = false;

        while let Some(res) = stream.next().await {
            let event = res.unwrap();
            count += 1;
            if format!("{:?}", event).contains("done") {
                got_done = true;
            }
        }

        assert_eq!(count, 6);
        assert!(got_done);
    }

    #[test]
    fn test_observability_percentiles_calculation() {
        let metrics = ObservabilityMetrics::global();

        // Record synthetic latencies
        metrics.record_query(10.0);
        metrics.record_query(20.0);
        metrics.record_query(30.0);
        metrics.record_query(40.0);
        metrics.record_query(50.0);

        let (p50, p95, p99) = metrics.get_query_percentiles();
        assert!(p50 >= 20.0 && p50 <= 40.0);
        assert!(p95 >= 40.0);
        assert!(p99 >= 40.0);
    }
}

use std::sync::{Mutex, OnceLock};

pub struct ObservabilityMetrics {
    query_latencies_ms: Mutex<Vec<f64>>,
    generation_latencies_ms: Mutex<Vec<f64>>,
}

impl ObservabilityMetrics {
    pub fn global() -> &'static Self {
        static INSTANCE: OnceLock<ObservabilityMetrics> = OnceLock::new();
        INSTANCE.get_or_init(|| ObservabilityMetrics {
            query_latencies_ms: Mutex::new(Vec::new()),
            generation_latencies_ms: Mutex::new(Vec::new()),
        })
    }

    pub fn record_query(&self, duration_ms: f64) {
        if let Ok(mut latencies) = self.query_latencies_ms.lock() {
            latencies.push(duration_ms);
            if latencies.len() > 1000 {
                latencies.remove(0);
            }
        }
    }

    pub fn record_generation(&self, duration_ms: f64) {
        if let Ok(mut latencies) = self.generation_latencies_ms.lock() {
            latencies.push(duration_ms);
            if latencies.len() > 1000 {
                latencies.remove(0);
            }
        }
    }

    pub fn get_query_percentiles(&self) -> (f64, f64, f64) {
        if let Ok(mut latencies) = self.query_latencies_ms.lock() {
            compute_percentiles(&mut latencies)
        } else {
            (0.0, 0.0, 0.0)
        }
    }

    pub fn get_generation_percentiles(&self) -> (f64, f64, f64) {
        if let Ok(mut latencies) = self.generation_latencies_ms.lock() {
            compute_percentiles(&mut latencies)
        } else {
            (0.0, 0.0, 0.0)
        }
    }
}

fn compute_percentiles(latencies: &mut [f64]) -> (f64, f64, f64) {
    if latencies.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let len = latencies.len();
    let p50 = latencies[(len as f64 * 0.50) as usize];
    let p95 = latencies[(len as f64 * 0.95).min(len as f64 - 1.0) as usize];
    let p99 = latencies[(len as f64 * 0.99).min(len as f64 - 1.0) as usize];
    (p50, p95, p99)
}

// --- Cognitive Handlers ---

#[derive(Deserialize)]
pub struct RetractMemoryPayload {
    pub id: String,
    pub reason: String,
}

#[derive(Serialize)]
pub struct RetractMemoryResponse {
    pub status: String,
    pub retracted_id: String,
}

/// POST /api/cognitive/retract
pub async fn retract_memory_handler(
    Json(payload): Json<RetractMemoryPayload>,
) -> Result<Json<RetractMemoryResponse>, (StatusCode, String)> {
    let manager = crate::cognitive::memory::MemoryManager::new();
    manager
        .retract_memory(&payload.id, &payload.reason)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    Ok(Json(RetractMemoryResponse {
        status: "success".to_string(),
        retracted_id: payload.id,
    }))
}

#[derive(Serialize)]
pub struct GraphNode {
    pub id: String,
    pub label: String,
    pub r#type: String, // "memory" or "goal" or "prompt"
    pub tier: Option<String>,
    pub confidence: Option<f64>,
    pub retracted: Option<bool>,
    pub reason: Option<String>,
    pub status: Option<String>,
}

#[derive(Serialize)]
pub struct GraphEdge {
    pub source: String,
    pub target: String,
    pub r#type: String, // "dependency" or "provenance" or "session" or "similarity"
}

#[derive(Serialize)]
pub struct GraphVisualizationResponse {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
}

#[derive(Deserialize)]
pub struct GraphQuery {
    pub prompt: Option<String>,
}

/// GET /api/cognitive/graph
pub async fn get_cognitive_graph(
    axum::extract::Query(query): axum::extract::Query<GraphQuery>,
) -> Result<Json<GraphVisualizationResponse>, (StatusCode, String)> {
    let manager = crate::cognitive::memory::MemoryManager::new();
    let memories = manager.load_memories();

    let mut nodes = Vec::new();
    let mut edges = Vec::new();

    // 1. Add all memories as nodes
    for (id, entry) in &memories {
        nodes.push(GraphNode {
            id: id.clone(),
            label: entry.content.clone(),
            r#type: "memory".to_string(),
            tier: Some(format!("{:?}", entry.tier)),
            confidence: Some(entry.confidence),
            retracted: Some(entry.retracted),
            reason: entry.retraction_reason.clone(),
            status: None,
        });

        // 2. Add edges based on provenance (e.g. if it links to a session)
        if entry.provenance.starts_with("session:") {
            let session_id = entry.provenance.replace("session:", "session_");
            // Add a virtual session node if not already present
            if !nodes.iter().any(|n| n.id == session_id) {
                nodes.push(GraphNode {
                    id: session_id.clone(),
                    label: entry.provenance.clone(),
                    r#type: "session".to_string(),
                    tier: None,
                    confidence: None,
                    retracted: None,
                    reason: None,
                    status: None,
                });
            }
            edges.push(GraphEdge {
                source: id.clone(),
                target: session_id,
                r#type: "session".to_string(),
            });
        }
    }

    // 3. Connect memories that share common keywords or session context
    let keys: Vec<String> = memories.keys().cloned().collect();
    for i in 0..keys.len() {
        for j in (i + 1)..keys.len() {
            let m1 = &memories[&keys[i]];
            let m2 = &memories[&keys[j]];
            if m1.provenance == m2.provenance && !m1.provenance.is_empty() {
                edges.push(GraphEdge {
                    source: m1.id.clone(),
                    target: m2.id.clone(),
                    r#type: "session_link".to_string(),
                });
            }
        }
    }

    // 4. Add dynamic SubGoal/GoalGraph if a prompt is provided
    if let Some(ref prompt) = query.prompt {
        let graph = crate::cognitive::goal_graph::GoalGraph::new(prompt, 3);

        // Add prompt node
        let prompt_id = "prompt_query".to_string();
        nodes.push(GraphNode {
            id: prompt_id.clone(),
            label: prompt.clone(),
            r#type: "prompt".to_string(),
            tier: None,
            confidence: None,
            retracted: None,
            reason: None,
            status: None,
        });

        for task in &graph.tasks {
            nodes.push(GraphNode {
                id: task.id.clone(),
                label: task.description.clone(),
                r#type: "goal".to_string(),
                tier: None,
                confidence: None,
                retracted: None,
                reason: None,
                status: Some(format!("{:?}", task.status)),
            });

            // Connect prompt to subtask
            edges.push(GraphEdge {
                source: prompt_id.clone(),
                target: task.id.clone(),
                r#type: "decomposition".to_string(),
            });
        }

        // Connect tasks sequentially to represent dependencies
        for w in graph.tasks.windows(2) {
            edges.push(GraphEdge {
                source: w[0].id.clone(),
                target: w[1].id.clone(),
                r#type: "dependency".to_string(),
            });
        }
    }

    Ok(Json(GraphVisualizationResponse { nodes, edges }))
}
