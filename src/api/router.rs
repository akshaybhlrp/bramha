use axum::{
    Router,
    routing::{get, post},
};
use tower_http::cors::{Any, CorsLayer};
use tower_http::services::ServeDir;

use crate::api::handlers::*;

/// Configures the Axum REST API router with thread-safe database state and CORS policies.
pub fn create_router(state: SharedState) -> Router {
    let allowed_origins = std::env::var("BRAMHA_CORS_ALLOWED_ORIGINS").ok();
    let cors = if let Some(origins_str) = allowed_origins {
        let mut origins = Vec::new();
        for origin in origins_str.split(',') {
            if let Ok(val) = origin.trim().parse::<axum::http::HeaderValue>() {
                origins.push(val);
            }
        }
        CorsLayer::new()
            .allow_origin(origins)
            .allow_methods(Any)
            .allow_headers(Any)
    } else {
        let is_prod = std::env::var("RUST_ENV").unwrap_or_default() == "production"
            || std::env::var("BRAMHA_ENV").unwrap_or_default() == "production";
        if is_prod {
            CorsLayer::new().allow_methods(Any).allow_headers(Any)
        } else {
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any)
        }
    };

    let protected_routes = Router::new()
        // System Statistics
        .route("/api/stats", get(get_stats))
        // Collections CRUD & Vector operations
        .route(
            "/api/collections",
            get(list_collections).post(create_collection),
        )
        .route(
            "/api/collections/:name",
            get(get_collection).delete(delete_collection),
        )
        .route("/api/collections/:name/upsert", post(upsert_vectors))
        .route("/api/collections/:name/query", post(query_collection))
        .route("/api/collections/:name/delete", post(delete_vectors))
        .route("/api/collections/:name/reindex", post(reindex_collection))
        .route("/api/collections/:name/repair", post(repair_collection))
        // Semantic prompt caching endpoints
        .route("/api/cache/check", post(check_cache))
        .route("/api/cache/store", post(store_cache))
        .route("/api/cache/clear", post(clear_cache))
        // KV Cache session offloading endpoints
        .route("/api/sessions/kv/store", post(store_kv_cache_handler))
        .route("/api/sessions/kv/retrieve", post(retrieve_kv_cache_handler))
        .route("/api/sessions/kv/clear", post(clear_kv_cache_handler))
        // AirLLM sidecar proxy endpoints
        .route("/api/llm/embed", post(llm_embed))
        .route("/api/llm/embed_batch", post(llm_embed_batch))
        .route("/api/llm/rag", post(llm_rag))
        .route("/api/llm/logs", get(llm_logs))
        .route("/api/llm/health", get(llm_health))
        .route("/api/llm/generate", post(generate_text))
        .route("/api/llm/generate/stream", post(generate_text_stream))
        .route("/api/llm/load_model", post(llm_load_model))
        .route(
            "/api/llm/quantization/benchmark",
            post(benchmark_quantization),
        )
        .route("/api/llm/hardware", get(llm_hardware))
        // Tensor DB Endpoints
        .route("/api/tensor/models", get(list_models))
        .route(
            "/api/tensor/models/:model_name",
            get(get_model_layers)
                .post(ingest_model)
                .delete(delete_model),
        )
        .route(
            "/api/tensor/models/:model_name/build_index",
            post(build_model_index),
        )
        .route("/api/tensor/tasks/:task_id", get(get_ingest_task_status))
        .route("/api/tensor/:model_name/:layer_id", get(fetch_tensor_layer))
        .route(
            "/api/tensor/settings",
            get(get_tensor_settings).post(update_tensor_settings),
        )
        // Cognitive Self-Healing & Diagnostic routes
        .route("/api/system/diagnostics", get(system_diagnostics))
        .route("/api/system/heal", post(system_heal))
        .route("/api/system/spanda/status", get(get_spanda_status))
        .route("/api/system/spanda/degraded", post(set_spanda_degraded))
        .route("/api/cognitive/retract", post(retract_memory_handler))
        .route("/api/cognitive/graph", get(get_cognitive_graph))
        .route("/api/cognitive/multi_hop", post(multi_hop_handler))
        // Conversational Session history routes
        .route("/api/sessions", get(list_sessions).post(upsert_session))
        .route("/api/sessions/:session_id/export", get(export_session))
        .route("/api/sessions/import", post(import_session))
        .layer(axum::middleware::from_fn(
            crate::middleware::auth::require_read_only_middleware,
        ));

    let public_routes = Router::new()
        .route("/health", get(health_handler))
        .route("/ready", get(ready_handler))
        .route("/metrics", get(metrics_handler))
        .fallback_service(ServeDir::new("dashboard"));

    protected_routes
        .merge(public_routes)
        .layer(cors)
        .layer(axum::middleware::from_fn(add_security_headers))
        .with_state(state)
}

async fn health_handler() -> axum::Json<serde_json::Value> {
    axum::Json(serde_json::json!({ "status": "healthy" }))
}

async fn ready_handler() -> axum::Json<serde_json::Value> {
    axum::Json(serde_json::json!({ "status": "ready" }))
}

async fn metrics_handler() -> axum::Json<serde_json::Value> {
    let metrics = crate::api::handlers::ObservabilityMetrics::global();
    let (q50, q95, q99) = metrics.get_query_percentiles();
    let (g50, g95, g99) = metrics.get_generation_percentiles();
    axum::Json(serde_json::json!({
        "query_latency_p50_ms": q50,
        "query_latency_p95_ms": q95,
        "query_latency_p99_ms": q99,
        "generation_latency_p50_ms": g50,
        "generation_latency_p95_ms": g95,
        "generation_latency_p99_ms": g99,
    }))
}

async fn add_security_headers(
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    let mut response = next.run(request).await;
    let headers = response.headers_mut();
    headers.insert(
        axum::http::header::CONTENT_SECURITY_POLICY,
        axum::http::HeaderValue::from_static("default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; frame-ancestors 'none';"),
    );
    headers.insert(
        axum::http::header::X_FRAME_OPTIONS,
        axum::http::HeaderValue::from_static("DENY"),
    );
    headers.insert(
        axum::http::header::X_CONTENT_TYPE_OPTIONS,
        axum::http::HeaderValue::from_static("nosniff"),
    );
    response
}
