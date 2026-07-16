use bramha::api::create_router;
use bramha::storage::Database;
use http_body_util::BodyExt;
use std::sync::Arc;
use tower::ServiceExt; // for oneshot/call

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_cognitive_retract_and_graph_endpoints() {
        let db_path = "storage/test_cognitive_api.db";
        let _ = std::fs::remove_file(db_path);

        // 1. Initialize DB and create router
        let db = Database::new(Some(db_path.to_string()), 384);
        let shared_db = Arc::new(db);
        let app = create_router(shared_db);

        // 2. Setup a test memory file
        let memory_manager = bramha::cognitive::memory::MemoryManager::new();
        let memory_file = std::path::Path::new("storage/cognitive_memory.json");
        let _ = std::fs::remove_file(memory_file);

        let entry = bramha::cognitive::memory::MemoryEntry {
            id: "mem_api_test".to_string(),
            content: "Bramha API supports memory retraction".to_string(),
            tier: bramha::cognitive::memory::MemoryTier::Semantic,
            confidence: 0.95,
            usage_count: 1,
            last_accessed_ms: 100000,
            created_at_ms: 100000,
            provenance: "session:test_sess".to_string(),
            retracted: false,
            retraction_reason: None,
        };
        memory_manager.insert_memory(entry).unwrap();

        // 3. Test GET /api/cognitive/graph (without prompt)
        let req = axum::http::Request::builder()
            .uri("/api/cognitive/graph")
            .method("GET")
            .header("Authorization", "Bearer read_key")
            .body(axum::body::Body::empty())
            .unwrap();

        let response = app.clone().oneshot(req).await.unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::OK);

        let body = response.into_body().collect().await.unwrap().to_bytes();
        let body_str = String::from_utf8(body.to_vec()).unwrap();
        assert!(body_str.contains("mem_api_test"));
        assert!(body_str.contains("Bramha API supports memory retraction"));
        assert!(body_str.contains("session_test_sess"));

        // 4. Test GET /api/cognitive/graph?prompt=compare
        let req = axum::http::Request::builder()
            .uri("/api/cognitive/graph?prompt=compare+entities")
            .method("GET")
            .header("Authorization", "Bearer read_key")
            .body(axum::body::Body::empty())
            .unwrap();

        let response = app.clone().oneshot(req).await.unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::OK);

        let body = response.into_body().collect().await.unwrap().to_bytes();
        let body_str = String::from_utf8(body.to_vec()).unwrap();
        assert!(body_str.contains("prompt_query"));
        assert!(body_str.contains("subtask_1"));

        // 5. Test POST /api/cognitive/retract
        let payload = serde_json::json!({
            "id": "mem_api_test",
            "reason": "Test retraction api"
        });
        let req = axum::http::Request::builder()
            .uri("/api/cognitive/retract")
            .method("POST")
            .header("Authorization", "Bearer write_key")
            .header("Content-Type", "application/json")
            .body(axum::body::Body::from(
                serde_json::to_vec(&payload).unwrap(),
            ))
            .unwrap();

        let response = app.oneshot(req).await.unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::OK);

        let body = response.into_body().collect().await.unwrap().to_bytes();
        let body_str = String::from_utf8(body.to_vec()).unwrap();
        assert!(body_str.contains("success"));
        assert!(body_str.contains("mem_api_test"));

        // Verify it was indeed retracted in memory store
        let memories = memory_manager.load_memories();
        let entry = memories.get("mem_api_test").unwrap();
        assert!(entry.retracted);
        assert_eq!(
            entry.retraction_reason.as_deref(),
            Some("Test retraction api")
        );

        let _ = std::fs::remove_file(memory_file);
        let _ = std::fs::remove_file(db_path);
    }
}
