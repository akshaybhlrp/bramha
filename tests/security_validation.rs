use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use bramha::api::create_router;
use bramha::storage::Database;
use serde_json::json;
use std::sync::Arc;
use tower::ServiceExt;

#[tokio::test]
async fn test_auth_defaults_unauthorized() {
    let db_path = "storage/test_security_unauth.db";
    let _ = std::fs::remove_file(db_path);

    let db = Database::new(Some(db_path.to_string()), 1536);
    let shared_db = Arc::new(db);
    let app = create_router(shared_db);

    let req = Request::builder()
        .uri("/api/stats")
        .method("GET")
        .body(Body::empty())
        .unwrap();

    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::UNAUTHORIZED);

    let _ = std::fs::remove_file(db_path);
}

#[tokio::test]
async fn test_auth_defaults_authorized() {
    let db_path = "storage/test_security_auth.db";
    let _ = std::fs::remove_file(db_path);

    let db = Database::new(Some(db_path.to_string()), 1536);
    let shared_db = Arc::new(db);
    let app = create_router(shared_db);

    let req = Request::builder()
        .uri("/api/stats")
        .method("GET")
        .header("Authorization", "Bearer read_key")
        .body(Body::empty())
        .unwrap();

    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    let _ = std::fs::remove_file(db_path);
}

#[tokio::test]
async fn test_auth_insufficient_privileges() {
    let db_path = "storage/test_security_insufficient.db";
    let _ = std::fs::remove_file(db_path);

    let db = Database::new(Some(db_path.to_string()), 1536);
    let shared_db = Arc::new(db);
    let app = create_router(shared_db);

    let payload = json!({
        "name": "test_collection",
        "dimension": 128,
        "metric": "Cosine"
    });

    let req = Request::builder()
        .uri("/api/collections")
        .method("POST")
        .header("Authorization", "Bearer read_key")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();

    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::FORBIDDEN);

    let _ = std::fs::remove_file(db_path);
}

#[tokio::test]
async fn test_public_routes() {
    let db_path = "storage/test_security_public.db";
    let _ = std::fs::remove_file(db_path);

    let db = Database::new(Some(db_path.to_string()), 1536);
    let shared_db = Arc::new(db);
    let app = create_router(shared_db);

    let public_paths = vec!["/health", "/ready", "/metrics"];
    for path in public_paths {
        let req = Request::builder()
            .uri(path)
            .method("GET")
            .body(Body::empty())
            .unwrap();

        let res = app.clone().oneshot(req).await.unwrap();
        assert_eq!(res.status(), StatusCode::OK);
    }

    let _ = std::fs::remove_file(db_path);
}

#[tokio::test]
async fn test_ingest_model_path_traversal() {
    let db_path = "storage/test_security_traversal.db";
    let _ = std::fs::remove_file(db_path);

    let db = Database::new(Some(db_path.to_string()), 1536);
    let shared_db = Arc::new(db);
    let app = create_router(shared_db);

    // Path traversal payload
    let payloads = vec![
        json!({ "path": "../../../etc/passwd" }),
        json!({ "path": "/etc/passwd" }),
        json!({ "path": "storage/../../etc" }),
    ];

    for payload in payloads {
        let req = Request::builder()
            .uri("/api/tensor/models/test_model")
            .method("POST")
            .header("Authorization", "Bearer write_key")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&payload).unwrap()))
            .unwrap();

        let res = app.clone().oneshot(req).await.unwrap();
        assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    }

    let _ = std::fs::remove_file(db_path);
}

#[tokio::test]
async fn test_generate_text_bounds() {
    let db_path = "storage/test_security_bounds.db";
    let _ = std::fs::remove_file(db_path);

    let db = Database::new(Some(db_path.to_string()), 1536);
    let shared_db = Arc::new(db);
    let app = create_router(shared_db);

    let payload = json!({
        "model_name": "tinyllama",
        "prompt": "a".repeat(5000),
        "max_new_tokens": 5000
    });

    let req = Request::builder()
        .uri("/api/llm/generate")
        .method("POST")
        .header("Authorization", "Bearer read_key")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();

    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);

    let _ = std::fs::remove_file(db_path);
}

#[tokio::test]
async fn test_ingest_model_symlink_rejection() {
    let db_path = "storage/test_security_symlink.db";
    let _ = std::fs::remove_file(db_path);

    let db = Database::new(Some(db_path.to_string()), 1536);
    let shared_db = Arc::new(db);
    let app = create_router(shared_db);

    let target_dir = "storage/symlink_target";
    let symlink_path = "storage/symlink_link";
    let _ = std::fs::create_dir_all(target_dir);
    let _ = std::fs::remove_file(symlink_path);

    #[cfg(unix)]
    {
        if std::os::unix::fs::symlink(target_dir, symlink_path).is_ok() {
            let payload = json!({ "path": symlink_path });
            let req = Request::builder()
                .uri("/api/tensor/models/test_model")
                .method("POST")
                .header("Authorization", "Bearer write_key")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&payload).unwrap()))
                .unwrap();

            let res = app.oneshot(req).await.unwrap();
            assert_eq!(res.status(), StatusCode::BAD_REQUEST);
        }
        let _ = std::fs::remove_file(symlink_path);
    }
    let _ = std::fs::remove_dir_all(target_dir);
    let _ = std::fs::remove_file(db_path);
}

#[tokio::test]
async fn test_llm_load_model_authorization() {
    let db_path = "storage/test_security_load_model.db";
    let _ = std::fs::remove_file(db_path);

    let db = Database::new(Some(db_path.to_string()), 1536);
    let shared_db = Arc::new(db);
    let app = create_router(shared_db);

    let payload = json!({
        "model_name": "tinyllama",
        "device": "cpu"
    });

    // 1. Unauthenticated -> 401
    let req = Request::builder()
        .uri("/api/llm/load_model")
        .method("POST")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();

    let res = app.clone().oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::UNAUTHORIZED);

    // 2. ReadOnly -> 403
    let req = Request::builder()
        .uri("/api/llm/load_model")
        .method("POST")
        .header("Authorization", "Bearer read_key")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();

    let res = app.clone().oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::FORBIDDEN);

    // 3. Admin -> Not 401 or 403 (could be 400 Bad Request or 200 depending on actual file existence)
    let req = Request::builder()
        .uri("/api/llm/load_model")
        .method("POST")
        .header("Authorization", "Bearer admin_key")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();

    let res = app.oneshot(req).await.unwrap();
    assert!(res.status() != StatusCode::UNAUTHORIZED && res.status() != StatusCode::FORBIDDEN);

    let _ = std::fs::remove_file(db_path);
}
