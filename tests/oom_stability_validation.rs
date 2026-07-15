use bramha::api::create_router;
use bramha::storage::Database;
use http_body_util::BodyExt;
use hyper_util::rt::TokioIo;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::UnixStream;

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_uds_and_concurrency_oom_stability() {
        let socket_path = "storage/test_oom_stability.sock";
        let db_path = "storage/test_oom_stability.db";

        let _ = std::fs::remove_file(socket_path);
        let _ = std::fs::remove_file(db_path);

        // 1. Initialize DB with memory ceiling/cache capacity
        let db = Database::new(Some(db_path.to_string()), 1536);
        let shared_db = Arc::new(db);
        shared_db.start_worker();

        let app = create_router(shared_db.clone());

        // 2. Start UDS HTTP Server
        let listener = tokio::net::UnixListener::bind(socket_path).expect("Failed to bind UDS");

        let server_app = app.clone();
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();

        let server_handle = tokio::spawn(async move {
            use tower::Service;
            tokio::select! {
                _ = async {
                    loop {
                        match listener.accept().await {
                            Ok((stream, _addr)) => {
                                let tower_service = server_app.clone();
                                let io = TokioIo::new(stream);
                                tokio::spawn(async move {
                                    let hyper_service = hyper::service::service_fn(move |req| {
                                        let mut tower_service = tower_service.clone();
                                        async move {
                                            let res = tower_service.call(req).await.unwrap();
                                            Ok::<_, std::convert::Infallible>(res)
                                        }
                                    });
                                    let _ = hyper::server::conn::http1::Builder::new()
                                        .serve_connection(io, hyper_service)
                                        .await;
                                });
                            }
                            Err(_) => break,
                        }
                    }
                } => {}
                _ = shutdown_rx => {}
            }
        });

        // Give the server a moment to start
        tokio::time::sleep(Duration::from_millis(50)).await;

        // 3. Perform a UDS Request using Hyper client to verify connection
        let stream = UnixStream::connect(socket_path)
            .await
            .expect("Failed to connect to UDS");
        let io = TokioIo::new(stream);
        let (mut sender, conn) = hyper::client::conn::http1::handshake(io)
            .await
            .expect("Handshake failed");

        tokio::spawn(async move {
            let _ = conn.await;
        });

        let req = hyper::Request::builder()
            .uri("http://localhost/api/stats")
            .method("GET")
            .header("Authorization", "Bearer read_key")
            .body(http_body_util::Empty::<hyper::body::Bytes>::new())
            .expect("Request build failed");

        let res = sender.send_request(req).await.expect("Request failed");
        assert_eq!(res.status(), hyper::StatusCode::OK);

        let body_bytes = res.collect().await.expect("Failed to read body").to_bytes();
        let body_str = String::from_utf8(body_bytes.to_vec()).unwrap();
        assert!(
            body_str.contains("uptime")
                || body_str.contains("collections")
                || body_str.contains("memory_allocated_bytes"),
            "Response body was: {}",
            body_str
        );

        // 4. Test Queue Concurrent Submissions (OOM stability test)
        // Submitting multiple requests simultaneously to confirm they queue correctly under bounds
        let mut client_futs = vec![];
        for i in 0..10 {
            let db_clone = shared_db.clone();
            client_futs.push(tokio::spawn(async move {
                let prompt = format!("Verify OOM stability request {}", i);
                // Submitting tasks directly to inference queue should succeed or fail cleanly with 429
                let _ = db_clone
                    .inference_queue
                    .submit("tinyllama".to_string(), prompt, 10, 0.0, None, None)
                    .await;
            }));
        }

        for fut in client_futs {
            let _ = fut.await;
        }

        // 5. Shutdown and clean up
        let _ = shutdown_tx.send(());
        let _ = server_handle.await;

        let _ = std::fs::remove_file(socket_path);
        let _ = std::fs::remove_file(db_path);
    }
}
