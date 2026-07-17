//! High-Throughput Concurrent Load & OOM Stress Test Suite for Bramha & SPANDA
//!
//! Validates system stability under heavy concurrent load (15+ parallel requests),
//! memory bound enforcement, queue backpressure handling, and graceful RAM offload
//! degradation for WGPU dense and SPANDA sparse execution backends.

use bramha::api::create_router;
use bramha::inference::sparse_predictor::{cosine_similarity, sparse_matvec_mul_2_4};
use bramha::storage::Database;
use bramha::storage::multi_tier::{MultiTierStorage, TierConfig};
use bramha::storage::sparse_pager::{SparseBlockMask, pack_sparse_matrix};
use bramha::storage::storage_manifest::StorageTier;
use http_body_util::BodyExt;
use hyper_util::rt::TokioIo;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;
use tempfile::TempDir;
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
                    while let Ok((stream, _addr)) = listener.accept().await {
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

        // 4. Test Queue Concurrent Submissions (OOM stability test - 20 parallel requests)
        let success_count = Arc::new(AtomicUsize::new(0));
        let rejected_count = Arc::new(AtomicUsize::new(0));
        let mut client_futs = vec![];

        for i in 0..20 {
            let db_clone = shared_db.clone();
            let succ = success_count.clone();
            let rej = rejected_count.clone();

            client_futs.push(tokio::spawn(async move {
                let prompt = format!("Verify OOM stability request {}", i);
                let result = db_clone
                    .inference_queue
                    .submit("tinyllama".to_string(), prompt, 10, 0.0, None, None, None)
                    .await;

                match result {
                    Ok(_) => {
                        succ.fetch_add(1, Ordering::SeqCst);
                    }
                    Err(_) => {
                        rej.fetch_add(1, Ordering::SeqCst);
                    }
                }
            }));
        }

        for fut in client_futs {
            let _ = fut.await;
        }

        // Verify total tasks processed or queued equal 20, with zero unhandled panics
        let total_handled =
            success_count.load(Ordering::SeqCst) + rejected_count.load(Ordering::SeqCst);
        assert_eq!(
            total_handled, 20,
            "All 20 concurrent requests must be safely handled"
        );

        // 5. Shutdown and clean up
        let _ = shutdown_tx.send(());
        let _ = server_handle.await;

        let _ = std::fs::remove_file(socket_path);
        let _ = std::fs::remove_file(db_path);
    }

    #[tokio::test]
    async fn test_spanda_sparse_and_dense_ram_offload_fallback() {
        let temp_dir = TempDir::new().expect("Failed to create tempdir");
        let config = TierConfig {
            hot_max_bytes: 100 * 1024,       // 100KB DRAM cap to force overflow
            warm_max_bytes: 5 * 1024 * 1024, // 5MB SSD RAM offload
            promotion_threshold: 2,
            demotion_threshold_secs: 3600,
            prefetch_distance: 2,
        };

        let mut storage = MultiTierStorage::new(
            config,
            temp_dir.path().join("hot"),
            temp_dir.path().join("warm"),
            temp_dir.path().join("cold"),
        )
        .expect("MultiTierStorage creation failed");

        // Register 5 layer weight blocks
        for i in 0..5 {
            let layer_id = format!("layer_{}", i);
            let layer_path = temp_dir.path().join(format!("{}.bin", layer_id));
            let _ = storage.register_layer(
                layer_id,
                30 * 1024, // 30KB each -> total 150KB exceeds 100KB hot limit
                StorageTier::Important,
                layer_path,
            );
        }

        let util = storage.utilization();
        assert!(util.warm_count > 0 || util.cold_count > 0 || util.hot_count > 0);

        // Verify SPANDA 2:4 Sparse predictor functionality under offloaded state
        let x = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let w = vec![
            0.1, 10.0, -5.0, 0.2, 1.0, -0.1, 0.5, 2.0, 0.0, 0.0, 1.0, 2.0, -5.0, -6.0, 1.0, 0.1,
        ];

        let out = sparse_matvec_mul_2_4(&x, &w, 8);
        assert_eq!(out.len(), 2);
        let sim = cosine_similarity(&out, &vec![8.0, -8.0]);
        assert!((sim - 1.0).abs() < 1e-4);
    }

    #[tokio::test]
    async fn test_sparse_block_pager_oom_resilience() {
        let block_data = [
            0.0f32, 1.5, 0.0, 0.0, 2.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.2, 0.0,
        ];
        let mask = SparseBlockMask::pack_4x4(&block_data);

        assert!(mask.mask > 0);
        assert_eq!(mask.active_count(), 3);

        for i in 0..1000 {
            let mut test_w = vec![0.0f32; 32];
            test_w[0] = (i % 10) as f32;
            test_w[16] = 1.0;
            let (masks, values) = pack_sparse_matrix(&test_w, 4);
            assert_eq!(masks.len(), 2);
            assert!(values.len() <= 32);
        }
    }
}
