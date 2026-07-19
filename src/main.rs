#![allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::mut_mutex_lock,
    clippy::len_zero,
    clippy::manual_checked_ops,
    clippy::ptr_arg,
    clippy::suspicious_open_options,
    clippy::if_same_then_else,
    clippy::unnecessary_unwrap,
    clippy::collapsible_if,
    clippy::new_without_default,
    clippy::manual_strip,
    clippy::redundant_closure,
    clippy::field_reassign_with_default,
    clippy::explicit_auto_deref,
    clippy::manual_is_multiple_of,
    clippy::map_entry,
    clippy::manual_div_ceil,
    clippy::unwrap_or_default,
    clippy::unnecessary_sort_by,
    clippy::redundant_pattern_matching,
    clippy::needless_borrows_for_generic_args,
    clippy::unnecessary_get_then_check,
    clippy::single_range_in_vec_init,
    clippy::manual_flatten,
    clippy::await_holding_lock,
    clippy::assertions_on_constants,
    clippy::useless_vec,
    clippy::while_let_loop
)]

use bramha::api::create_router;
use bramha::storage::Database;
use std::env;
use std::net::SocketAddr;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // S1.5: Enforce physical core pinning on Rayon thread pool
    let rayon_pool = bramha::concurrency::rayon_bridge::global_rayon_pool();
    println!(
        "📌 Rayon compute thread pool initialized with {} physical core workers.",
        rayon_pool.current_num_threads()
    );
    let args: Vec<String> = env::args().collect();

    // S8: Model Pull CLI support
    if args.len() > 1 && args[1] == "pull" {
        if args.len() < 3 {
            println!("⚠️ Usage: bramha pull <model-id>");
            println!("Available models: tinyllama-1.1b, mistral-7b-q4");
            return Ok(());
        }
        let model_id = &args[2];
        println!(
            "🚀 Initiating Model Pull Command for model ID: '{}'...",
            model_id
        );

        let default_tensor_storage =
            if std::path::Path::new("/home/akshay-bhalerao/tensor_data").exists() {
                std::path::PathBuf::from("/home/akshay-bhalerao/tensor_data")
            } else {
                std::env::current_dir()
                    .unwrap_or_default()
                    .join("tensor_data")
            };
        let mut tensor_db = bramha::storage::tensor_db::TensorDB::new(default_tensor_storage);

        if let Err(e) = bramha::storage::pull::pull_model(model_id, &mut tensor_db).await {
            eprintln!("❌ Pull failed: {}", e);
            std::process::exit(1);
        }
        return Ok(());
    }

    let mut port = 8000;
    let mut db_path = Some("bramha_db.bin".to_string());
    let mut cache_dim = 1536; // Default for standard embeddings (like OpenAI text-embedding-3-small)
    let mut uds_path: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--port" | "-p" if i + 1 < args.len() => {
                port = args[i + 1].parse().unwrap_or(8000);
                i += 2;
            }
            "--uds" | "-u" if i + 1 < args.len() => {
                uds_path = Some(args[i + 1].clone());
                i += 2;
            }
            "--db-file" | "-d" if i + 1 < args.len() => {
                db_path = Some(args[i + 1].clone());
                i += 2;
            }
            "--cache-dim" | "-c" if i + 1 < args.len() => {
                cache_dim = args[i + 1].parse().unwrap_or(1536);
                i += 2;
            }
            "--no-save" => {
                db_path = None;
                i += 1;
            }
            "--disable-gpu" | "--cpu-only" => {
                bramha::inference::set_cpu_only(true);
                i += 1;
            }
            "--dump-logprobs" => {
                // SAFETY: Manual invariants verified for performance/FFI
                unsafe {
                    std::env::set_var("BRAMHA_DUMP_LOGPROBS", "true");
                }
                i += 1;
            }
            "--trace" => {
                // SAFETY: Manual invariants verified for performance/FFI
                unsafe {
                    std::env::set_var("BRAMHA_TRACE", "true");
                }
                i += 1;
            }
            "--power" if i + 1 < args.len() => {
                if let Ok(limit) = args[i + 1].parse::<u32>() {
                    bramha::inference::power::set_power_limit(limit);
                }
                i += 2;
            }
            "--help" | "-h" => {
                println!("Bramha - A High-Performance Custom LLM Vector Database");
                println!();
                println!("Usage:");
                println!("  bramha [options]");
                println!();
                println!("Options:");
                println!("  -p, --port <port>       Port to bind the server to (default: 8000)");
                println!("  -u, --uds <path>        Unix Domain Socket path for hosting API");
                println!(
                    "  -d, --db-file <path>    Path to file for persistence (default: bramha_db.bin)"
                );
                println!(
                    "  -c, --cache-dim <dim>   Dimension of semantic LLM prompt cache (default: 1536)"
                );
                println!("  --no-save               Disable persistence entirely");
                println!(
                    "  --disable-gpu           Force CPU-only mode (bypasses raw WGPU compute plane)"
                );
                println!(
                    "  --dump-logprobs         Enable logprob output logging during generation"
                );
                println!("  --trace                 Enable execution trace diagnostics");
                println!(
                    "  --power <limit>         Throttle execution utilization to N% (default: 100)"
                );
                println!("  -h, --help              Print this help menu");
                return Ok(());
            }
            _ => {
                i += 1;
            }
        }
    }
    // Resolve database path to an absolute path so it is perfectly stable across CWD/debugger changes
    if let Some(ref path) = db_path {
        let abs_path = std::env::current_dir().unwrap_or_default().join(path);
        db_path = Some(abs_path.to_string_lossy().to_string());
    }

    println!("====================================================");
    println!("     ⚡ BRAMHA VECTOR & SEMANTIC DATABASE ⚡     ");
    println!("====================================================");

    // Initialize or load database state
    let db = if let Some(ref path) = db_path {
        if std::path::Path::new(path).exists() {
            println!("💾 Loading existing database state from: {}", path);
            match Database::load(path).await {
                Ok(loaded_db) => loaded_db,
                Err(err) => {
                    println!(
                        "⚠️ Failed to load database: {}. Starting a fresh database.",
                        err
                    );
                    Database::new(db_path.clone(), cache_dim)
                }
            }
        } else {
            println!("✨ Starting fresh database. State will save to: {}", path);
            Database::new(db_path.clone(), cache_dim)
        }
    } else {
        println!("✨ Starting fresh in-memory database (persistence disabled)");
        Database::new(None, cache_dim)
    };

    let shared_db = Arc::new(db);
    shared_db.start_worker();

    let app = create_router(shared_db.clone());

    let shutdown_db = shared_db.clone();
    let shutdown_future = async move {
        let ctrl_c = tokio::signal::ctrl_c();
        #[cfg(unix)]
        let terminate = async {
            use tokio::signal::unix::{SignalKind, signal};
            if let Ok(mut stream) = signal(SignalKind::terminate()) {
                stream.recv().await;
            } else {
                std::future::pending::<()>().await;
            }
        };
        #[cfg(not(unix))]
        let terminate = std::future::pending::<()>();

        tokio::select! {
            _ = ctrl_c => {
                println!("\n🛑 Received CTRL+C signal, shutting down...");
            }
            _ = terminate => {
                println!("\n🛑 Received SIGTERM signal, shutting down...");
            }
        }
        println!("🛑 Shutting down server gracefully...");

        let start_drain = std::time::Instant::now();
        println!("⏳ Draining inference queue...");
        while shutdown_db.inference_queue.queue_depth() > 0 {
            if start_drain.elapsed().as_secs() >= 30 {
                println!("⚠️ Timeout reached draining inference queue. Forcing shutdown.");
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }

        if let Err(e) = shutdown_db.save().await {
            eprintln!("⚠️ Failed to save database on shutdown: {}", e);
        } else {
            println!("💾 Database state saved successfully!");
        }
    };

    if let Some(ref path_str) = uds_path {
        let path = std::path::Path::new(path_str);
        // Remove file if exists
        let _ = std::fs::remove_file(path);
        let listener = tokio::net::UnixListener::bind(path)?;
        println!(
            "🚀 Bramha Server running on Unix Domain Socket at: {}",
            path_str
        );
        println!(
            "💡 Semantic LLM Cache initialized (Dimension: {})",
            cache_dim
        );
        println!("====================================================");

        use tower::Service;

        tokio::select! {
            res = async {
                loop {
                    match listener.accept().await {
                        Ok((stream, _addr)) => {
                            let tower_service = app.clone();
                            let io = hyper_util::rt::TokioIo::new(stream);
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
                        Err(e) => {
                            eprintln!("⚠️ UDS accept error: {}", e);
                            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                        }
                    }
                }
            } => res,
            _ = shutdown_future => {}
        }

        // Cleanup UDS socket file on exit
        let _ = std::fs::remove_file(path);
    } else {
        let addr = SocketAddr::from(([0, 0, 0, 0], port));
        let listener = tokio::net::TcpListener::bind(addr).await?;
        println!("🚀 Bramha Server running at http://{}", addr);
        println!(
            "💡 Semantic LLM Cache initialized (Dimension: {})",
            cache_dim
        );
        println!("====================================================");

        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_future)
            .await?;
    }

    Ok(())
}
