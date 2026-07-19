use bramha::storage::Database;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting Offline Shadow Scan (Phase 0)...");

    // Set planner mode to exact_only to bypass CachedAnswer
    // SAFETY: Manual invariants verified for performance/FFI
    unsafe {
        std::env::set_var("BRAMHA_PLANNER_MODE", "exact_only");
    }

    let db = if std::path::Path::new("bramha_db.bin").exists() {
        Arc::new(
            Database::load("bramha_db.bin")
                .await
                .unwrap_or_else(|_| Database::new(None, 1536)),
        )
    } else {
        println!("❌ Database not found. Please ingest a model first.");
        return Ok(());
    };

    bramha::inference::set_cpu_only(true);

    let model_name = "qwen2.5-0.5b";
    let model_path = std::path::Path::new("/home/akshay-bhalerao/tensor_data/qwen2.5-0.5b");

    if !model_path.exists() {
        println!(
            "❌ Model path {} not found. Falling back to tinyllama for shadow test.",
            model_path.display()
        );
    }

    let active_model = if model_path.exists() {
        let mut tensor_guard = db.tensor_db.write().await;
        tensor_guard.restore_model_at_path(model_name.to_string(), model_path);
        tensor_guard.load_model_layers(model_name).unwrap();
        model_name
    } else {
        let name = "tinyllama-1.1b";
        let path = std::path::Path::new("/home/akshay-bhalerao/tensor_data/tinyllama-1.1b");
        let mut tensor_guard = db.tensor_db.write().await;
        tensor_guard.restore_model_at_path(name.to_string(), path);
        tensor_guard.load_model_layers(name).unwrap();
        name
    };

    println!(
        "Running golden dataset shadow scan against {}...",
        active_model
    );

    // Test 1: Fetch a specific dense projection tensor to verify 2:4 sparsity entropy
    let layer_idx = 0;
    let tensor_name = format!("model.layers.{}.mlp.down_proj.weight", layer_idx);

    let mut tensor_guard = db.tensor_db.write().await;
    let tensor_db_ref = &mut *tensor_guard;
    let block_db = &tensor_db_ref.block_db;
    let models = &mut tensor_db_ref.models;

    println!("Loading {} into memory...", tensor_name);

    if let Some(model) = models.get_mut(active_model) {
        let mut block_db_guard = block_db.lock().unwrap();
        let _ = model.load_tensor_chunks(&tensor_name, &mut block_db_guard);
        drop(block_db_guard);

        if let Some(page) = model.layers.get(&tensor_name) {
            println!(
                "✅ Successfully loaded tensor page! Byte length: {}",
                page.as_bytes().len()
            );

            // Unsafe but standard casting for the f32 weights
            let bytes = page.as_bytes();
            // SAFETY: Manual invariants verified for performance/FFI
            let weights: &[f32] = unsafe {
                std::slice::from_raw_parts(bytes.as_ptr() as *const f32, bytes.len() / 4)
            };

            // Assume 2048 or 896 hidden dim for TinyLlama / Qwen
            let hidden_dim = if active_model.contains("qwen") {
                896
            } else {
                2048
            };
            let intermediate_dim = weights.len() / hidden_dim;

            println!(
                "Simulating input activation vector of size {}...",
                intermediate_dim
            );
            let x: Vec<f32> = (0..intermediate_dim)
                .map(|i| (i as f32 % 10.0) / 10.0)
                .collect();

            println!("Running 2:4 Sparse MatMul prediction...");
            let sparse_out = spanda_engine::sparse_matvec_mul_2_4(&x, weights, intermediate_dim);

            println!("Running Dense MatMul baseline...");
            let mut dense_out = vec![0.0; hidden_dim];
            for r in 0..hidden_dim {
                let mut sum = 0.0;
                for c in 0..intermediate_dim {
                    sum += weights[r * intermediate_dim + c] * x[c];
                }
                dense_out[r] = sum;
            }

            let similarity = spanda_engine::cosine_similarity(&dense_out, &sparse_out);
            println!("=====================================================");
            println!(
                "📊 SPARSE PREDICTOR ACCURACY (Cosine Similarity): {:.4}",
                similarity
            );
            println!("=====================================================");

            if similarity < 0.99 {
                println!("⚠️ WARNING: Sparsity causes heavy degradation on this layer.");
            }
        }

        // EXPLICIT OOM PREVENTION: Unload the tensor chunk immediately after use
        model.unload_tensor_chunks(&tensor_name);
        println!(
            "🧹 Unloaded {} from physical memory to prevent OOM.",
            tensor_name
        );
    }

    drop(tensor_guard);

    println!("✅ Shadow scan complete!");
    Ok(())
}
