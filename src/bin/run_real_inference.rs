use bramha::inference::engine::InferenceEngine;
use bramha::storage::Database;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Loading Database...");

    // Set planner mode to exact_only to bypass CachedAnswer and force actual generation pass
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
        Arc::new(Database::new(None, 1536))
    };

    let model_path = std::path::Path::new("/home/akshay-bhalerao/tensor_data/tinyllama-1.1b");

    // Register the model at /home/akshay-bhalerao/tensor_data/tinyllama-1.1b if not already registered
    {
        let mut tensor_guard = db.tensor_db.write().await;
        tensor_guard.restore_model_at_path("tinyllama-1.1b".to_string(), model_path);
        if let Some(model) = tensor_guard.models.get_mut("tinyllama-1.1b") {
            model.active_device = "cpu".to_string();
        }
    }

    println!("\n=========================================");
    println!("🧪 TESTING ACTUAL INFERENCE - CPU-ONLY MODE");
    println!("=========================================");

    // Force CPU mode
    bramha::inference::set_cpu_only(true);
    let prompt = "Explain what is a black hole in a single simple sentence.";
    let result_cpu = InferenceEngine::new(None)
        .generate(
            db.clone(),
            "tinyllama-1.1b",
            prompt,
            25,  // generate 25 tokens
            0.0, // temperature = 0 (deterministic)
            None,
            None,
        )
        .await;

    match result_cpu {
        Ok(res) => {
            println!("\n[CPU RESULT]");
            println!("Completion: {}", res.completion);
            println!("Tokens generated: {}", res.tokens_generated);
            println!("Tokens/sec: {:.2} tokens/s", res.tokens_per_second);
            println!("Elapsed: {:.2}s", res.elapsed_seconds);
        }
        Err(e) => {
            println!("❌ CPU Inference failed: {}", e);
        }
    }

    println!("\n=========================================");
    println!("🧪 TESTING ACTUAL INFERENCE - GPU-ONLY MODE");
    println!("=========================================");

    // Switch to GPU mode
    bramha::inference::set_cpu_only(false);
    {
        let mut tensor_guard = db.tensor_db.write().await;
        if let Some(model) = tensor_guard.models.get_mut("tinyllama-1.1b") {
            model.active_device = "gpu".to_string();
        }
        // Disable VRAM cap so scheduler routes tinyllama-1.1b (0.55GB) to GPU
        let mut cache = bramha::inference::engine::VramCache::global()
            .lock()
            .unwrap();
        cache.max_vram_bytes = None;
    }

    let result_gpu = InferenceEngine::new(None)
        .generate(
            db.clone(),
            "tinyllama-1.1b",
            prompt,
            25,  // generate 25 tokens
            0.0, // temperature = 0 (deterministic)
            None,
            None,
        )
        .await;

    match result_gpu {
        Ok(res) => {
            println!("\n[GPU RESULT]");
            println!("Completion: {}", res.completion);
            println!("Tokens generated: {}", res.tokens_generated);
            println!("Tokens/sec: {:.2} tokens/s", res.tokens_per_second);
            println!("Elapsed: {:.2}s", res.elapsed_seconds);
        }
        Err(e) => {
            println!("❌ GPU Inference failed: {}", e);
        }
    }

    Ok(())
}
