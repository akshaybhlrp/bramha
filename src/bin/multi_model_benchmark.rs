use bramha::inference::engine::InferenceEngine;
use bramha::storage::Database;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Loading Database...");

    // Set planner mode to exact_only to bypass CachedAnswer and force actual generation pass
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

    // Force CPU mode for clean comparison
    bramha::inference::set_cpu_only(true);

    let prompt = "What is the meaning of life? Answer in one sentence.";

    // =========================================================
    // Model 1: TinyLlama-1.1B
    // =========================================================
    {
        let model_name = "tinyllama-1.1b";
        let model_path = std::path::Path::new("/home/akshay-bhalerao/tensor_data/tinyllama-1.1b");

        let mut tensor_guard = db.tensor_db.write().await;
        tensor_guard.restore_model_at_path(model_name.to_string(), model_path);
        if let Some(model) = tensor_guard.models.get_mut(model_name) {
            model.active_device = "cpu".to_string();
        }
        drop(tensor_guard);

        println!("\n╔═══════════════════════════════════════════════════╗");
        println!("║  MODEL 1: TinyLlama-1.1B-Chat  (2.2 GB FP32)    ║");
        println!("╚═══════════════════════════════════════════════════╝");

        let result = InferenceEngine::new(None)
            .generate(db.clone(), model_name, prompt, 30, 0.0, None, None)
            .await;

        match result {
            Ok(res) => {
                println!("\n📝 Completion: {}", res.completion);
                println!(
                    "   Tokens: {} | Speed: {:.2} tok/s | Time: {:.2}s",
                    res.tokens_generated, res.tokens_per_second, res.elapsed_seconds
                );
            }
            Err(e) => println!("❌ TinyLlama inference failed: {}", e),
        }

        // UNLOAD TINYLLAMA FROM MEMORY COMPLETELY TO PREVENT OOM
        let mut tensor_guard = db.tensor_db.write().await;
        tensor_guard.models.remove(model_name);
        println!("🔄 Unloaded model '{}' completely from memory", model_name);
    }

    // =========================================================
    // Model 2: Qwen2.5-0.5B-Instruct
    // =========================================================
    {
        let model_name = "qwen2.5-0.5b";
        let model_path = std::path::Path::new("/home/akshay-bhalerao/tensor_data/qwen2.5-0.5b");

        let mut tensor_guard = db.tensor_db.write().await;
        tensor_guard.restore_model_at_path(model_name.to_string(), model_path);
        if let Some(model) = tensor_guard.models.get_mut(model_name) {
            model.active_device = "cpu".to_string();
        }
        drop(tensor_guard);

        println!("\n╔═══════════════════════════════════════════════════╗");
        println!("║  MODEL 2: Qwen2.5-0.5B-Instruct  (988 MB FP32)  ║");
        println!("╚═══════════════════════════════════════════════════╝");

        let result = InferenceEngine::new(None)
            .generate(db.clone(), model_name, prompt, 30, 0.0, None, None)
            .await;

        match result {
            Ok(res) => {
                println!("\n📝 Completion: {}", res.completion);
                println!(
                    "   Tokens: {} | Speed: {:.2} tok/s | Time: {:.2}s",
                    res.tokens_generated, res.tokens_per_second, res.elapsed_seconds
                );
            }
            Err(e) => println!("❌ Qwen inference failed: {}", e),
        }
    }

    println!("\n═══════════════════════════════════════════════════");
    println!("✅ Multi-model benchmark complete!");
    println!("═══════════════════════════════════════════════════");

    Ok(())
}
