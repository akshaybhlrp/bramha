/// Simple benchmarking script to test profiling and measure token speed
use std::sync::Arc;
use bramha::storage::Database;
use bramha::inference::cpu_engine::generate_cpu;
use bramha::inference::set_cpu_only;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    set_cpu_only(true);
    
    // Try to load existing database
    let db = if std::path::Path::new("bramha_db.bin").exists() {
        Arc::new(Database::load("bramha_db.bin").await?)
    } else {
        println!("⚠️ No bramha_db.bin found. Creating empty database.");
        println!("Please ingest a model first using the main API.");
        return Ok(());
    };

    // Check available models
    {
        let tensor_guard = db.tensor_db.read().await;
        let models: Vec<_> = tensor_guard.models.keys().cloned().collect();
        
        if models.is_empty() {
            println!("⚠️ No models found in database. Please ingest a model first.");
            return Ok(());
        }
        
        println!("📦 Available models: {:?}", models);
    }

    // Test with the first available model
    let model_name = {
        let tensor_guard = db.tensor_db.read().await;
        tensor_guard.models.keys().next().cloned().unwrap_or_default()
    };

    if model_name.is_empty() {
        println!("❌ Could not find a model name");
        return Ok(());
    }

    println!("\n🚀 Starting CPU Inference Benchmark");
    println!("Model: {}", model_name);
    println!("═══════════════════════════════════════════════════════════════\n");

    // Test prompt
    let prompt = "Hello";
    println!("📝 Prompt: {}", prompt);
    println!();

    // Run inference
    match generate_cpu(
        db.clone(),
        &model_name,
        prompt,
        20,  // max_new_tokens
        0.0, // temperature (greedy)
    ).await {
        Ok(result) => {
            println!("\n✅ Inference Complete!");
            println!("───────────────────────────────────────────────────────────");
            println!("Generated Tokens: {}", result.tokens_generated);
            println!("Elapsed Time: {:.2}s", result.elapsed_seconds);
            println!("Tokens/Second: {:.2} tps", result.tokens_per_second);
            println!("───────────────────────────────────────────────────────────");
            println!("\n📄 Completion:\n{}\n", result.completion);
            
            if result.tokens_per_second >= 50.0 {
                println!("🎉 CPU TARGET MET! {:.2} tps >= 50.0 tps", result.tokens_per_second);
            } else if result.tokens_per_second >= 25.0 {
                println!("✓ Good performance: {:.2} tps (target: 50+ tps)", result.tokens_per_second);
            } else {
                println!("⚠️ Below target: {:.2} tps (target: 50+ tps)", result.tokens_per_second);
            }
        }
        Err(e) => {
            println!("❌ Error: {}", e);
        }
    }

    println!("\n📊 Storage Efficiency Statistics");
    {
        let tensor_guard = db.tensor_db.read().await;
        if let Ok(content_storage) = tensor_guard.content_storage.lock() {
            content_storage.report();
        }
        if let Ok(multi_tier) = tensor_guard.multi_tier.lock() {
            multi_tier.report();
        }
    }

    Ok(())
}
