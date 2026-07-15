/// Minimal benchmark: Test decode-only performance (single token generation)
/// Skip prefill, prefetcher, speculative decoding, and database setup
use std::sync::Arc;
use bramha::storage::Database;
use bramha::inference::cpu_engine::generate_cpu;
use bramha::inference::set_cpu_only;
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    set_cpu_only(true);
    
    let db = if std::path::Path::new("bramha_db.bin").exists() {
        Arc::new(Database::load("bramha_db.bin").await?)
    } else {
        println!("⚠️ No bramha_db.bin found");
        return Ok(());
    };

    let model_name = {
        let tensor_guard = db.tensor_db.read().await;
        tensor_guard.models.keys().next().cloned().unwrap_or_default()
    };

    println!("🚀 DECODE-ONLY Benchmark (no prefix cache)");
    println!("Model: {}", model_name);
    println!("Generating 10 tokens from scratch...\n");

    let start = Instant::now();
    
    match generate_cpu(
        db.clone(),
        &model_name,
        "Hello",
        10,  // max_new_tokens
        0.0, // temperature
    ).await {
        Ok(result) => {
            let elapsed = start.elapsed();
            println!("\n📊 Results:");
            println!("Elapsed: {:.4}s", elapsed.as_secs_f64());
            println!("Tokens: {}", result.tokens_generated);
            println!("Speed: {:.2} tps", result.tokens_per_second);
            println!("\nCompletion: {}", result.completion);
        }
        Err(e) => println!("❌ Error: {}", e),
    }

    Ok(())
}
