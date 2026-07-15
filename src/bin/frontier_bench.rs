use bramha::inference::engine::InferenceEngine;
use bramha::inference::set_cpu_only;
use bramha::storage::Database;
use std::fs::File;
use std::io::Write;
use std::sync::Arc;
use std::time::Instant;
use tokio::runtime::Runtime;

async fn setup_mock_model() -> (Arc<Database>, std::path::PathBuf) {
    let db = Arc::new(Database::new(None, 1536));
    let temp_dir = std::env::temp_dir().join("bramha_frontier_bench_mock");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();

    // Write a mock tokenizer
    let mock_tokenizer = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,"model":{"type":"BPE","vocab":{"<s>":0,"</s>":1,"<unk>":2,"hi":3,"hello":4,"world":5},"merges":[]}}"#;
    std::fs::write(temp_dir.join("tokenizer.json"), mock_tokenizer).unwrap();

    let write_dummy_weight = |name: &str, size: usize| {
        let data = vec![0.0f32; size];
        let bytes = bytemuck::cast_slice(&data);
        let p = temp_dir.join(name.replace(".", "_") + ".bin");
        std::fs::write(&p, bytes).unwrap();
    };

    let vocab_size = 256;
    let hidden_size = 64;
    let head_dim = 16;
    let num_q_heads = 4;
    let num_kv_heads = 1;
    let mlp_size = 64;

    write_dummy_weight("model.embed_tokens.weight", vocab_size * hidden_size);
    write_dummy_weight("lm_head.weight", vocab_size * hidden_size);
    write_dummy_weight("model.norm.weight", hidden_size);
    write_dummy_weight("model.layers.0.input_layernorm.weight", hidden_size);
    write_dummy_weight(
        "model.layers.0.self_attn.q_proj.weight",
        (num_q_heads * head_dim) * hidden_size,
    );
    write_dummy_weight(
        "model.layers.0.self_attn.k_proj.weight",
        (num_kv_heads * head_dim) * hidden_size,
    );
    write_dummy_weight(
        "model.layers.0.self_attn.v_proj.weight",
        (num_kv_heads * head_dim) * hidden_size,
    );
    write_dummy_weight(
        "model.layers.0.self_attn.o_proj.weight",
        hidden_size * (num_q_heads * head_dim),
    );
    write_dummy_weight(
        "model.layers.0.post_attention_layernorm.weight",
        hidden_size,
    );
    write_dummy_weight(
        "model.layers.0.mlp.gate_proj.weight",
        mlp_size * hidden_size,
    );
    write_dummy_weight("model.layers.0.mlp.up_proj.weight", mlp_size * hidden_size);
    write_dummy_weight(
        "model.layers.0.mlp.down_proj.weight",
        hidden_size * mlp_size,
    );

    bramha::storage::storage_manifest::write_mock_manifest(
        &temp_dir,
        "frontier-mock-model",
        vocab_size,
        hidden_size,
        num_q_heads,
        num_kv_heads,
        head_dim,
        mlp_size,
    );

    {
        let mut tensor_guard = db.tensor_db.write().await;
        tensor_guard.restore_model_at_path("frontier-mock-model".to_string(), &temp_dir);
    }

    (db, temp_dir)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🏁 Starting Frontier-Based Benchmark...");

    // Disable prefix caching to get raw measurements per configuration
    unsafe {
        std::env::set_var("BRAMHA_PREFIX_CACHE", "false");
    }
    set_cpu_only(true);

    let rt = Runtime::new().unwrap();
    let (db, temp_dir) = rt.block_on(setup_mock_model());

    let prefill_frontiers = vec![16, 64, 128, 2048, 4096, 8192];
    let generation_frontiers = vec![16, 64, 128];

    let csv_path = "frontier_benchmarks.csv";
    let mut csv_file = File::create(csv_path)?;
    writeln!(
        csv_file,
        "prefill_tokens,generation_tokens,prefill_latency_ms,prefill_rate_tps,generation_latency_ms,generation_rate_tps"
    )?;

    println!("--------------------------------------------------------------------------------");
    println!(" Prefill | Gen | Prefill Latency | Prefill Rate | Gen Latency | Gen Rate");
    println!("--------------------------------------------------------------------------------");

    for &prefill_len in &prefill_frontiers {
        // Construct prompt of approx `prefill_len` tokens
        let words = vec!["hello"; prefill_len];
        let prompt = words.join(" ");

        for &gen_len in &generation_frontiers {
            // Measure prefill (time to generate 1 token)
            let start_prefill = Instant::now();
            rt.block_on(async {
                let _ = InferenceEngine::new(None)
                    .generate(
                        db.clone(),
                        "frontier-mock-model",
                        &prompt,
                        1,
                        0.0,
                        None,
                        None,
                    )
                    .await
                    .unwrap();
            });
            let prefill_dur = start_prefill.elapsed();
            let prefill_rate = (prefill_len as f64) / prefill_dur.as_secs_f64();

            // Measure generation (generate `gen_len` tokens)
            let start_gen = Instant::now();
            rt.block_on(async {
                let _ = InferenceEngine::new(None)
                    .generate(
                        db.clone(),
                        "frontier-mock-model",
                        &prompt,
                        gen_len,
                        0.0,
                        None,
                        None,
                    )
                    .await
                    .unwrap();
            });
            let gen_dur = start_gen.elapsed();
            let gen_rate = (gen_len as f64) / gen_dur.as_secs_f64();

            println!(
                " {:7} | {:3} | {:12.2}ms | {:8.2} tps | {:9.2}ms | {:6.2} tps",
                prefill_len,
                gen_len,
                prefill_dur.as_secs_f64() * 1000.0,
                prefill_rate,
                gen_dur.as_secs_f64() * 1000.0,
                gen_rate
            );

            writeln!(
                csv_file,
                "{},{},{:.2},{:.2},{:.2},{:.2}",
                prefill_len,
                gen_len,
                prefill_dur.as_secs_f64() * 1000.0,
                prefill_rate,
                gen_dur.as_secs_f64() * 1000.0,
                gen_rate
            )?;
        }
    }

    println!("--------------------------------------------------------------------------------");
    println!(
        "✅ Frontier benchmark completed. Results saved to: {}",
        csv_path
    );

    let _ = std::fs::remove_dir_all(temp_dir);
    Ok(())
}
