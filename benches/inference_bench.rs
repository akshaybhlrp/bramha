use bramha::inference::cpu_engine::generate_cpu;
use bramha::inference::engine::InferenceEngine;
use bramha::inference::set_cpu_only;
use bramha::storage::Database;
use criterion::{Criterion, criterion_group, criterion_main};
use std::sync::Arc;
use tokio::runtime::Runtime;

// Helper function to setup a self-contained mock model
async fn setup_mock_model() -> (Arc<Database>, std::path::PathBuf) {
    let db = Arc::new(Database::new(None, 1536));

    // Attempt to locate an existing tokenizer.json
    let mut tokenizer_src = std::path::PathBuf::new();
    let candidate_paths = [
        "models/all-MiniLM-L6-v2/tokenizer.json",
        "tensor_data/tinyllama-1.1b/tokenizer.json",
        "tensor_data/tinyllama/tokenizer.json",
        "/home/akshay-bhalerao/tensor_data/tinyllama-1.1b/tokenizer.json",
        "/home/akshay-bhalerao/tensor_data/tinyllama/tokenizer.json",
    ];

    for path_str in &candidate_paths {
        let p = std::path::PathBuf::from(path_str);
        if p.exists() {
            tokenizer_src = p;
            break;
        }
    }

    let temp_dir = std::env::temp_dir().join("bramha_bench_mock_model");
    let _ = std::fs::remove_dir_all(&temp_dir); // clean up old bench run dir if any
    std::fs::create_dir_all(&temp_dir).unwrap();

    if !tokenizer_src.as_os_str().is_empty() {
        std::fs::copy(&tokenizer_src, temp_dir.join("tokenizer.json")).unwrap();
    } else {
        // Fallback: Write a minimal mock tokenizer if none found
        let mock_tokenizer = r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": null,
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "BPE",
                "vocab": {
                    "<s>": 0,
                    "</s>": 1,
                    "<unk>": 2,
                    "hi": 3,
                    "hello": 4,
                    "world": 5
                },
                "merges": []
            }
        }"#;
        std::fs::write(temp_dir.join("tokenizer.json"), mock_tokenizer).unwrap();
    }

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

    bramha::storage::storage_manifest::write_test_manifest(
        &temp_dir,
        "bench-mock-model",
        vocab_size,
        hidden_size,
        num_q_heads,
        num_kv_heads,
        head_dim,
        mlp_size,
    );

    {
        let mut tensor_guard = db.tensor_db.write().await;
        tensor_guard.restore_model_at_path("bench-mock-model".to_string(), &temp_dir);
    }

    (db, temp_dir)
}

fn bench_inference(c: &mut Criterion) {
    // Disable prefix KV caching to ensure clean, isolated state per benchmark iteration.
    // Without this, a prefix cache saved on the first warm-up run would be reused on subsequent
    // iterations with incompatible KV state, causing slice index panics.
    // SAFETY: We are the only thread at this point (before Criterion spawns benchmark threads),
    // and this environment variable is only read, never concurrently mutated elsewhere.
    unsafe {
        std::env::set_var("BRAMHA_PREFIX_CACHE", "false");
    }

    let rt = Runtime::new().unwrap();
    let (db, temp_dir) = rt.block_on(setup_mock_model());

    // Clean any residual prefix cache data from a prior run
    let _ = std::fs::remove_dir_all(temp_dir.join("prefix_kv_cache_data"));

    // --- CPU BENCHMARKS ---
    {
        let mut group = c.benchmark_group("CPU Inference");

        // CPU Prefill Phase: Long Prompt, 1 target token
        let long_prompt = "hello world hello world hello world hello world hello world hello world hello world hello world hello world hello world";
        group.bench_function("prefill_phase", |b| {
            b.iter(|| {
                set_cpu_only(true);
                rt.block_on(async {
                    let _ = generate_cpu(db.clone(), "bench-mock-model", long_prompt, 1, 0.0)
                        .await
                        .unwrap();
                });
            });
        });

        // CPU Sequential Decode Phase: Short 1-token Prompt, 30 sequential tokens
        group.bench_function("sequential_decode_phase", |b| {
            b.iter(|| {
                set_cpu_only(true);
                rt.block_on(async {
                    let _ = generate_cpu(db.clone(), "bench-mock-model", "hi", 30, 0.0)
                        .await
                        .unwrap();
                });
            });
        });

        // CPU Speculative Decode Phase: Repeating n-gram Prompt, 30 sequential tokens
        let repeating_prompt = "hello world hello world hello world";
        group.bench_function("speculative_decode_phase", |b| {
            b.iter(|| {
                set_cpu_only(true);
                rt.block_on(async {
                    let _ = generate_cpu(db.clone(), "bench-mock-model", repeating_prompt, 30, 0.0)
                        .await
                        .unwrap();
                });
            });
        });

        group.finish();
    }

    // --- WGPU BENCHMARKS ---
    {
        let mut group = c.benchmark_group("WGPU Inference");

        // WGPU Prefill Phase: Long Prompt, 1 target token
        let long_prompt = "hello world hello world hello world hello world hello world hello world hello world hello world hello world hello world";
        group.bench_function("prefill_phase", |b| {
            b.iter(|| {
                set_cpu_only(false);
                rt.block_on(async {
                    let _ = InferenceEngine::new(None)
                        .generate(
                            db.clone(),
                            "bench-mock-model",
                            long_prompt,
                            1,
                            0.0,
                            None,
                            None,
                        )
                        .await
                        .unwrap();
                });
            });
        });

        // WGPU Sequential Decode Phase: Short 1-token Prompt, 30 sequential tokens
        group.bench_function("sequential_decode_phase", |b| {
            b.iter(|| {
                set_cpu_only(false);
                rt.block_on(async {
                    let _ = InferenceEngine::new(None)
                        .generate(db.clone(), "bench-mock-model", "hi", 30, 0.0, None, None)
                        .await
                        .unwrap();
                });
            });
        });

        // WGPU Speculative Decode Phase: Repeating n-gram Prompt, 30 sequential tokens
        let repeating_prompt = "hello world hello world hello world";
        group.bench_function("speculative_decode_phase", |b| {
            b.iter(|| {
                set_cpu_only(false);
                rt.block_on(async {
                    let _ = InferenceEngine::new(None)
                        .generate(
                            db.clone(),
                            "bench-mock-model",
                            repeating_prompt,
                            30,
                            0.0,
                            None,
                            None,
                        )
                        .await
                        .unwrap();
                });
            });
        });

        group.finish();
    }

    // Cleanup mock model directories
    let _ = std::fs::remove_dir_all(temp_dir);
}

criterion_group!(benches, bench_inference);
criterion_main!(benches);
