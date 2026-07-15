pub mod engine;
pub mod cpu_engine;
pub mod prefill_cache;
pub mod prefetcher;
pub mod calibration;
pub mod embedder;
pub mod reranker;
pub mod tokenizer;
pub mod paged_kv;
pub mod profiler;
pub mod pipeline;
pub mod sparse_predictor;


use std::sync::atomic::{AtomicBool, Ordering};

pub static CPU_ONLY: AtomicBool = AtomicBool::new(false);

pub fn set_cpu_only(val: bool) {
    CPU_ONLY.store(val, Ordering::SeqCst);
}

pub fn is_cpu_only() -> bool {
    CPU_ONLY.load(Ordering::SeqCst) || std::env::var("BRAMHA_CPU").is_ok()
}
pub mod spanda_backend;
pub mod power;
