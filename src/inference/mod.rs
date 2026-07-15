pub mod calibration;
pub mod cpu_engine;
pub mod embedder;
pub mod engine;
pub mod paged_kv;
pub mod pipeline;
pub mod prefetcher;
pub mod prefill_cache;
pub mod profiler;
pub mod reranker;
pub mod sparse_predictor;
pub mod tokenizer;

use std::sync::atomic::{AtomicBool, Ordering};

pub static CPU_ONLY: AtomicBool = AtomicBool::new(false);

pub fn set_cpu_only(val: bool) {
    CPU_ONLY.store(val, Ordering::SeqCst);
}

pub fn is_cpu_only() -> bool {
    CPU_ONLY.load(Ordering::SeqCst) || std::env::var("BRAMHA_CPU").is_ok()
}
pub mod power;
pub mod spanda_backend;
