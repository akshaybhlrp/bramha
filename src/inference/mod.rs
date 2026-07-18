pub mod calibration;
pub mod cpu_engine;
pub mod embedder;
pub mod engine;
pub mod flash_attn_cpu;
pub mod paged_kv;
pub mod pipeline;
pub mod prefetcher;
pub mod prefill_cache;
pub mod profiler;
pub mod reranker;
// sparse_predictor module removed: merged into spanda-engine (spanda_engine::sparse) —
// this crate's sparse math now lives in the crate actually named for the sparse
// architecture, instead of duplicated/disconnected from it.

pub mod speculative;
pub mod tokenizer;

use std::sync::atomic::{AtomicBool, Ordering};

pub static CPU_ONLY: AtomicBool = AtomicBool::new(false);

tokio::task_local! {
    pub static CPU_ONLY_TASK: bool;
}

pub fn set_cpu_only(val: bool) {
    CPU_ONLY.store(val, Ordering::SeqCst);
}

pub fn is_cpu_only() -> bool {
    CPU_ONLY_TASK
        .try_with(|val| *val)
        .unwrap_or_else(|_| CPU_ONLY.load(Ordering::SeqCst) || std::env::var("BRAMHA_CPU").is_ok())
}
pub mod power;
pub mod spanda_backend;
pub mod spanda_telemetry;
