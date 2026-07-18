//! Feeds real per-request telemetry into spanda-engine's Phase 0 Jaccard gate.
//!
//! Deliberately NOT hooked into the per-layer decode loop in `cpu_engine.rs`: that
//! loop has no early-exit (`total_exit_layers` there is a hardcoded constant, not a
//! measurement — see hardening audit) so every request touches every layer, every
//! time. Recording that would only ever produce a trivial Jaccard score of 1.0 —
//! real-looking telemetry with no actual signal behind it, which is exactly the
//! "looks implemented, isn't" pattern this whole audit has been finding elsewhere.
//!
//! Instead this hooks the one access pattern in the request path that is *actually*
//! variable today: which model each request loads. This server serves multiple
//! models (`TensorDB::ensure_model_loaded` / `unload_model_if_virtual` run per
//! request), so a real production traffic mix has real locality/repetition
//! structure worth measuring — and recording it here carries zero risk to
//! inference correctness, since it's pure telemetry, no numerical path is touched.

use spanda_engine::PagingEngine;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::{Mutex, OnceLock};

/// Placeholder budget: this tracker is currently used for Phase 0 measurement only
/// (`gate_status()`), not for actual page storage — see module doc. Real paging
/// activation (`try_activate_paging`) is intentionally not called anywhere yet;
/// wiring it to actually evict/reload model weights is future work once the gate
/// has accumulated real production traffic data to evaluate.
const TELEMETRY_ONLY_BUDGET_BYTES: usize = 0;

static MODEL_ACCESS_TRACKER: OnceLock<Mutex<PagingEngine>> = OnceLock::new();

fn tracker() -> &'static Mutex<PagingEngine> {
    MODEL_ACCESS_TRACKER.get_or_init(|| Mutex::new(PagingEngine::new(TELEMETRY_ONLY_BUDGET_BYTES)))
}

fn stable_id_for_model(model_name: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    model_name.hash(&mut hasher);
    hasher.finish()
}

/// Call once per inference request with the model it's about to use. Cheap
/// (one hash, one lock, one Vec push into a capped ring buffer) — safe to call
/// unconditionally on the request hot path.
pub fn record_model_access(model_name: &str) {
    let id = stable_id_for_model(model_name);
    if let Ok(mut engine) = tracker().lock() {
        engine.record_step(&[id]);
    }
}

/// Current Phase 0 gate status for this process's real model-access traffic.
/// Wired into `GET /api/system/diagnostics` (was a stub returning
/// `{"status":"unimplemented"}`) so this is actually observable instead of only
/// living in-process with no way to check it.
pub fn current_gate_status() -> spanda_engine::GateResult {
    match tracker().lock() {
        Ok(engine) => engine.gate_status(),
        Err(_) => spanda_engine::GateResult {
            score: 0.0,
            samples: 0,
            passed: false,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_same_model_repeated_is_stable_id() {
        let a = stable_id_for_model("qwen2-0.5b");
        let b = stable_id_for_model("qwen2-0.5b");
        assert_eq!(a, b);
    }

    #[test]
    fn test_different_models_get_different_ids() {
        let a = stable_id_for_model("qwen2-0.5b");
        let b = stable_id_for_model("llama-3-8b");
        assert_ne!(a, b);
    }

    #[test]
    fn test_gate_status_accessible_before_any_traffic() {
        // Fresh process state isn't guaranteed in a shared test binary (global
        // OnceLock), so just assert this doesn't panic and returns a well-formed
        // result either way.
        let status = current_gate_status();
        assert!(status.score >= 0.0 && status.score <= 1.0);
    }
}
