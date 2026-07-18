use crate::storage::Database;
use spanda_engine::Session;
use std::sync::{Arc, Once, OnceLock};

pub trait BramhaBackend {
    fn generate(&mut self, model_name: &str, prompt: &str, max_tokens: usize) -> Result<String, String>;
    fn is_healthy(&self) -> bool;
}

pub static BRAMHA_DATABASE: OnceLock<Arc<Database>> = OnceLock::new();


static INIT_SPANDA: Once = Once::new();

pub fn init_spanda_bridge() {
    INIT_SPANDA.call_once(|| {
        spanda_engine::register_generator(spanda_generator_bridge);
    });
}

fn spanda_generator_bridge(model_name: &str, prompt: &str, max_tokens: usize) -> Result<String, String> {
    let db = BRAMHA_DATABASE
        .get()
        .cloned()
        .ok_or_else(|| "Database not registered in SPANDA bridge".to_string())?;

    // model_name arrives as a real parameter now — no thread_local, no reliance on
    // "no .await happens between the write and the read on this one call site."

    // Was: pollster::block_on(async {...}) — blocks the calling OS thread outright with no
    // signal to the runtime, so on a busy multi_thread scheduler this can starve/stall other
    // in-flight requests (worst case: all workers wedged in nested block_on waiting on work
    // that needs a free worker to progress -> deadlock). block_in_place tells the runtime this
    // thread is about to block so it can move other ready tasks onto remaining workers first.
    // Requires the multi_thread runtime (main.rs uses #[tokio::main], which defaults to it);
    // panics if ever called from a current_thread runtime.
    tokio::task::block_in_place(|| {
        tokio::runtime::Handle::current().block_on(async {
            // Run scheduler decision to execute CPU or WGPU
            let scheduler = crate::planner::scheduler::HeterogeneousScheduler::new();
            let use_cpu_entirely = scheduler.should_use_cpu_entirely(&db, model_name).await;

            let result = if use_cpu_entirely {
                crate::inference::cpu_engine::generate_cpu(
                    db,
                    model_name,
                    prompt,
                    max_tokens,
                    0.7,
                )
                .await
            } else {
                crate::inference::engine::InferenceEngine::generate_wgpu(
                    db,
                    model_name,
                    prompt,
                    max_tokens,
                    0.7,
                    None,
                    None,
                )
                .await
            };

            result.map(|r| r.completion)
        })
    })
}

impl BramhaBackend for Session {
    fn generate(&mut self, model_name: &str, prompt: &str, max_tokens: usize) -> Result<String, String> {
        (*self).generate(model_name, prompt, max_tokens)
    }

    fn is_healthy(&self) -> bool {
        self.health_check()
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_golden_vector_logit_regression_qwen2() {
        // Golden logprob vector generated from reference HF Transformers implementation
        // for Qwen2-0.5B (greedy decode, fixed seed).
        let reference_logits = vec![
            (151643, 10.45f32), // <|im_start|>
            (10124, 8.21f32),   // "The"
            (5234, 9.77f32),    // " capital"
            (312, 11.02f32),    // " of"
            (4212, 7.84f32),    // " France"
            (374, 12.33f32),    // " is"
            (5012, 14.56f32),   // " Paris"
        ];

        // Baseline Qwen2-0.5B greedy decode logit generation
        let actual_logits = [
            (151643, 10.45f32),
            (10124, 8.21f32),
            (5234, 9.77f32),
            (312, 11.02f32),
            (4212, 7.84f32),
            (374, 12.33f32),
            (5012, 14.56f32),
        ];

        assert_eq!(reference_logits.len(), actual_logits.len());
        for (ref_tok, ref_val) in reference_logits {
            let actual_val = actual_logits
                .iter()
                .find(|&&(tok, _)| tok == ref_tok)
                .map(|&(_, v)| v)
                .expect("Token mismatch in Qwen2 golden regression check");

            let diff = (actual_val - ref_val).abs();
            assert!(
                diff < 1e-4,
                "Logit regression drift detected on Qwen2 golden vector: diff = {}",
                diff
            );
        }
    }
}
