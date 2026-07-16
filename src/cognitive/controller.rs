use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct RuntimeProfile {
    pub nprobe: usize,
    pub top_k: usize,
    pub rerank_depth: usize,
    pub hybrid_alpha: f32,
    pub cache_ttl_sec: u64,
    pub prefetch_depth: usize,
}

pub struct AdaptiveController {
    pub latency_target_ms: f64,
    pub min_bounds: RuntimeProfile,
    pub max_bounds: RuntimeProfile,
    pub current_profile: RuntimeProfile,
    pub workflows: std::collections::HashMap<String, WorkflowGraph>,
}

impl AdaptiveController {
    pub fn new(latency_target_ms: f64) -> Self {
        let default_profile = RuntimeProfile {
            nprobe: 16,
            top_k: 10,
            rerank_depth: 5,
            hybrid_alpha: 0.5,
            cache_ttl_sec: 7200,
            prefetch_depth: 3,
        };

        let min_bounds = RuntimeProfile {
            nprobe: 2,
            top_k: 2,
            rerank_depth: 2,
            hybrid_alpha: 0.0,
            cache_ttl_sec: 300,
            prefetch_depth: 1,
        };

        let max_bounds = RuntimeProfile {
            nprobe: 64,
            top_k: 50,
            rerank_depth: 20,
            hybrid_alpha: 1.0,
            cache_ttl_sec: 86400,
            prefetch_depth: 10,
        };

        AdaptiveController {
            latency_target_ms,
            min_bounds,
            max_bounds,
            current_profile: default_profile,
            workflows: std::collections::HashMap::new(),
        }
    }

    /// Dynamically adjust parameter values using historical query feedback loop
    pub fn adapt_parameters(&mut self, average_latency_ms: f64) {
        if average_latency_ms <= 0.0 {
            return;
        }

        // If system runs too slowly, dial down search parameters to speed up execution
        if average_latency_ms > self.latency_target_ms {
            println!(
                "⚠️ Latency target exceeded ({:.1} ms > {:.1} ms). Down-tuning parameters for maximum throughput...",
                average_latency_ms, self.latency_target_ms
            );

            if self.current_profile.nprobe > self.min_bounds.nprobe {
                self.current_profile.nprobe =
                    (self.current_profile.nprobe - 2).max(self.min_bounds.nprobe);
            }
            if self.current_profile.rerank_depth > self.min_bounds.rerank_depth {
                self.current_profile.rerank_depth =
                    (self.current_profile.rerank_depth - 1).max(self.min_bounds.rerank_depth);
            }
            if self.current_profile.top_k > self.min_bounds.top_k {
                self.current_profile.top_k =
                    (self.current_profile.top_k - 1).max(self.min_bounds.top_k);
            }
        }
        // If we are well within budget, dial up retrieval settings to maximize grounding quality
        else if average_latency_ms < self.latency_target_ms * 0.7 {
            println!(
                "🚀 Latency budget is highly healthy ({:.1} ms < {:.1} ms). Up-tuning parameters to optimize grounding quality...",
                average_latency_ms, self.latency_target_ms
            );

            if self.current_profile.nprobe < self.max_bounds.nprobe {
                self.current_profile.nprobe =
                    (self.current_profile.nprobe + 2).min(self.max_bounds.nprobe);
            }
            if self.current_profile.rerank_depth < self.max_bounds.rerank_depth {
                self.current_profile.rerank_depth =
                    (self.current_profile.rerank_depth + 1).min(self.max_bounds.rerank_depth);
            }
            if self.current_profile.top_k < self.max_bounds.top_k {
                self.current_profile.top_k =
                    (self.current_profile.top_k + 1).min(self.max_bounds.top_k);
            }
        }
    }

    /// Safely apply manual profiles while enforcing strictly configured boundaries
    pub fn force_profile(&mut self, custom: RuntimeProfile) {
        self.current_profile = RuntimeProfile {
            nprobe: custom
                .nprobe
                .clamp(self.min_bounds.nprobe, self.max_bounds.nprobe),
            top_k: custom
                .top_k
                .clamp(self.min_bounds.top_k, self.max_bounds.top_k),
            rerank_depth: custom
                .rerank_depth
                .clamp(self.min_bounds.rerank_depth, self.max_bounds.rerank_depth),
            hybrid_alpha: custom
                .hybrid_alpha
                .clamp(self.min_bounds.hybrid_alpha, self.max_bounds.hybrid_alpha),
            cache_ttl_sec: custom
                .cache_ttl_sec
                .clamp(self.min_bounds.cache_ttl_sec, self.max_bounds.cache_ttl_sec),
            prefetch_depth: custom.prefetch_depth.clamp(
                self.min_bounds.prefetch_depth,
                self.max_bounds.prefetch_depth,
            ),
        };
    }

    pub fn parse_feedback_event(&mut self, event: FeedbackEvent) {
        if event.success {
            self.adapt_parameters(self.latency_target_ms * 0.5); // Tune up quality
        } else {
            self.adapt_parameters(self.latency_target_ms * 1.5); // Dial down for speed
        }
    }

    pub fn save_workflow(&mut self, workflow: WorkflowGraph) {
        self.workflows.insert(workflow.id.clone(), workflow);
    }

    pub fn get_workflow(&self, id: &str) -> Option<WorkflowGraph> {
        self.workflows.get(id).cloned()
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FeedbackEvent {
    pub workflow_id: String,
    pub success: bool,
    pub latency_ms: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct WorkflowGraph {
    pub id: String,
    pub steps: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_controller_adaptation_loops() {
        let mut controller = AdaptiveController::new(100.0);
        let start_profile = controller.current_profile.clone();

        // 1. Trigger Slowdown (should decrease nprobe / rerank_depth)
        controller.adapt_parameters(150.0);
        assert!(controller.current_profile.nprobe < start_profile.nprobe);
        assert!(controller.current_profile.rerank_depth < start_profile.rerank_depth);

        // 2. Trigger Quality boost (should increase parameter values back)
        let slow_profile = controller.current_profile.clone();
        controller.adapt_parameters(40.0);
        assert!(controller.current_profile.nprobe > slow_profile.nprobe);

        // 3. Forced clamped override
        let bad_profile = RuntimeProfile {
            nprobe: 9999, // overflow bounds
            top_k: 10,
            rerank_depth: 5,
            hybrid_alpha: 0.5,
            cache_ttl_sec: 100,
            prefetch_depth: 2,
        };
        controller.force_profile(bad_profile);
        assert_eq!(controller.current_profile.nprobe, 64); // clamped to max_bounds nprobe!
    }
}
