use crate::planner::cost_model::CostModel;
use crate::planner::policy::{PlannerDecision, PlannerPolicy};
use crate::storage::storage_manifest::StorageTier;

pub struct ExecutionPathOptimizer;

impl ExecutionPathOptimizer {
    /// Select the optimal execution pathway for the given query parameters
    pub fn optimize(
        policy: &PlannerPolicy,
        prompt: &str,
        _model_name: &str,
        _context_chunks: &[(String, String)],
        has_cache: bool,
        historical_accept_rate: f32,
        spanda_healthy: bool,
        route_confidences: &std::collections::HashMap<String, f32>,
        has_activation_view: bool,
    ) -> PlannerDecision {
        // Step 1: Evaluate policy-level exact-only bypasses
        if policy.planner_mode == "exact_only" {
            return PlannerDecision::ExactDecode;
        }

        // Step 2: Deterministic cache hit has absolute priority (sub-millisecond O(1) latency)
        if has_cache {
            return PlannerDecision::CachedAnswer;
        }

        // If an activation view exists, the prefill compute is near zero.
        // We override speculative pathways and use ExactDecode to securely restore the branch deterministically.
        if has_activation_view {
            return PlannerDecision::ExactDecode;
        }

        let complexity = 1.0;
        let exact_cost = CostModel::estimate_path_cost(
            prompt,
            50,
            PlannerDecision::ExactDecode,
            complexity,
            historical_accept_rate,
            has_activation_view,
        );
        let spec_cost = CostModel::estimate_path_cost(
            prompt,
            50,
            PlannerDecision::SpeculativeDecode,
            complexity,
            historical_accept_rate,
            has_activation_view,
        );
        let spanda_cost = CostModel::estimate_path_cost(
            prompt,
            50,
            PlannerDecision::SpandaSparse,
            complexity,
            historical_accept_rate,
            has_activation_view,
        );

        let mut best_decision = PlannerDecision::ExactDecode;
        let mut min_cost = exact_cost;

        // Check speculative decoding eligibility based on policy thresholds and adaptive route confidence
        let spec_confidence = route_confidences
            .get("SpeculativeDecode")
            .copied()
            .unwrap_or(0.5);
        let spec_eligible =
            historical_accept_rate >= policy.min_speculative_accept_rate && spec_confidence >= 0.3;

        if spec_eligible && spec_cost < exact_cost {
            best_decision = PlannerDecision::SpeculativeDecode;
            min_cost = spec_cost;
        }

        // Check SPANDA graceful degradation state machine with adaptive confidence
        let spanda_confidence = route_confidences
            .get("SpandaSparse")
            .copied()
            .unwrap_or(0.5);
        if spanda_healthy && policy.planner_mode == "spanda" && spanda_confidence >= 0.2 {
            if best_decision == PlannerDecision::SpeculativeDecode {
                if spanda_cost < min_cost {
                    best_decision = PlannerDecision::SpandaSparse;
                }
            } else {
                best_decision = PlannerDecision::SpandaSparse;
            }
        }

        best_decision
    }

    /// Select the target storage tier for a specific layer during inference planning
    pub fn route_layer_tier(
        layer_idx: usize,
        total_layers: usize,
        access_count: u64,
        last_accessed: u64,
        current_time: u64,
    ) -> StorageTier {
        let planner_tier_aware_feature = cfg!(feature = "planner_tier_aware");
        let planner_tier_aware_env = std::env::var("BRAMHA_PLANNER_TIER_AWARE")
            .map(|v| v.trim().to_lowercase() != "false")
            .unwrap_or(true);

        if !planner_tier_aware_feature || !planner_tier_aware_env {
            return StorageTier::Critical;
        }

        CostModel::tier_preference(
            layer_idx,
            total_layers,
            access_count,
            last_accessed,
            current_time,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimizer_decision_flow() {
        let policy = PlannerPolicy {
            planner_mode: "auto".to_string(),
            min_speculative_accept_rate: 0.5,
            max_cached_age_seconds: 86400,
        };

        let confidences = std::collections::HashMap::new();

        // 1. Hit cache -> CachedAnswer
        let decision = ExecutionPathOptimizer::optimize(
            &policy,
            "prompt",
            "model",
            &[],
            true,
            0.9,
            true,
            &confidences,
            false,
        );
        assert_eq!(decision, PlannerDecision::CachedAnswer);

        // 2. Miss cache, high accept rate -> SpeculativeDecode
        let decision = ExecutionPathOptimizer::optimize(
            &policy,
            "prompt",
            "model",
            &[],
            false,
            0.7,
            true,
            &confidences,
            false,
        );
        assert_eq!(decision, PlannerDecision::SpeculativeDecode);

        // 3. Miss cache, low accept rate, healthy SPANDA -> SpandaSparse (needs mode = "spanda")
        let spanda_policy = PlannerPolicy {
            planner_mode: "spanda".to_string(),
            ..policy.clone()
        };
        let decision = ExecutionPathOptimizer::optimize(
            &spanda_policy,
            "prompt",
            "model",
            &[],
            false,
            0.3,
            true,
            &confidences,
            false,
        );
        assert_eq!(decision, PlannerDecision::SpandaSparse);

        // 4. Miss cache, low accept rate, unhealthy SPANDA -> ExactDecode
        let decision = ExecutionPathOptimizer::optimize(
            &spanda_policy,
            "prompt",
            "model",
            &[],
            false,
            0.3,
            false,
            &confidences,
            false,
        );
        assert_eq!(decision, PlannerDecision::ExactDecode);

        // 5. Force exact_only policy override -> ExactDecode
        let exact_policy = PlannerPolicy {
            planner_mode: "exact_only".to_string(),
            ..policy
        };
        let decision = ExecutionPathOptimizer::optimize(
            &exact_policy,
            "prompt",
            "model",
            &[],
            true,
            0.95,
            true,
            &confidences,
            false,
        );
        assert_eq!(decision, PlannerDecision::ExactDecode);

        // 6. Activation View override tests
        let view_decision = ExecutionPathOptimizer::optimize(
            &policy,
            "long prompt here",
            "model",
            &[],
            false,
            0.9,
            true,
            &confidences,
            true,
        );
        assert_eq!(
            view_decision,
            PlannerDecision::ExactDecode,
            "Should override to ExactDecode when view exists because cost is cheaper"
        );
    }

    #[test]
    fn test_planner_tier_aware_bypass() {
        let current_time = 10000;
        let total_layers = 24;

        // With default/enabled tier aware, middle layer should be Important (Warm)
        let tier_default =
            ExecutionPathOptimizer::route_layer_tier(10, total_layers, 1, 9900, current_time);
        assert_eq!(tier_default, StorageTier::Important);

        // Set env var to false to bypass/disable tier routing
        unsafe {
            std::env::set_var("BRAMHA_PLANNER_TIER_AWARE", "false");
        }

        let tier_bypassed =
            ExecutionPathOptimizer::route_layer_tier(10, total_layers, 1, 9900, current_time);

        unsafe {
            std::env::remove_var("BRAMHA_PLANNER_TIER_AWARE");
        }

        // Bypassed tier routing should default all layers to Critical (DRAM)
        assert_eq!(tier_bypassed, StorageTier::Critical);
    }
}
