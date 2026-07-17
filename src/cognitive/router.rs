use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
pub enum RouteProfile {
    FastPath, // Early exits / smaller quantized model profile
    DeepPath, // Deep reasoning / multi-hop goal execution profile
}

pub struct ModelRouter;

impl ModelRouter {
    /// Dynamically route prompt queries to either FastPath or DeepPath execution layers.
    /// Incorporates dynamic benchmark-based latency feedback loop when `analytics` store is present.
    pub fn determine_route(
        prompt: &str,
        average_exit_layer: usize,
        analytics: Option<&crate::cognitive::analytics::AnalyticsStore>,
    ) -> (RouteProfile, String) {
        // Step 1: Query benchmark history to check for SLA breaches
        if let Some(store) = analytics
            && let Ok(avg_latency) = store.get_average_latency_ms() {
                // If historical latency exceeds our 200.0ms SLA target, force FastPath routing
                // to maintain low-latency responsiveness.
                if avg_latency > 200.0 {
                    let reason = format!(
                        "Benchmark SLA breached ({:.2}ms > 200.0ms) - dynamically routing to FastPath variant",
                        avg_latency
                    );
                    return (RouteProfile::FastPath, reason);
                }
            }

        let prompt_lower = prompt.to_lowercase();

        // Verb classification heuristics
        let complex_reasoning = prompt_lower.contains("compare")
            || prompt_lower.contains("why")
            || prompt_lower.contains("explain")
            || prompt_lower.contains("analyze")
            || prompt_lower.contains("difference")
            || prompt_lower.contains("how do i")
            || prompt_lower.contains("what is the relation");

        let long_prompt = prompt.split_whitespace().count() > 18;

        // Feedback-loop trigger: if past exit layer count average is very high,
        // it indicates high intrinsic complexity, so route to DeepPath!
        let high_layer_feedback = average_exit_layer >= 8;

        if complex_reasoning || long_prompt || high_layer_feedback {
            let reason = if complex_reasoning {
                "Heuristic identified reasoning-heavy keywords".to_string()
            } else if long_prompt {
                "Prompt length exceeds FastPath token threshold".to_string()
            } else {
                "Analytics store reports high layer depth bounds".to_string()
            };
            (RouteProfile::DeepPath, reason)
        } else {
            (
                RouteProfile::FastPath,
                "Prompt qualifies for standard FastPath early-exits".to_string(),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cognitive::analytics::{AnalyticsStore, QueryTrace};

    #[test]
    fn test_model_router_complexity_routing_rules() {
        // 1. Simple direct prompt -> FastPath
        let (r1, reason1) = ModelRouter::determine_route("What is 2+2?", 3, None);
        assert_eq!(r1, RouteProfile::FastPath);
        assert!(reason1.contains("FastPath"));

        // 2. Reasoning prompt -> DeepPath
        let (r2, reason2) = ModelRouter::determine_route(
            "Explain the core differences between IVF and HNSW indices.",
            3,
            None,
        );
        assert_eq!(r2, RouteProfile::DeepPath);
        assert!(reason2.contains("reasoning-heavy"));

        // 3. Simple prompt but analytics history indicates high layer exit depth -> DeepPath
        let (r3, reason3) = ModelRouter::determine_route("Retrieve it.", 10, None);
        assert_eq!(r3, RouteProfile::DeepPath);
        assert!(reason3.contains("Analytics store"));
    }

    #[test]
    fn test_benchmark_based_routing_sla_fallback() {
        let db_file = "storage/test_router_analytics.db";
        let _ = std::fs::remove_file(db_file);

        let store = AnalyticsStore::new_with_path(db_file);

        // Log a slow trace that breaches the 200ms SLA
        let slow_trace = QueryTrace {
            id: None,
            query_string: "heavy math query".to_string(),
            retrieval_ms: 100.0,
            rerank_ms: 50.0,
            inference_ms: 150.0, // Total = 300ms (> 200ms SLA)
            cache_hit: false,
            exit_layer: 12,
            timestamp_ms: 1000,
        };
        store.log_trace(slow_trace).unwrap();

        // Even though this is a complex reasoning query that would normally go to DeepPath,
        // it should be routed to FastPath to protect the latency SLA because avg latency is 300ms!
        let (r, reason) =
            ModelRouter::determine_route("Explain why the universe is expanding.", 3, Some(&store));

        let _ = std::fs::remove_file(db_file);

        assert_eq!(r, RouteProfile::FastPath);
        assert!(reason.contains("SLA breached"), "Actual reason: {}", reason);
    }
}
