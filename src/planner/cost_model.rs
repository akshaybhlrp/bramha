use crate::planner::policy::PlannerDecision;
use crate::storage::storage_manifest::StorageTier;
use std::sync::{Mutex, OnceLock};

#[derive(Clone, Debug)]
pub struct CostModelParams {
    pub exact_multiplier: f32,
    pub speculative_multiplier: f32,
    pub spanda_multiplier: f32,
    pub cache_multiplier: f32,
}

impl Default for CostModelParams {
    fn default() -> Self {
        Self {
            exact_multiplier: 1.0,
            speculative_multiplier: 1.0,
            spanda_multiplier: 1.0,
            cache_multiplier: 1.0,
        }
    }
}

impl CostModelParams {
    pub fn global() -> &'static Mutex<Self> {
        static INSTANCE: OnceLock<Mutex<CostModelParams>> = OnceLock::new();
        INSTANCE.get_or_init(|| Mutex::new(CostModelParams::default()))
    }
}

pub struct CostModel;

impl CostModel {
    /// Recalibrate cost parameters using the statistics logged in MetadataSqlStore
    pub fn recalibrate_from_analytics(analytics: &crate::storage::metadata_sql::MetadataSqlStore) {
        let mut params = CostModelParams::global().lock().unwrap();

        if let Ok(Some(avg_exact)) = analytics.get_route_average_latency("ExactDecode") {
            let baseline = 615.0; // typical exact decode time baseline
            params.exact_multiplier = (avg_exact as f32 / baseline).clamp(0.1, 10.0);
        }

        if let Ok(Some(avg_spec)) = analytics.get_route_average_latency("SpeculativeDecode") {
            let baseline = 256.0; // typical speculative decode baseline
            params.speculative_multiplier = (avg_spec as f32 / baseline).clamp(0.1, 10.0);
        }

        if let Ok(Some(avg_spanda)) = analytics.get_route_average_latency("SpandaSparse") {
            let baseline = 707.0; // typical spanda decode baseline
            params.spanda_multiplier = (avg_spanda as f32 / baseline).clamp(0.1, 10.0);
        }

        if let Ok(Some(avg_cache)) = analytics.get_route_average_latency("CachedAnswer") {
            let baseline = 0.5; // typical cache hit baseline
            params.cache_multiplier = (avg_cache as f32 / baseline).clamp(0.1, 10.0);
        }
    }

    /// Calculate estimated latency in milliseconds for a chosen execution path
    pub fn estimate_path_cost(
        prompt: &str,
        max_new_tokens: usize,
        path: PlannerDecision,
        model_complexity: f32, // complexity indicator (e.g., parameter count, layer depth scale)
        historical_accept_rate: f32, // historical speculative accept rate [0.0, 1.0]
        has_activation_view: bool,
    ) -> f32 {
        let params = CostModelParams::global().lock().unwrap().clone();

        match path {
            PlannerDecision::CachedAnswer => {
                // Retrieving a response from the deterministic cache is a constant O(1) hash map lookup
                0.5f32 * params.cache_multiplier
            }
            PlannerDecision::ExactDecode => {
                // Standard generation latency combines prompt ingestion overhead and sequential token decode passes
                // If an activation view is available, prompt ingestion time is bypassed.
                let prompt_ingestion_cost = if has_activation_view {
                    0.5f32
                } else {
                    prompt.len() as f32 * 0.15
                };
                let decode_cost = max_new_tokens as f32 * 12.0 * model_complexity;
                (prompt_ingestion_cost + decode_cost) * params.exact_multiplier
            }
            PlannerDecision::SpeculativeDecode => {
                // Speculative decoding expected speedup scales with the historical speculation acceptance rate
                let prompt_ingestion_cost = if has_activation_view {
                    0.5f32
                } else {
                    prompt.len() as f32 * 0.15
                };
                let decode_cost = max_new_tokens as f32 * 12.0 * model_complexity;
                let base_exact_cost = prompt_ingestion_cost + decode_cost;
                // Expected speedup factor ranges from 1.0x (accept_rate = 0) up to ~3.0x (accept_rate = 1.0)
                let speedup_factor = 1.0 + (historical_accept_rate * 2.0);
                (base_exact_cost / speedup_factor) * params.speculative_multiplier
            }
            PlannerDecision::SpandaSparse => {
                // Spanda introduces a 15% P99 overhead vs baseline
                let prompt_ingestion_cost = if has_activation_view {
                    0.5f32
                } else {
                    prompt.len() as f32 * 0.15
                };
                let decode_cost = max_new_tokens as f32 * 12.0 * model_complexity;
                let base_exact_cost = prompt_ingestion_cost + decode_cost;
                (base_exact_cost * 1.15) * params.spanda_multiplier
            }
        }
    }

    /// Select storage tier based on access pattern rules (BRM-S9-002)
    pub fn tier_preference(
        layer_idx: usize,
        total_layers: usize,
        access_count: u64,
        last_accessed: u64,
        current_time: u64,
    ) -> StorageTier {
        // Hot tier: layers 0, 1, final 2 layers always resident
        if layer_idx == 0 || layer_idx == 1 || layer_idx >= total_layers.saturating_sub(2) {
            return StorageTier::Critical;
        }

        // Unused variants / demoted after 1 hour idle
        let idle_time = current_time.saturating_sub(last_accessed);
        if last_accessed > 0 && idle_time >= 3600 {
            return StorageTier::Redundant;
        }

        // Warm tier: middle layers, promoted on second access within 5 minutes
        if access_count >= 2 && idle_time <= 300 {
            return StorageTier::Critical;
        }

        // Warm tier default
        StorageTier::Important
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    static TEST_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn test_cost_model_relative_ordering() {
        let _guard = TEST_MUTEX.lock().unwrap();
        {
            let mut params = CostModelParams::global().lock().unwrap();
            *params = CostModelParams::default();
        }
        let prompt = "Explain the heterogeneous scheduler in Bramha.";
        let max_tokens = 50;
        let complexity = 1.0;

        let cache_cost = CostModel::estimate_path_cost(
            prompt,
            max_tokens,
            PlannerDecision::CachedAnswer,
            complexity,
            0.0,
            false,
        );
        let exact_cost = CostModel::estimate_path_cost(
            prompt,
            max_tokens,
            PlannerDecision::ExactDecode,
            complexity,
            0.0,
            false,
        );

        // Cache cost must be near-zero and vastly cheaper than exact decode
        assert!(cache_cost < 1.0);
        assert!(exact_cost > cache_cost);

        // Speculative with low accept rate should be close to exact
        let spec_bad = CostModel::estimate_path_cost(
            prompt,
            max_tokens,
            PlannerDecision::SpeculativeDecode,
            complexity,
            0.0,
            false,
        );
        assert!((spec_bad - exact_cost).abs() < 1e-4);

        // Speculative with high accept rate should show a major cost reduction
        let spec_good = CostModel::estimate_path_cost(
            prompt,
            max_tokens,
            PlannerDecision::SpeculativeDecode,
            complexity,
            0.8,
            false,
        );
        assert!(spec_good < exact_cost);
        assert!(spec_good > cache_cost);

        // Activation view should drastically reduce exact decode cost (simulating zero prefill)
        let exact_cost_with_view = CostModel::estimate_path_cost(
            prompt,
            max_tokens,
            PlannerDecision::ExactDecode,
            complexity,
            0.0,
            true,
        );
        assert!(exact_cost_with_view < exact_cost);
    }

    #[test]
    fn test_tier_preference() {
        let current_time = 10000;
        let total_layers = 24;

        // Hot tier for first two and last two
        assert_eq!(
            CostModel::tier_preference(0, total_layers, 0, 0, current_time),
            StorageTier::Critical
        );
        assert_eq!(
            CostModel::tier_preference(1, total_layers, 0, 0, current_time),
            StorageTier::Critical
        );
        assert_eq!(
            CostModel::tier_preference(22, total_layers, 0, 0, current_time),
            StorageTier::Critical
        );
        assert_eq!(
            CostModel::tier_preference(23, total_layers, 0, 0, current_time),
            StorageTier::Critical
        );

        // Middle layer default warm
        assert_eq!(
            CostModel::tier_preference(10, total_layers, 1, 9900, current_time),
            StorageTier::Important
        );

        // Middle layer promoted to hot on second access within 5 minutes
        assert_eq!(
            CostModel::tier_preference(10, total_layers, 2, 9800, current_time),
            StorageTier::Critical
        );

        // Middle layer cold after 1 hour idle
        assert_eq!(
            CostModel::tier_preference(10, total_layers, 1, 5000, current_time),
            StorageTier::Redundant
        );
    }

    #[test]
    fn test_cost_model_recalibration_and_dynamic_routing() {
        let _guard = TEST_MUTEX.lock().unwrap();
        {
            let mut params = CostModelParams::global().lock().unwrap();
            *params = CostModelParams::default();
        }
        let db_file = "storage/test_cost_model_recal.db";
        let _ = std::fs::remove_file(db_file);

        let store = crate::storage::metadata_sql::MetadataSqlStore::new_with_path(db_file);

        // Before recalibration, assert multipliers are 1.0 (default)
        {
            let params = CostModelParams::global().lock().unwrap().clone();
            assert_eq!(params.exact_multiplier, 1.0);
            assert_eq!(params.speculative_multiplier, 1.0);
        }

        // Log very high latency for SpeculativeDecode (e.g. 5000 ms, whereas baseline is 256.0 ms)
        store
            .update_route_quality("SpeculativeDecode", 5000.0, true)
            .unwrap();
        // Log low latency for ExactDecode (e.g. 100 ms)
        store
            .update_route_quality("ExactDecode", 100.0, true)
            .unwrap();

        // Perform recalibration
        CostModel::recalibrate_from_analytics(&store);

        // Verify multipliers have changed
        {
            let params = CostModelParams::global().lock().unwrap().clone();
            assert!(params.speculative_multiplier > 1.0);
            assert!(params.exact_multiplier < 1.0);
        }

        // Verify that cost of SpeculativeDecode now exceeds ExactDecode
        let prompt = "Explain cost model recalibration.";
        let spec_cost = CostModel::estimate_path_cost(
            prompt,
            50,
            PlannerDecision::SpeculativeDecode,
            1.0,
            0.9,
            false,
        );
        let exact_cost = CostModel::estimate_path_cost(
            prompt,
            50,
            PlannerDecision::ExactDecode,
            1.0,
            0.9,
            false,
        );

        assert!(
            spec_cost > exact_cost,
            "Speculative cost ({} ms) should exceed exact cost ({} ms) due to high actual logged latency",
            spec_cost,
            exact_cost
        );

        // Cleanup
        let _ = std::fs::remove_file(db_file);

        // Reset multipliers to default for other tests
        {
            let mut params = CostModelParams::global().lock().unwrap();
            *params = CostModelParams::default();
        }
    }
}
