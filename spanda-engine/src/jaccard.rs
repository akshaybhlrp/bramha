//! Phase 0 gate: the SPANDA plan requires a measured access-pattern predictability
//! number before any paging/eviction/prefetch phase is allowed to run. Before this
//! module, that gate did not exist anywhere in the codebase (`grep -r jaccard` was
//! empty across both bramha-engine and spanda-engine) despite doc comments claiming
//! later phases were implemented. This computes the real number from observed
//! per-step access sets, using Jaccard similarity between consecutive sets as the
//! predictability signal.

use std::collections::{HashSet, VecDeque};

/// Minimum average Jaccard similarity between consecutive access windows required
/// to consider access patterns "predictable enough" to justify query-conditional
/// paging + confidence-based eviction. Below this, paging overhead isn't justified
/// by predictability — callers should fall back to loading everything densely.
pub const JACCARD_PASS_THRESHOLD: f32 = 0.55;

/// Minimum number of observed access-set transitions before the gate can be
/// evaluated at all. Mirrors the shadow-scan pattern already used elsewhere in
/// bramha-engine (>=20 samples before a gate decision is trusted).
pub const MIN_SAMPLES: usize = 20;

/// Rolling window size for aggregation, to bound memory growth over a long-running
/// process instead of accumulating similarity samples forever.
const WINDOW_CAPACITY: usize = 500;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GateResult {
    pub score: f32,
    pub samples: usize,
    pub passed: bool,
}

/// Tracks which page/tensor ids were accessed on each inference step and computes
/// a rolling Jaccard-similarity-based predictability number across consecutive steps.
pub struct AccessTracker {
    previous: Option<HashSet<u64>>,
    similarities: VecDeque<f32>,
}

impl Default for AccessTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl AccessTracker {
    pub fn new() -> Self {
        AccessTracker {
            previous: None,
            similarities: VecDeque::with_capacity(WINDOW_CAPACITY),
        }
    }

    /// Record the set of page/tensor ids touched during one inference step
    /// (e.g. one decode step, or one forward pass through the active layers).
    pub fn record_access(&mut self, active_ids: &[u64]) {
        let current: HashSet<u64> = active_ids.iter().copied().collect();

        if let Some(ref prev) = self.previous {
            let intersection = prev.intersection(&current).count();
            let union = prev.union(&current).count();
            let jaccard = if union == 0 {
                // Both sets empty: define as fully predictable (a no-op step),
                // rather than an undefined 0/0.
                1.0
            } else {
                intersection as f32 / union as f32
            };

            if self.similarities.len() == WINDOW_CAPACITY {
                self.similarities.pop_front();
            }
            self.similarities.push_back(jaccard);
        }

        self.previous = Some(current);
    }

    /// Evaluate the Phase 0 gate: is the access pattern predictable enough to
    /// justify building paging/eviction/prefetch on top of it?
    pub fn evaluate_gate(&self) -> GateResult {
        let samples = self.similarities.len();
        if samples < MIN_SAMPLES {
            return GateResult {
                score: 0.0,
                samples,
                passed: false,
            };
        }
        let sum: f32 = self.similarities.iter().sum();
        let score = sum / samples as f32;
        GateResult {
            score,
            samples,
            passed: score >= JACCARD_PASS_THRESHOLD,
        }
    }

    pub fn reset(&mut self) {
        self.previous = None;
        self.similarities.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gate_fails_below_min_samples() {
        let mut tracker = AccessTracker::new();
        tracker.record_access(&[1, 2, 3]);
        tracker.record_access(&[1, 2, 3]);
        let result = tracker.evaluate_gate();
        assert!(!result.passed);
        assert_eq!(result.samples, 1);
    }

    #[test]
    fn test_gate_passes_on_stable_repeating_pattern() {
        let mut tracker = AccessTracker::new();
        // Same page set every step -> Jaccard = 1.0 every transition -> highly predictable.
        for _ in 0..(MIN_SAMPLES + 5) {
            tracker.record_access(&[10, 11, 12, 13]);
        }
        let result = tracker.evaluate_gate();
        assert!(result.passed);
        assert!((result.score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_gate_fails_on_fully_random_disjoint_pattern() {
        let mut tracker = AccessTracker::new();
        // Disjoint sets every step -> Jaccard = 0.0 every transition -> unpredictable.
        let mut next_id: u64 = 0;
        for _ in 0..(MIN_SAMPLES + 5) {
            let batch: Vec<u64> = (next_id..next_id + 4).collect();
            next_id += 4;
            tracker.record_access(&batch);
        }
        let result = tracker.evaluate_gate();
        assert!(!result.passed);
        assert!(result.score < JACCARD_PASS_THRESHOLD);
    }

    #[test]
    fn test_window_capacity_bounds_memory() {
        let mut tracker = AccessTracker::new();
        for i in 0..(WINDOW_CAPACITY + 50) {
            tracker.record_access(&[i as u64]);
        }
        assert!(tracker.similarities.len() <= WINDOW_CAPACITY);
    }

    #[test]
    fn test_reset_clears_state() {
        let mut tracker = AccessTracker::new();
        for _ in 0..(MIN_SAMPLES + 5) {
            tracker.record_access(&[1, 2, 3]);
        }
        assert!(tracker.evaluate_gate().passed);
        tracker.reset();
        assert_eq!(tracker.evaluate_gate().samples, 0);
    }
}
