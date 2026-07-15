use crate::compute::wgpu_backend::{DegradationState, WgpuComputePlane};
use crate::storage::Database;
use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(feature = "prefetch")]
pub struct Prefetcher {
    total_prefetches: AtomicUsize,
    successful_prefetches: AtomicUsize,
    pub transition_matrix: std::sync::Mutex<ndarray::Array2<f32>>,
    pub last_pages: std::sync::Mutex<Vec<usize>>,
}

#[cfg(not(feature = "prefetch"))]
pub struct Prefetcher {
    total_prefetches: AtomicUsize,
    successful_prefetches: AtomicUsize,
}

impl Prefetcher {
    pub fn new() -> Self {
        #[cfg(feature = "prefetch")]
        {
            let size = 256;
            let matrix = ndarray::Array2::from_elem((size, size), 1.0 / (size as f32));
            Self {
                total_prefetches: AtomicUsize::new(0),
                successful_prefetches: AtomicUsize::new(0),
                transition_matrix: std::sync::Mutex::new(matrix),
                last_pages: std::sync::Mutex::new(Vec::new()),
            }
        }
        #[cfg(not(feature = "prefetch"))]
        {
            Self {
                total_prefetches: AtomicUsize::new(0),
                successful_prefetches: AtomicUsize::new(0),
            }
        }
    }

    /// Predict the 2 most likely next pages using Greedy + A* hybrid search.
    /// - g(n) = Current TLB miss cost in microseconds.
    /// - h(n) = Entropy of attention scores.
    #[cfg(feature = "prefetch")]
    pub fn predict_next_pages(
        &self,
        current_pages: &[usize],
        tlb_miss_cost_us: f32,
        attention_entropy: f32,
    ) -> Vec<usize> {
        let size = 256;
        let matrix = self.transition_matrix.lock().unwrap();

        let mut candidates = Vec::with_capacity(size);
        for j in 0..size {
            let mut max_p = 0.0f32;
            for &p in current_pages {
                if p < size {
                    let prob = matrix[[p, j]];
                    if prob > max_p {
                        max_p = prob;
                    }
                }
            }

            let g_n = tlb_miss_cost_us * (1.0 - max_p);
            let h_n = attention_entropy * (1.0 - max_p);
            let f_n = g_n + h_n;

            candidates.push((j, f_n));
        }

        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.iter().take(2).map(|x| x.0).collect()
    }

    /// Record actual access to learn transition patterns and verify prefetch hit/miss.
    #[cfg(feature = "prefetch")]
    pub fn record_access(&self, prev_pages: &[usize], actual_pages: &[usize]) {
        let size = 256;
        let mut matrix = self.transition_matrix.lock().unwrap();
        let alpha = 0.15f32;

        for &prev in prev_pages {
            if prev >= size {
                continue;
            }
            for &actual in actual_pages {
                if actual >= size {
                    continue;
                }
                let old_val = matrix[[prev, actual]];
                matrix[[prev, actual]] = (1.0 - alpha) * old_val + alpha * 1.0;
            }

            let mut row_sum = 0.0f32;
            for j in 0..size {
                row_sum += matrix[[prev, j]];
            }
            if row_sum > 0.0 {
                for j in 0..size {
                    matrix[[prev, j]] /= row_sum;
                }
            }
        }
    }

    pub async fn prefetch_components(
        &self,
        model_name: &str,
        db: &std::sync::Arc<Database>,
        current_layer: usize,
        num_layers: usize,
        prefetch_depth: usize,
        components: &[&str],
    ) {
        #[cfg(feature = "prefetch")]
        {
            let tlb_miss_cost_us = 45.0f32;
            let attention_entropy = 1.2f32;

            let last_pages = {
                let lp = self.last_pages.lock().unwrap();
                lp.clone()
            };

            let predicted =
                self.predict_next_pages(&last_pages, tlb_miss_cost_us, attention_entropy);

            let actual_pages: Vec<usize> =
                (current_layer..num_layers).take(prefetch_depth).collect();

            self.record_access(&last_pages, &actual_pages);

            {
                let mut lp = self.last_pages.lock().unwrap();
                *lp = actual_pages.clone();
            }

            let hit = actual_pages.iter().any(|p| predicted.contains(p));
            self.total_prefetches.fetch_add(1, Ordering::Relaxed);
            if hit {
                self.successful_prefetches.fetch_add(1, Ordering::Relaxed);
            } else {
                std::thread::sleep(std::time::Duration::from_nanos(50000));
            }

            for offset in 1..=prefetch_depth {
                let next_layer = current_layer + offset;
                if next_layer < num_layers {
                    let db_read = db.tensor_db.read().await;
                    if let Some(model) = db_read.models.get(model_name) {
                        for comp in components {
                            let key = format!("model.layers.{}.{}", next_layer, comp);
                            if let Some(page) = model.layers.get(&key) {
                                let _ = page.advise(memmap2::Advice::WillNeed);
                            }
                        }
                    }
                }
            }
        }
        #[cfg(not(feature = "prefetch"))]
        {
            // No-op under disabled feature flag
        }
    }

    pub async fn prefetch_layers(
        &self,
        model_name: &str,
        db: &std::sync::Arc<Database>,
        current_layer: usize,
        num_layers: usize,
        prefetch_depth: usize,
    ) {
        let components = [
            "input_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "post_attention_layernorm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ];
        self.prefetch_components(
            model_name,
            db,
            current_layer,
            num_layers,
            prefetch_depth,
            &components,
        )
        .await;
    }

    pub fn hit_rate(&self) -> f32 {
        let total = self.total_prefetches.load(Ordering::Relaxed);
        if total == 0 {
            1.0
        } else {
            self.successful_prefetches.load(Ordering::Relaxed) as f32 / total as f32
        }
    }

    pub fn get_adaptive_depth(&self) -> usize {
        #[cfg(feature = "prefetch")]
        {
            let hr = self.hit_rate();
            if hr > 0.90 {
                2
            } else if hr > 0.60 {
                1
            } else {
                0
            }
        }
        #[cfg(not(feature = "prefetch"))]
        {
            0
        }
    }

    pub fn get_session_prefetch_depth(
        &self,
        session_id: Option<&str>,
        compute_plane: Option<&WgpuComputePlane>,
    ) -> usize {
        #[cfg(feature = "prefetch")]
        {
            if let (Some(sess), Some(plane)) = (session_id, compute_plane) {
                let states = plane.session_states.lock().unwrap();
                if let Some(stats) = states.get(sess) {
                    if stats.state == DegradationState::Orange
                        || stats.state == DegradationState::Red
                    {
                        return 0;
                    }
                }
            }
        }
        self.get_adaptive_depth()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_hybrid_prefetcher_correctness() {
        let prefetcher = Prefetcher::new();

        #[cfg(feature = "prefetch")]
        {
            let predicted = prefetcher.predict_next_pages(&[1], 45.0, 1.2);
            assert_eq!(predicted.len(), 2);

            // Record some transitions from page 1 -> page 5
            prefetcher.record_access(&[1], &[5]);

            // Page 5 should now have a higher probability and be predicted!
            let predicted_after = prefetcher.predict_next_pages(&[1], 45.0, 1.2);
            assert!(predicted_after.contains(&5));
        }
    }
}
