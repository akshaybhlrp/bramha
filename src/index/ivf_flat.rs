use serde::{Serialize, Deserialize};
use rayon::prelude::*;
use crate::core::collection::{Collection, SearchResult};
use crate::core::filter::Filter;
use crate::core::distance::l2_distance;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusteringAnalytics {
    pub davies_bouldin_index: f32,
    pub silhouette_score: f32,
    pub imbalance_ratio: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IvfFlatIndex {
    pub num_clusters: usize,
    pub n_probe: usize,
    pub centroids: Vec<Vec<f32>>,
    pub buckets: Vec<Vec<String>>, // Maps centroid index to a list of Vector IDs
    pub health_score: f32,
    pub recall_at_k: f32,
    pub analytics: Option<ClusteringAnalytics>,
}

impl IvfFlatIndex {
    /// Builds an IVF-Flat index from the vectors currently in the collection.
    pub fn build(collection: &Collection, num_clusters: usize, n_probe: usize) -> Self {
        if collection.vectors.is_empty() || num_clusters == 0 {
            return IvfFlatIndex {
                num_clusters: 0,
                n_probe: 0,
                centroids: vec![],
                buckets: vec![],
                health_score: 1.0,
                recall_at_k: 1.0,
                analytics: None,
            };
        }

        let vec_ids: Vec<String> = collection.vectors.keys().cloned().collect();
        let vec_data: Vec<Vec<f32>> = collection.vectors.values().map(|v| v.values.clone()).collect();

        // 1. Compute cluster centroids using K-Means
        let centroids = crate::index::kmeans::kmeans(
            &vec_data,
            num_clusters,
            100, // Maximum K-Means iterations
            collection.dimension,
        );

        let actual_k = centroids.len();
        let mut buckets = vec![vec![]; actual_k];

        // 2. Assign each vector to the nearest centroid bucket
        for (id, values) in vec_ids.iter().zip(vec_data.iter()) {
            let mut min_dist = f32::INFINITY;
            let mut closest_centroid = 0;

            for (c_idx, centroid) in centroids.iter().enumerate() {
                let dist = l2_distance(values, centroid);
                if dist < min_dist {
                    min_dist = dist;
                    closest_centroid = c_idx;
                }
            }
            buckets[closest_centroid].push(id.clone());
        }

        // S4.2: Compute health score based on cluster sizes standard deviation / imbalance
        let sizes: Vec<f32> = buckets.iter().map(|b| b.len() as f32).collect();
        let n = sizes.len() as f32;
        let health_score = if n > 0.0 {
            let mean: f32 = sizes.iter().sum::<f32>() / n;
            if mean > 0.0 {
                let variance: f32 = sizes.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / n;
                let std_dev = variance.sqrt();
                let cv = std_dev / mean;
                (1.0 - cv).max(0.0)
            } else {
                1.0
            }
        } else {
            1.0
        };

        // S4.2: Compute recall@K against exact search on 5 sample vectors
        let mut recall_at_k = 1.0;
        if !collection.vectors.is_empty() {
            let sample_keys: Vec<String> = collection.vectors.keys().cloned().collect();
            let num_samples = std::cmp::min(5, sample_keys.len());
            let mut total_recall = 0.0;
            
            // Build temporary index for validation
            let temp_idx = IvfFlatIndex {
                num_clusters: actual_k,
                n_probe: std::cmp::min(n_probe, actual_k),
                centroids: centroids.clone(),
                buckets: buckets.clone(),
                health_score,
                recall_at_k: 1.0,
                analytics: None,
            };
            
            for key in sample_keys.iter().take(num_samples) {
                if let Some(v) = collection.vectors.get(key) {
                    let exact_res = collection.search(&v.values, 5, None, false);
                    let exact_ids: std::collections::HashSet<String> = exact_res.iter().map(|r| r.id.clone()).collect();
                    
                    let ann_res = temp_idx.search(collection, &v.values, 5, None, None);
                    let ann_ids: std::collections::HashSet<String> = ann_res.iter().map(|r| r.id.clone()).collect();
                    
                    if !exact_ids.is_empty() {
                        let matches = exact_ids.intersection(&ann_ids).count() as f32;
                        total_recall += matches / exact_ids.len() as f32;
                    } else {
                        total_recall += 1.0;
                    }
                }
            }
            recall_at_k = total_recall / num_samples as f32;
        }

        // S4.3: Davies-Bouldin Index
        let mut db_index = 0.0;
        if actual_k > 1 {
            // 1. Calculate scatter s_i for each cluster
            let mut scatters = vec![0.0; actual_k];
            for i in 0..actual_k {
                let bucket = &buckets[i];
                if !bucket.is_empty() {
                    let sum_dist: f32 = bucket.iter()
                        .map(|id| collection.vectors.get(id).unwrap())
                        .map(|v| l2_distance(&v.values, &centroids[i]))
                        .sum();
                    scatters[i] = sum_dist / bucket.len() as f32;
                }
            }

            // 2. Compute R_ij and find max for each cluster i
            let mut sum_r = 0.0;
            for i in 0..actual_k {
                let mut max_r = 0.0;
                for j in 0..actual_k {
                    if i != j {
                        let c_dist = l2_distance(&centroids[i], &centroids[j]);
                        if c_dist > 1e-5 {
                            let r = (scatters[i] + scatters[j]) / c_dist;
                            if r > max_r {
                                max_r = r;
                            }
                        }
                    }
                }
                sum_r += max_r;
            }
            db_index = sum_r / actual_k as f32;
        }

        // S4.3: Sampled Silhouette Score (Sample up to 50 vectors)
        let mut sil_score = 0.0;
        if !collection.vectors.is_empty() && actual_k > 1 {
            let sample_keys: Vec<String> = collection.vectors.keys().cloned().collect();
            let num_sil_samples = std::cmp::min(50, sample_keys.len());
            let mut total_s = 0.0;

            for key in sample_keys.iter().take(num_sil_samples) {
                let target_v = collection.vectors.get(key).unwrap();
                
                // Find which cluster the target vector belongs to
                let mut own_cluster_idx = 0;
                for i in 0..actual_k {
                    if buckets[i].contains(key) {
                        own_cluster_idx = i;
                        break;
                    }
                }

                // Compute a(i): average distance to other vectors in the same cluster
                let own_bucket = &buckets[own_cluster_idx];
                let a = if own_bucket.len() > 1 {
                    let sum_a: f32 = own_bucket.iter()
                        .filter(|id| *id != key)
                        .map(|id| collection.vectors.get(id).unwrap())
                        .map(|v| l2_distance(&target_v.values, &v.values))
                        .sum();
                    sum_a / (own_bucket.len() - 1) as f32
                } else {
                    0.0
                };

                // Compute b(i): min average distance to vectors in other clusters
                let mut min_b = f32::INFINITY;
                for j in 0..actual_k {
                    if j != own_cluster_idx {
                        let other_bucket = &buckets[j];
                        if !other_bucket.is_empty() {
                            let sum_b: f32 = other_bucket.iter()
                                .map(|id| collection.vectors.get(id).unwrap())
                                .map(|v| l2_distance(&target_v.values, &v.values))
                                .sum();
                            let avg_b = sum_b / other_bucket.len() as f32;
                            if avg_b < min_b {
                                min_b = avg_b;
                            }
                        }
                    }
                }
                let b = if min_b.is_infinite() { 0.0 } else { min_b };

                // Compute silhouette value s(i)
                let max_ab = a.max(b);
                let s = if max_ab > 1e-5 {
                    (b - a) / max_ab
                } else {
                    0.0
                };
                total_s += s;
            }
            sil_score = total_s / num_sil_samples as f32;
        }

        // S4.3: Imbalance Ratio
        let max_size = buckets.iter().map(|b| b.len()).max().unwrap_or(0) as f32;
        let avg_size = if actual_k > 0 {
            buckets.iter().map(|b| b.len()).sum::<usize>() as f32 / actual_k as f32
        } else {
            1.0
        };
        let imbalance_ratio = if avg_size > 0.0 { max_size / avg_size } else { 1.0 };

        let analytics = Some(ClusteringAnalytics {
            davies_bouldin_index: db_index,
            silhouette_score: sil_score,
            imbalance_ratio,
        });

        IvfFlatIndex {
            num_clusters: actual_k,
            n_probe: std::cmp::min(n_probe, actual_k),
            centroids,
            buckets,
            health_score,
            recall_at_k,
            analytics,
        }
    }

    /// Performs Approximate Nearest Neighbor (ANN) search.
    pub fn search(
        &self,
        collection: &Collection,
        query: &[f32],
        k: usize,
        filter: Option<&Filter>,
        allowed_ids: Option<&std::collections::HashSet<String>>,
    ) -> Vec<SearchResult> {
        if self.centroids.is_empty() {
            return vec![];
        }

        // 1. Find the top n_probe closest centroids to the query vector
        let mut centroid_dists: Vec<(usize, f32)> = self.centroids
            .iter()
            .enumerate()
            .map(|(c_idx, centroid)| {
                // Centroid similarity is computed using L2 distance in clustering space
                let dist = l2_distance(query, centroid);
                (c_idx, dist)
            })
            .collect();

        // Sort centroids: closest (smallest L2 distance) first
        centroid_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        
        let factor = match collection.tuning_profile {
            crate::core::collection::TuningProfile::LowLatency => 0.05,
            crate::core::collection::TuningProfile::Balanced => 0.15,
            crate::core::collection::TuningProfile::HighRecall => 0.35,
        };
        let auto_probes = (self.centroids.len() as f32 * factor).round() as usize;
        let final_probes = std::cmp::max(1, auto_probes);
        
        let probes = std::cmp::min(final_probes, self.centroids.len());
        let probe_centroids: Vec<usize> = centroid_dists
            .iter()
            .take(probes)
            .map(|&(c_idx, _)| c_idx)
            .collect();

        // 2. Collect all candidate vectors from the selected buckets
        let mut candidate_ids = std::collections::HashSet::new();
        for &c_idx in &probe_centroids {
            if c_idx < self.buckets.len() {
                for id in &self.buckets[c_idx] {
                    candidate_ids.insert(id);
                }
            }
        }

        // 3. Rank and filter candidate vectors using the collection's target distance metric (Multi-threaded)
        let mut results: Vec<SearchResult> = candidate_ids
            .into_iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .filter_map(|id| collection.vectors.get(id))
            .filter(|v| {
                if let Some(allowed) = allowed_ids {
                    allowed.contains(&v.id)
                } else if let Some(f) = filter {
                    f.matches(&v.metadata)
                } else {
                    true
                }
            })
            .map(|v| {
                let score = collection.metric.distance(query, &v.values);
                SearchResult {
                    id: v.id.clone(),
                    score,
                    metadata: v.metadata.clone(),
                    ..Default::default()
                }
            })
            .collect();

        // 4. Sort according to collection metric rules
        if collection.metric.is_ascending() {
            results.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal));
        } else {
            results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        }

        results.truncate(k);
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::vector::{Vector, Metric};
    use crate::core::collection::TuningProfile;

    #[test]
    fn test_ivf_diagnostics_and_profiles() {
        let mut collection = Collection::new("test_ivf_diag".to_string(), 4, Metric::L2);
        
        // Add 20 deterministic vectors
        for i in 0..20 {
            let val = i as f32;
            let vec = Vector {
                id: format!("doc_{}", i),
                values: vec![val, val, val, val],
                metadata: None,
            };
            collection.insert(vec).unwrap();
        }

        // Build index with 4 clusters
        let idx = IvfFlatIndex::build(&collection, 4, 2);

        // Verify health score and recall are calculated and within [0, 1] bounds
        assert!(idx.health_score >= 0.0 && idx.health_score <= 1.0);
        assert!(idx.recall_at_k >= 0.0 && idx.recall_at_k <= 1.0);

        // Verify S4.3 Clustering quality analytics
        assert!(idx.analytics.is_some());
        let analytics = idx.analytics.as_ref().unwrap();
        assert!(analytics.davies_bouldin_index >= 0.0);
        assert!(analytics.silhouette_score >= -1.0 && analytics.silhouette_score <= 1.0);
        assert!(analytics.imbalance_ratio >= 1.0);

        // Verify profile-based probes selection
        collection.tuning_profile = TuningProfile::LowLatency;
        let res_ll = idx.search(&collection, &vec![5.0, 5.0, 5.0, 5.0], 3, None, None);
        assert!(!res_ll.is_empty());

        collection.tuning_profile = TuningProfile::HighRecall;
        let res_hr = idx.search(&collection, &vec![5.0, 5.0, 5.0, 5.0], 3, None, None);
        assert!(!res_hr.is_empty());
    }
}
