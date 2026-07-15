use rand::seq::SliceRandom;
use rayon::prelude::*;
use crate::core::distance::l2_distance;

/// Performs K-Means clustering on a dataset of vectors.
/// Returns a list of K centroids.
pub fn kmeans(
    data: &[Vec<f32>],
    k: usize,
    max_iters: usize,
    dimension: usize,
) -> Vec<Vec<f32>> {
    if data.is_empty() || k == 0 {
        return vec![];
    }
    
    // If we have fewer items than clusters, just return the items themselves as centroids
    if data.len() <= k {
        return data.to_vec();
    }

    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    
    // 1. K-Means++ initialization algorithm
    let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(k);
    
    // Choose the first centroid uniformly at random
    if let Some(first) = data.choose(&mut rng) {
        centroids.push(first.clone());
    }

    while centroids.len() < k {
        // Compute D(x)^2 for each point to nearest existing centroid
        let mut distances: Vec<f64> = vec![0.0; data.len()];
        let mut sum_sq_dist = 0.0;

        for (idx, point) in data.iter().enumerate() {
            let mut min_dist = f32::INFINITY;
            for centroid in &centroids {
                let dist = l2_distance(point, centroid);
                if dist < min_dist {
                    min_dist = dist;
                }
            }
            let d_sq = (min_dist * min_dist) as f64;
            distances[idx] = d_sq;
            sum_sq_dist += d_sq;
        }

        // Sample the next centroid with probability proportional to D(x)^2
        if sum_sq_dist == 0.0 {
            if let Some(random_point) = data.choose(&mut rng) {
                centroids.push(random_point.clone());
            }
        } else {
            let mut target: f64 = rand::Rng::gen_range(&mut rng, 0.0..sum_sq_dist);
            let mut selected_idx = 0;
            for (idx, &d_sq) in distances.iter().enumerate() {
                target -= d_sq;
                if target <= 0.0 {
                    selected_idx = idx;
                    break;
                }
            }
            centroids.push(data[selected_idx].clone());
        }
    }

    let tolerance = 1e-4;

    for _iter in 0..max_iters {
        // 2. Assign points to the closest centroid (Multi-threaded)
        let assignments: Vec<(usize, usize)> = data
            .par_iter()
            .enumerate()
            .map(|(idx, point)| {
                let mut min_dist = f32::INFINITY;
                let mut closest_centroid = 0;

                for (c_idx, centroid) in centroids.iter().enumerate() {
                    let dist = l2_distance(point, centroid);
                    if dist < min_dist {
                        min_dist = dist;
                        closest_centroid = c_idx;
                    }
                }
                (closest_centroid, idx)
            })
            .collect();

        // 2.5 Group indices into cluster buckets
        let mut clusters: Vec<Vec<usize>> = vec![vec![]; k];
        for (c_idx, p_idx) in assignments {
            clusters[c_idx].push(p_idx);
        }

        // 3. Compute new centroids as the mean of points in each cluster
        let mut new_centroids = vec![vec![0.0; dimension]; k];
        let mut shifted = false;

        for c_idx in 0..k {
            let points_in_cluster = &clusters[c_idx];
            if points_in_cluster.is_empty() {
                // If a cluster is empty, re-initialize it to a random data point
                if let Some(random_point) = data.choose(&mut rng) {
                    new_centroids[c_idx] = random_point.clone();
                }
                shifted = true;
                continue;
            }

            let num_points = points_in_cluster.len() as f32;
            let mut sum_vec = vec![0.0; dimension];

            for &point_idx in points_in_cluster {
                for i in 0..dimension {
                    sum_vec[i] += data[point_idx][i];
                }
            }

            for i in 0..dimension {
                new_centroids[c_idx][i] = sum_vec[i] / num_points;
            }

            // Check if centroid shifted significantly
            if l2_distance(&centroids[c_idx], &new_centroids[c_idx]) > tolerance {
                shifted = true;
            }
        }

        centroids = new_centroids;

        // If no centroids shifted significantly, we have converged
        if !shifted {
            break;
        }
    }

    centroids
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmeans_simple() {
        // 4 points clearly in 2 clusters:
        // Cluster 0: near (1.0, 1.0)
        // Cluster 1: near (10.0, 10.0)
        let data = vec![
            vec![1.0, 1.2],
            vec![0.9, 0.8],
            vec![10.1, 10.0],
            vec![9.8, 10.2],
        ];

        let centroids = kmeans(&data, 2, 50, 2);
        
        assert_eq!(centroids.len(), 2);
        // Verify centroids are around (1,1) and (10,10)
        let has_near_1_1 = centroids.iter().any(|c| l2_distance(c, &[1.0, 1.0]) < 1.0);
        let has_near_10_10 = centroids.iter().any(|c| l2_distance(c, &[10.0, 10.0]) < 1.0);
        
        assert!(has_near_1_1);
        assert!(has_near_10_10);
    }
}
