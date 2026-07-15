/// Computes the Euclidean (L2) distance between two vectors.
pub fn l2_distance(u: &[f32], v: &[f32]) -> f32 {
    if u.len() != v.len() {
        return f32::INFINITY;
    }
    let sum: f32 = u.iter()
        .zip(v.iter())
        .map(|(&a, &b)| {
            let diff = a - b;
            diff * diff
        })
        .sum();
    sum.sqrt()
}

/// Computes the dot product of two vectors.
pub fn dot_product(u: &[f32], v: &[f32]) -> f32 {
    if u.len() != v.len() {
        return 0.0;
    }
    u.iter().zip(v.iter()).map(|(&a, &b)| a * b).sum()
}

/// Computes the cosine similarity between two vectors.
/// Range is [-1.0, 1.0], where 1.0 means identical direction.
pub fn cosine_similarity(u: &[f32], v: &[f32]) -> f32 {
    if u.len() != v.len() || u.is_empty() {
        return 0.0;
    }
    let dot = dot_product(u, v);
    let norm_u: f32 = u.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_v: f32 = v.iter().map(|&x| x * x).sum::<f32>().sqrt();
    
    if norm_u == 0.0 || norm_v == 0.0 {
        return 0.0;
    }
    
    dot / (norm_u * norm_v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l2_distance() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 6.0, 3.0];
        // sqrt((1-4)^2 + (2-6)^2 + (3-3)^2) = sqrt(9 + 16 + 0) = sqrt(25) = 5.0
        assert_eq!(l2_distance(&a, &b), 5.0);
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32.0
        assert_eq!(dot_product(&a, &b), 32.0);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let c = vec![2.0, 0.0];
        
        assert_eq!(cosine_similarity(&a, &b), 0.0); // orthogonal
        assert!((cosine_similarity(&a, &c) - 1.0).abs() < 1e-6); // parallel
    }
}
