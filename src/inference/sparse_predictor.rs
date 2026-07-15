use rayon::prelude::*;

/// Simulates a 2:4 block-sparse matrix-vector multiplication.
/// In 2:4 sparsity, out of every contiguous 4 elements in the weight matrix,
/// only the 2 with the largest absolute magnitude are preserved. The rest are zeroed.
/// 
/// - `x`: Input vector of size N
/// - `w`: Weight matrix of size (M x N) in row-major order
/// - `out`: Output vector of size M
/// - `cols`: The number of columns N
pub fn sparse_matvec_mul_2_4(x: &[f32], w: &[f32], cols: usize) -> Vec<f32> {
    let rows = w.len() / cols;
    let mut out = vec![0.0; rows];

    out.par_iter_mut().enumerate().for_each(|(r, out_val)| {
        let row_start = r * cols;
        let row_slice = &w[row_start..row_start + cols];
        
        let mut sum = 0.0;
        
        // Process in chunks of 4
        let mut c = 0;
        while c + 4 <= cols {
            // Find the 2 largest magnitude elements in the 4-element block
            let mut mags = [
                (0, row_slice[c].abs()),
                (1, row_slice[c + 1].abs()),
                (2, row_slice[c + 2].abs()),
                (3, row_slice[c + 3].abs()),
            ];
            
            // Sort by magnitude descending
            mags.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            
            // Add the top 2 elements to the sum
            let idx1 = mags[0].0;
            let idx2 = mags[1].0;
            
            sum += row_slice[c + idx1] * x[c + idx1];
            sum += row_slice[c + idx2] * x[c + idx2];
            
            c += 4;
        }
        
        // Handle any remaining elements (if cols is not divisible by 4)
        // Usually LLM dimensions are divisible by 128, so this is just a fallback.
        while c < cols {
            sum += row_slice[c] * x[c];
            c += 1;
        }
        
        *out_val = sum;
    });

    out
}

/// Computes the cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    
    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;
    
    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    
    dot / (norm_a.sqrt() * norm_b.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_matvec_mul_2_4() {
        let x = vec![1.0, 1.0, 1.0, 1.0,  1.0, 1.0, 1.0, 1.0];
        
        // Row 1: 0.1, 10.0, -5.0, 0.2 (Top 2: 10.0, -5.0) -> sum: 5.0
        //        1.0, -0.1, 0.5, 2.0 (Top 2: 1.0, 2.0) -> sum: 3.0
        // Expected out[0] = 8.0
        
        // Row 2: 0.0, 0.0, 1.0, 2.0 (Top 2: 1.0, 2.0) -> sum: 3.0
        //        -5.0, -6.0, 1.0, 0.1 (Top 2: -5.0, -6.0) -> sum: -11.0
        // Expected out[1] = -8.0
        
        let w = vec![
            0.1, 10.0, -5.0, 0.2,   1.0, -0.1, 0.5, 2.0,
            0.0, 0.0, 1.0, 2.0,     -5.0, -6.0, 1.0, 0.1
        ];
        
        let out = sparse_matvec_mul_2_4(&x, &w, 8);
        assert_eq!(out.len(), 2);
        assert!((out[0] - 8.0).abs() < 1e-5);
        assert!((out[1] - -8.0).abs() < 1e-5);
    }
    
    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-5);
        
        let c = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &c) - 0.0).abs() < 1e-5);
        
        let d = vec![-1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &d) - -1.0).abs() < 1e-5);
    }
}
