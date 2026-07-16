use nalgebra::DMatrix;
use rand::prelude::*;

/// Performs Randomized Singular Value Decomposition (SVD) on a matrix.
/// W ≈ U @ S @ V^T
///
/// Under the hood, this folds the singular values S into V^T (creating B = diag(S) @ V^T)
/// to return two matrices A (U, shape: rows x rank) and B (folded V^T, shape: rank x cols)
/// such that W ≈ A @ B.
///
/// Returns: Ok((A, B, target_rank))
pub fn randomized_svd(
    w_data: &[f32],
    rows: usize,
    cols: usize,
    rank: usize,
) -> Result<(Vec<f32>, Vec<f32>, usize), String> {
    if rank == 0 {
        return Err("Rank must be greater than 0".to_string());
    }
    let target_rank = rank.min(rows).min(cols);

    // Convert flat slice to nalgebra DMatrix
    let w = DMatrix::from_row_slice(rows, cols, w_data);

    // Oversampling parameter p
    let p = 5;
    let k_oversampled = (target_rank + p).min(rows).min(cols);

    // 1. Generate random matrix Omega with standard normal distribution
    let mut rng = StdRng::seed_from_u64(42);
    let mut omega_data = Vec::with_capacity(cols * k_oversampled);
    for _ in 0..((cols * k_oversampled + 1) / 2) {
        let u1: f32 = rng.r#gen();
        let u2: f32 = rng.r#gen();
        let u1 = u1.max(1e-30); // Avoid log(0)
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        omega_data.push(r * theta.cos());
        omega_data.push(r * theta.sin());
    }
    omega_data.truncate(cols * k_oversampled);
    let omega = DMatrix::from_row_slice(cols, k_oversampled, &omega_data);

    // 2. Compute Y = W @ Omega
    let y = &w * &omega;

    // 3. QR decomposition of Y
    let qr = y.qr();
    let q = qr.q(); // Q matrix of shape rows x k_oversampled

    // 4. Project W onto the subspace: B = Q^T @ W
    let b = q.transpose() * &w; // B matrix of shape k_oversampled x cols

    // 5. Compute SVD of the small matrix B
    let svd = b.svd(true, true);
    let u_hat = svd.u.ok_or("Failed to compute SVD U-factor".to_string())?;
    let singular_values = svd.singular_values;
    let v_t = svd
        .v_t
        .ok_or("Failed to compute SVD V^T-factor".to_string())?;

    // 6. Compute U = Q @ u_hat
    let u = q * u_hat;

    // 7. Truncate to target_rank
    // Extract truncated U (shape: rows x target_rank)
    let u_truncated = u.columns(0, target_rank).into_owned();

    // Extract truncated S (shape: target_rank)
    let s_truncated = singular_values.rows(0, target_rank).into_owned();

    // Extract truncated V^T (shape: target_rank x cols)
    let vt_truncated = v_t.rows(0, target_rank).into_owned();

    // Flat vector representation of U (A) (rows x target_rank) with sqrt(S) folded symmetrically
    let mut a_flat = vec![0.0f32; rows * target_rank];
    for r in 0..rows {
        for c in 0..target_rank {
            let s_sqrt = s_truncated[c].sqrt();
            a_flat[r * target_rank + c] = u_truncated[(r, c)] * s_sqrt;
        }
    }

    // Fold sqrt(S) symmetrically into V^T (B) (target_rank x cols)
    let mut b_flat = vec![0.0f32; target_rank * cols];
    for r in 0..target_rank {
        let s_sqrt = s_truncated[r].sqrt();
        for c in 0..cols {
            b_flat[r * cols + c] = vt_truncated[(r, c)] * s_sqrt;
        }
    }

    Ok((a_flat, b_flat, target_rank))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_randomized_svd_reconstruction() {
        // Create a rank-2 matrix of shape 4x4
        // W = U_true @ V_true^T
        // U_true: 4x2, V_true^T: 2x4
        let a_true = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b_true = [0.5, 1.5, -0.5, 2.0, -1.0, 0.0, 1.0, 0.5];

        let mut w = vec![0.0f32; 16];
        for r in 0..4 {
            for c in 0..4 {
                let mut sum = 0.0;
                for k in 0..2 {
                    sum += a_true[r * 2 + k] * b_true[k * 4 + c];
                }
                w[r * 4 + c] = sum;
            }
        }

        // Decompose W with rank 2
        let (a, b, r) = randomized_svd(&w, 4, 4, 2).unwrap();
        assert_eq!(r, 2);
        assert_eq!(a.len(), 8);
        assert_eq!(b.len(), 8);

        // Reconstruct W_approx = A @ B
        let mut w_approx = vec![0.0f32; 16];
        for r_idx in 0..4 {
            for c_idx in 0..4 {
                let mut sum = 0.0;
                for k in 0..2 {
                    sum += a[r_idx * 2 + k] * b[k * 4 + c_idx];
                }
                w_approx[r_idx * 4 + c_idx] = sum;
            }
        }

        // Compare reconstruction vs original
        for i in 0..16 {
            let diff = (w[i] - w_approx[i]).abs();
            assert!(
                diff < 1e-3,
                "Index {}: expected {}, got {}, diff {}",
                i,
                w[i],
                w_approx[i],
                diff
            );
        }
    }
}
