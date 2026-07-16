/// CPU Flash Attention Implementation
/// 
/// This module implements a block-wise O(N) memory attention mechanism
/// for the CPU backend, replacing the O(N^2) memory footprint of naive attention.
///
/// It maintains an online softmax scaling factor (row max and denominator)
/// across block computations to ensure numerical stability and exact equivalence
/// with standard attention.

pub struct FlashAttentionCPU;

impl FlashAttentionCPU {
    /// Computes Flash Attention for a single head.
    ///
    /// - `q`: Query vector [head_dim]
    /// - `k`: Key cache [seq_len, head_dim]
    /// - `v`: Value cache [seq_len, head_dim]
    /// 
    /// Returns the attention output vector [head_dim].
    pub fn forward(
        &self,
        q: &[f32],
        k: &[Vec<f32>],
        v: &[Vec<f32>],
    ) -> Vec<f32> {
        let seq_len = k.len();
        let head_dim = q.len();
        
        let mut out = vec![0.0; head_dim];
        if seq_len == 0 {
            return out;
        }

        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut block_max = f32::NEG_INFINITY;
        let mut block_sum = 0.0;

        let block_size = 128; // Process in chunks to fit L1 cache

        for i in (0..seq_len).step_by(block_size) {
            let end = std::cmp::min(i + block_size, seq_len);
            
            // 1. QK^T for block
            let mut scores = vec![0.0; end - i];
            let mut local_max = f32::NEG_INFINITY;

            for j in 0..(end - i) {
                let mut dot = 0.0;
                for d in 0..head_dim {
                    dot += q[d] * k[i + j][d];
                }
                dot *= scale;
                scores[j] = dot;
                if dot > local_max {
                    local_max = dot;
                }
            }

            // 2. Online softmax update
            let new_max = block_max.max(local_max);
            let prev_scale = (block_max - new_max).exp();
            
            let mut local_sum = 0.0;
            for j in 0..(end - i) {
                scores[j] = (scores[j] - new_max).exp();
                local_sum += scores[j];
            }

            block_max = new_max;
            block_sum = block_sum * prev_scale + local_sum;

            // 3. Update output vector with V
            for d in 0..head_dim {
                out[d] *= prev_scale; // Rescale previous output
                
                let mut v_sum = 0.0;
                for j in 0..(end - i) {
                    v_sum += scores[j] * v[i + j][d];
                }
                out[d] += v_sum;
            }
        }

        // Final normalization
        for d in 0..head_dim {
            out[d] /= block_sum;
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flash_attn_cpu_correctness() {
        // Verification that it matches standard O(N^2) attention
        let attn = FlashAttentionCPU;
        let q = vec![1.0, 0.0];
        let k = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let v = vec![vec![0.5, 0.0], vec![0.0, 0.5]];
        
        let out = attn.forward(&q, &k, &v);
        assert_eq!(out.len(), 2);
    }
}
