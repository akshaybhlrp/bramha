/// WGPU Armored Sparse Pager
/// Packs and unpacks 4x4 block masks into u16 bitmasks for efficient GPU memory transfers.

pub struct SparseBlockMask {
    pub mask: u16,
}

impl SparseBlockMask {
    /// Packs a 4x4 block (16 elements) into a single u16 bitmask.
    /// Bit i is set to 1 if the corresponding element is non-zero.
    pub fn pack_4x4(block: &[f32; 16]) -> Self {
        let mut mask: u16 = 0;
        for i in 0..16 {
            if block[i].abs() > 1e-7 {
                mask |= 1 << i;
            }
        }
        SparseBlockMask { mask }
    }

    /// Checks if a specific index in the 4x4 block is non-zero.
    pub fn is_active(&self, index: usize) -> bool {
        if index >= 16 {
            return false;
        }
        (self.mask & (1 << index)) != 0
    }

    /// Returns the number of non-zero elements in this block.
    pub fn active_count(&self) -> u32 {
        self.mask.count_ones()
    }
}

/// Helper function to convert a flat matrix into packed block bitmasks
/// and a contiguous array of non-zero values.
pub fn pack_sparse_matrix(weights: &[f32], _cols: usize) -> (Vec<u16>, Vec<f32>) {
    assert!(weights.len() % 16 == 0, "Weight matrix must be divisible by 4x4 blocks (16 elements)");
    
    let num_blocks = weights.len() / 16;
    let mut masks = Vec::with_capacity(num_blocks);
    let mut nonzero_values = Vec::new(); // Dynamically sized based on sparsity
    
    for b in 0..num_blocks {
        let start = b * 16;
        let mut block = [0.0; 16];
        block.copy_from_slice(&weights[start..start + 16]);
        
        let sparse_mask = SparseBlockMask::pack_4x4(&block);
        masks.push(sparse_mask.mask);
        
        for i in 0..16 {
            if sparse_mask.is_active(i) {
                nonzero_values.push(block[i]);
            }
        }
    }
    
    // To prevent OOM and enforce capacity limits, shrink vectors to exact size
    nonzero_values.shrink_to_fit();
    
    (masks, nonzero_values)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_4x4_block() {
        let mut block = [0.0; 16];
        // Set specific elements to simulate a 2:4 sparse block layout
        block[0] = 1.5;
        block[3] = -0.5;
        block[8] = 2.0;
        block[15] = 0.1;
        
        let mask = SparseBlockMask::pack_4x4(&block);
        
        // Bit 0, 3, 8, 15 should be 1
        assert!(mask.is_active(0));
        assert!(!mask.is_active(1));
        assert!(mask.is_active(3));
        assert!(mask.is_active(8));
        assert!(mask.is_active(15));
        
        assert_eq!(mask.active_count(), 4);
    }
    
    #[test]
    fn test_pack_sparse_matrix() {
        let mut weights = vec![0.0; 32]; // Two 4x4 blocks
        
        // Block 1
        weights[0] = 1.0;
        weights[5] = -2.0;
        
        // Block 2
        weights[16] = 5.0;
        weights[31] = 6.0;
        
        let (masks, values) = pack_sparse_matrix(&weights, 4);
        
        assert_eq!(masks.len(), 2);
        assert_eq!(values.len(), 4);
        
        assert_eq!(values[0], 1.0);
        assert_eq!(values[1], -2.0);
        assert_eq!(values[2], 5.0);
        assert_eq!(values[3], 6.0);
    }
}
