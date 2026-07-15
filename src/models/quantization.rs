/// Quantization logic for INT8 and INT4 (U4) weight parameters in Bramha.
///
/// Provides high-performance, robust per-row/channel symmetric INT8 quantization,
/// and scale-and-zero-point INT4 quantization with packed nibbles.

/// Quantize f32 weights to i8 using row-wise symmetric quantization.
pub fn quantize_to_int8(weights: &[f32], out_features: usize, in_features: usize) -> (Vec<i8>, Vec<f32>) {
    let mut q_weights = vec![0i8; out_features * in_features];
    let mut scales = vec![0.0f32; out_features];

    for j in 0..out_features {
        let row_offset = j * in_features;
        let row = &weights[row_offset..row_offset + in_features];
        
        let mut max_abs = 0.0f32;
        for &x in row {
            let abs = x.abs();
            if abs > max_abs {
                max_abs = abs;
            }
        }

        let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
        scales[j] = scale;

        for i in 0..in_features {
            let x = row[i];
            let q = (x / scale).round().clamp(-127.0, 127.0) as i8;
            q_weights[row_offset + i] = q;
        }
    }

    (q_weights, scales)
}

/// Quantize f32 weights to 4-bit unsigned integers (0-15) packed into u8, using row-wise scale and zero-point=8.
pub fn quantize_to_int4(weights: &[f32], out_features: usize, in_features: usize) -> (Vec<u8>, Vec<f32>) {
    assert!(in_features % 2 == 0, "in_features must be a multiple of 2 for INT4 packing");
    let mut q_bytes = vec![0u8; out_features * (in_features / 2)];
    let mut scales = vec![0.0f32; out_features];

    for j in 0..out_features {
        let row_offset = j * in_features;
        let row = &weights[row_offset..row_offset + in_features];

        let mut max_abs = 0.0f32;
        for &x in row {
            let abs = x.abs();
            if abs > max_abs {
                max_abs = abs;
            }
        }

        // Map f32 to range [-7.0, 7.0] first, then add 8.0 to fit in [1, 15]
        let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 7.0 };
        scales[j] = scale;

        let byte_row_offset = j * (in_features / 2);
        for i in 0..(in_features / 2) {
            let x1 = row[i * 2];
            let x2 = row[i * 2 + 1];

            let q1 = ((x1 / scale).round().clamp(-7.0, 7.0) + 8.0) as u8;
            let q2 = ((x2 / scale).round().clamp(-7.0, 7.0) + 8.0) as u8;

            // Pack q1 in high nibble, q2 in low nibble
            q_bytes[byte_row_offset + i] = (q1 << 4) | (q2 & 0x0F);
        }
    }

    (q_bytes, scales)
}

/// Dequantize row-wise i8 weights back to f32.
pub fn dequantize_int8(q_weights: &[i8], scales: &[f32], in_features: usize) -> Vec<f32> {
    let out_features = scales.len();
    let mut weights = vec![0.0f32; out_features * in_features];

    for j in 0..out_features {
        let row_offset = j * in_features;
        let scale = scales[j];
        for i in 0..in_features {
            weights[row_offset + i] = q_weights[row_offset + i] as f32 * scale;
        }
    }

    weights
}

/// Dequantize row-wise packed 4-bit weights back to f32.
pub fn dequantize_int4(q_bytes: &[u8], scales: &[f32], in_features: usize) -> Vec<f32> {
    let out_features = scales.len();
    let mut weights = vec![0.0f32; out_features * in_features];

    for j in 0..out_features {
        let byte_row_offset = j * (in_features / 2);
        let row_offset = j * in_features;
        let scale = scales[j];

        for i in 0..(in_features / 2) {
            let byte = q_bytes[byte_row_offset + i];
            let q1 = ((byte >> 4) & 0x0F) as f32 - 8.0;
            let q2 = (byte & 0x0F) as f32 - 8.0;

            weights[row_offset + i * 2] = q1 * scale;
            weights[row_offset + i * 2 + 1] = q2 * scale;
        }
    }

    weights
}
