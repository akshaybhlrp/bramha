struct Params {
    in_features: u32,
    out_features: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input_vector: array<f32>;
@group(0) @binding(2) var<storage, read> masks: array<u32>;
@group(0) @binding(3) var<storage, read> values: array<f32>;
@group(0) @binding(4) var<storage, read> row_offsets: array<u32>;
@group(0) @binding(5) var<storage, read_write> output_vector: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    if (row >= params.out_features) {
        return;
    }

    let in_features = params.in_features;
    let blocks_per_row = in_features / 16u;
    let mask_offset = row * blocks_per_row;

    var val_idx = row_offsets[row];
    var sum = 0.0;

    for (var b = 0u; b < blocks_per_row; b = b + 1u) {
        let m = masks[mask_offset + b];
        if (m == 0u) {
            continue;
        }
        let input_start = b * 16u;
        for (var i = 0u; i < 16u; i = i + 1u) {
            if ((m & (1u << i)) != 0u) {
                sum = sum + input_vector[input_start + i] * values[val_idx];
                val_idx = val_idx + 1u;
            }
        }
    }

    output_vector[row] = sum;
}
