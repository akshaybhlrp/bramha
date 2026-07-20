// 2:4 block-sparse GEMV shader.
// Each invocation computes one output element.
// For every 4 input elements, exactly 2 non-zeros are stored.
// Non-zero positions are implicit: column indices packed in `indices` (u32, 2 per group).
// Non-zero values stored compactly in `values`.

struct Params {
    in_features_vec4: u32,  // cols / 4
    out_features: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;  // per output row: groups * 2 u32s
@group(0) @binding(3) var<storage, read> values: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= params.out_features) { return; }

    let groups = params.in_features_vec4;
    var sum: f32 = 0.0;

    for (var g: u32 = 0u; g < groups; g = g + 1u) {
        let idx_word = indices[row * groups + g];
        let col0 = (idx_word >> 0u) & 0x3FFu;   // bits 0-9
        let col1 = (idx_word >> 10u) & 0x3FFu;  // bits 10-19

        let val_idx = row * groups * 2u + g * 2u;
        let v0 = values[val_idx + 0u];
        let v1 = values[val_idx + 1u];

        let base_col = g * 4u;
        sum += v0 * input[base_col + col0];
        sum += v1 * input[base_col + col1];
    }

    output[row] = sum;
}
