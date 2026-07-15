struct Params {
    in_features_vec4: u32,
    out_features: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input_vector: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> weights_quant: array<u32>;
@group(0) @binding(3) var<storage, read> scales: array<f32>;
@group(0) @binding(4) var<storage, read_write> output_vector: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    if (row >= params.out_features) {
        return;
    }

    let in_features_vec4 = params.in_features_vec4;
    let offset_in_u32 = row * in_features_vec4;

    var sum = 0.0;
    for (var i = 0u; i < in_features_vec4; i = i + 1u) {
        let word = weights_quant[offset_in_u32 + i];
        
        let b0 = i32(word & 0xFFu);
        let b1 = i32((word >> 8u) & 0xFFu);
        let b2 = i32((word >> 16u) & 0xFFu);
        let b3 = i32((word >> 24u) & 0xFFu);
        
        let q0 = select(f32(b0), f32(b0 - 256), b0 >= 128);
        let q1 = select(f32(b1), f32(b1 - 256), b1 >= 128);
        let q2 = select(f32(b2), f32(b2 - 256), b2 >= 128);
        let q3 = select(f32(b3), f32(b3 - 256), b3 >= 128);
        
        let q_vec = vec4<f32>(q0, q1, q2, q3);
        sum = sum + dot(input_vector[i], q_vec);
    }

    output_vector[row] = sum * scales[row];
}
