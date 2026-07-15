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
    let offset_in_u32 = row * (in_features_vec4 / 2u);

    var sum = 0.0;
    let loop_limit = in_features_vec4 / 2u;
    for (var i = 0u; i < loop_limit; i = i + 1u) {
        let word = weights_quant[offset_in_u32 + i];
        
        let e0 = i32((word >> 4u) & 0x0Fu);
        let e1 = i32(word & 0x0Fu);
        let e2 = i32((word >> 12u) & 0x0Fu);
        let e3 = i32((word >> 8u) & 0x0Fu);
        let e4 = i32((word >> 20u) & 0x0Fu);
        let e5 = i32((word >> 16u) & 0x0Fu);
        let e6 = i32((word >> 28u) & 0x0Fu);
        let e7 = i32((word >> 24u) & 0x0Fu);
        
        let q_vec0 = vec4<f32>(f32(e0 - 8), f32(e1 - 8), f32(e2 - 8), f32(e3 - 8));
        let q_vec1 = vec4<f32>(f32(e4 - 8), f32(e5 - 8), f32(e6 - 8), f32(e7 - 8));
        
        sum = sum + dot(input_vector[i * 2u], q_vec0) + dot(input_vector[i * 2u + 1u], q_vec1);
    }

    output_vector[row] = sum * scales[row];
}
