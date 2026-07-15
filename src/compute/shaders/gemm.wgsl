struct Params {
    in_features_vec4: u32,
    out_features: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input_vector: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> weights: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> output_vector: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    if (row >= params.out_features) {
        return;
    }

    let in_features_vec4 = params.in_features_vec4;
    let offset = row * in_features_vec4;

    var sum = 0.0;
    for (var i = 0u; i < in_features_vec4; i = i + 1u) {
        sum = sum + dot(input_vector[i], weights[offset + i]);
    }

    output_vector[row] = sum;
}
