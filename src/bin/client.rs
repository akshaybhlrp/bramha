use burn::backend::Wgpu;
use burn::backend::wgpu::WgpuDevice;
use burn::tensor::{Data, Shape, Tensor};
use half::f16;
use reqwest::Client;
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("===========================================");
    println!("  🚀 PURE RUST INFERENCE PIPELINE (WGPU)");
    println!("===========================================\n");

    let model_name = "tinyllama";
    let layer_name = "model.layers.5.self_attn.q_proj.weight";
    let url = format!(
        "http://localhost:8000/api/tensor/{}/{}",
        model_name, layer_name
    );

    // --- STEP 1: FETCH LAYER FROM BRAMHA DB ---
    println!("1. Connecting to Bramha Tensor Database...");
    println!("   Fetching layer: {}", layer_name);

    let start = Instant::now();
    let client = Client::new();
    let response = client.get(&url).send().await?;

    if !response.status().is_success() {
        println!("❌ Failed to fetch: HTTP {}", response.status());
        return Ok(());
    }

    let raw_bytes = response.bytes().await?;
    let fetch_time = start.elapsed();
    println!(
        "   ✅ Fetched {} bytes in {:?}",
        raw_bytes.len(),
        fetch_time
    );

    // --- STEP 2: ZERO-COPY CAST TO F16 ---
    let float_data: &[f16] = bytemuck::cast_slice(&raw_bytes);

    // TinyLlama has a hidden size of 2048, and Query projection is [2048, 2048]
    let rows = 2048;
    let cols = 2048;

    // --- STEP 3: LOAD INTO BURN WGPU TENSOR ---
    println!("\n2. Loading bytes into Burn WGPU Tensor...");
    type B = Wgpu;
    let device = WgpuDevice::default();

    let f32_data: Vec<f32> = float_data.iter().map(|f| f.to_f32()).collect();
    let data = Data::new(f32_data, Shape::from([rows, cols])).convert();
    let q_weight_tensor = Tensor::<B, 2>::from_data(data, &device);
    println!(
        "   ✅ Tensor loaded in memory: Shape {:?}",
        q_weight_tensor.shape()
    );

    // --- STEP 4: PERFORM RUST MATRIX MULTIPLICATION ---
    println!("\n3. Performing Neural Network Math (Matrix Multiplication)...");

    // Simulate an incoming word embedding (a vector of 2048 floats representing a token)
    let test_input_embedding = Tensor::<B, 2>::random(
        Shape::from([1, cols]),
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device,
    );
    println!(
        "   - Created incoming embedding: Shape {:?}",
        test_input_embedding.shape()
    );

    let math_start = Instant::now();

    // Execute: output = input @ weight.T
    let output = test_input_embedding.matmul(q_weight_tensor.transpose());

    let math_time = math_start.elapsed();

    // --- STEP 5: RESULTS ---
    println!("   ✅ Math Complete! Calculated in {:?}", math_time);
    println!("   - Resulting Output Tensor Shape: {:?}", output.shape());

    println!("\n🎉 SUCCESS! Bramha successfully fed a Rust WGPU matrix multiplication engine!");
    Ok(())
}
