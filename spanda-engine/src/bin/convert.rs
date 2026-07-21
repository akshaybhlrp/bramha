use std::collections::HashMap;
use safetensors::SafeTensors;
use spanda_engine::{ModelMetadata, SpandaModel, SpandaTensor};

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        println!("Usage: spanda-convert <input.safetensors> -o <output.spanda>");
        return Ok(());
    }

    let input_path = &args[1];
    let mut output_path = "model.spanda".to_string();

    let mut i = 2;
    while i < args.len() {
        if args[i] == "-o" && i + 1 < args.len() {
            output_path = args[i + 1].clone();
            break;
        }
        i += 1;
    }

    println!("🔄 Converting model {} to SPANDA sparse format...", input_path);

    // Read the safetensors file
    let file_data = std::fs::read(input_path)
        .map_err(|e| format!("Failed to read input file: {}", e))?;
    
    let safetensors = SafeTensors::deserialize(&file_data)
        .map_err(|e| format!("Failed to parse safetensors: {}", e))?;

    // Create default metadata (normally extracted from config.json, but fallback here)
    let metadata = ModelMetadata {
        name: "converted-model".to_string(),
        architecture: "Qwen2".to_string(),
        hidden_size: 2048,
        num_attention_heads: 32,
        num_key_value_heads: 32,
        head_dim: 64,
        num_hidden_layers: 24,
        vocab_size: 151936,
    };

    let mut tensors = HashMap::new();

    for (name, tensor) in safetensors.tensors() {
        let shape = tensor.shape().to_vec();
        
        // Fetch raw bytes and convert to f32
        let raw_bytes = tensor.data();
        let mut f32_data = vec![0.0f32; raw_bytes.len() / 4];
        bytemuck::cast_slice_mut::<f32, u8>(&mut f32_data).copy_from_slice(raw_bytes);

        // Apply 2:4 block sparsity optimization on MLP/Projection layers
        let spanda_tensor = if name.contains("mlp") || name.contains("proj") {
            if f32_data.len() % 4 == 0 {
                let num_chunks = f32_data.len() / 4;
                let mut masks = Vec::with_capacity(num_chunks);
                let mut values = Vec::new();

                for i in 0..num_chunks {
                    let start = i * 4;
                    let mut chunk = [
                        (0, f32_data[start].abs()),
                        (1, f32_data[start + 1].abs()),
                        (2, f32_data[start + 2].abs()),
                        (3, f32_data[start + 3].abs()),
                    ];

                    chunk.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                    let mut mask: u16 = 0;

                    // Keep top 2 indices with largest absolute magnitude
                    for k in 0..2 {
                        let original_index = chunk[k].0;
                        mask |= 1 << original_index;
                    }
                    masks.push(mask);

                    // Push non-zero values in index order (0..4) matching mask bits
                    for bit in 0..4 {
                        if (mask & (1 << bit)) != 0 {
                            values.push(f32_data[start + bit]);
                        }
                    }
                }

                SpandaTensor::BlockSparse2_4 {
                    shape: shape.clone(),
                    masks,
                    nonzero_values: values,
                }
            } else {
                SpandaTensor::Dense(f32_data)
            }
        } else {
            SpandaTensor::Dense(f32_data)
        };

        tensors.insert(name.to_string(), spanda_tensor);
    }

    let spanda_model = SpandaModel {
        metadata,
        tensors,
    };

    spanda_model.save_to_file(&output_path)?;
    println!("✅ Saved SPANDA model to {}", output_path);

    Ok(())
}
