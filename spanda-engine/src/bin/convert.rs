use std::collections::HashMap;
use std::path::PathBuf;
use safetensors::SafeTensors;
use spanda_engine::{ModelMetadata, SpandaModel, SpandaTensor, pack_4x4_block};

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
            if f32_data.len() % 16 == 0 {
                let num_blocks = f32_data.len() / 16;
                let mut masks = Vec::with_capacity(num_blocks);
                let mut values = Vec::new();

                for b in 0..num_blocks {
                    let start = b * 16;
                    let mut block = [0.0f32; 16];
                    block.copy_from_slice(&f32_data[start..start + 16]);

                    // Apply 2:4 sparsity: set 2 smallest absolute elements to 0
                    let mut indexed_block: Vec<(usize, f32)> = block.iter().copied().enumerate().collect();
                    indexed_block.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
                    
                    // Zero out indices beyond the top 2
                    for k in 2..16 {
                        block[indexed_block[k].0] = 0.0;
                    }

                    // Pack into mask
                    let mask = pack_4x4_block(&block);
                    masks.push(mask);

                    // Push non-zeros
                    for j in 0..16 {
                        if (mask & (1 << j)) != 0 {
                            values.push(block[j]);
                        }
                    }
                }
                
                SpandaTensor::BlockSparse_2_4 {
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
