#![allow(clippy::needless_range_loop)]
use bramha::models::quantization::quantize_to_int4;
use bramha::storage::Database;
use bramha::storage::storage_manifest::{CompressionFormat, LayerMetadata, ModelManifest};
use std::io::Write;
use std::path::Path;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=========================================");
    println!("⚙️ BRAMHA NEURAL ENGINE MODEL QUANTIZER");
    println!("=========================================\n");

    let args: Vec<String> = std::env::args().collect();
    let mut use_svd = false;
    let mut use_columnar = false;
    let mut use_differential = false;
    for i in 1..args.len() {
        if args[i] == "--svd" {
            use_svd = true;
        }
        if args[i] == "--columnar" {
            use_columnar = true;
        }
        if args[i] == "--differential" {
            use_differential = true;
        }
    }

    if !use_svd && !use_columnar && !use_differential {
        println!("Error: Must specify a quantization mode (--svd, --columnar, or --differential)");
        return Ok(());
    }

    let source_name = "tinyllama";
    let target_name = "tinyllama-q4";

    let db = if std::path::Path::new("bramha_db.bin").exists() {
        Arc::new(
            Database::load("bramha_db.bin")
                .await
                .unwrap_or_else(|_| Database::new(None, 1536)),
        )
    } else {
        Arc::new(Database::new(None, 1536))
    };

    let source_path = Path::new("/home/akshay-bhalerao/tensor_data/tinyllama");
    let target_path = Path::new("/home/akshay-bhalerao/tensor_data/tinyllama-q4");
    std::fs::create_dir_all(target_path).unwrap_or_default();

    println!("Loading source model manifest...");
    let manifest_path = source_path.join("manifest.json");
    let manifest_data = std::fs::read_to_string(&manifest_path)?;
    let source_manifest: ModelManifest = serde_json::from_str(&manifest_data)?;

    println!(
        "Found {} layers in source model.",
        source_manifest.layers.len()
    );

    // Register source model so layers are mapped in memory
    {
        let mut tensor_guard = db.tensor_db.write().await;
        tensor_guard.restore_model_at_path(source_name.to_string(), source_path);
        tensor_guard.ensure_model_loaded(source_name)?;
    }

    let tensor_db_guard = db.tensor_db.read().await;
    let model = tensor_db_guard
        .models
        .get(source_name)
        .ok_or_else(|| "Source model not loaded".to_string())?;

    let mut target_manifest = ModelManifest::new(
        target_name.to_string(),
        "q4".to_string(),
        source_manifest.num_layers,
        source_manifest.architecture.clone(),
        source_manifest.hidden_size,
        source_manifest.num_heads,
        source_manifest.num_kv_heads,
        target_path.to_path_buf(),
    );

    println!("\nQuantizing layers to INT4 (packed)...");
    for (i, layer) in source_manifest.layers.values().enumerate() {
        let name = &layer.layer_id;

        // We only quantize layers that are 2D weights for self_attn or mlp projections
        let should_quantize = layer.shape.len() == 2
            && (name.contains("self_attn") || name.contains("mlp"))
            && !name.contains("embed_tokens")
            && !name.contains("lm_head");

        if should_quantize {
            let out_features = layer.shape[0];
            let in_features = layer.shape[1];

            // Read layer F32 weight bytes
            let page = model
                .layers
                .get(name)
                .ok_or_else(|| format!("Weight not found: {}", name))?;
            let f32_data: &[f32] = bytemuck::cast_slice(page.as_bytes());

            if use_columnar {
                print!(
                    "\r[{}/{}] Columnar Dict Encoding: '{}'...",
                    i + 1,
                    source_manifest.layers.len(),
                    name
                );
                std::io::stdout().flush().unwrap_or_default();

                // Transpose to Column-Major: n_in columns, n_out rows
                // Normally W is row-major: size (out_features x in_features)
                // We want to store it as (in_features x out_features) where each column (size out_features) is contiguous.
                let mut transposed = vec![0.0f32; in_features * out_features];
                for row in 0..out_features {
                    for col in 0..in_features {
                        transposed[col * out_features + row] = f32_data[row * in_features + col];
                    }
                }

                // Dictionary encoding: 256 evenly spaced percentiles
                let mut sorted = transposed.clone();
                sorted
                    .sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let mut dict = vec![0.0f32; 256];
                for j in 0..256 {
                    let idx = (j * (sorted.len() - 1)) / 255;
                    dict[j] = sorted[idx];
                }

                // Quantize to indices
                let mut indices = vec![0u8; in_features * out_features];
                for (idx, &val) in transposed.iter().enumerate() {
                    // Binary search or simple linear scan for closest dict value
                    // Since dict is sorted, binary search works well
                    let pos = dict.binary_search_by(|v| {
                        v.partial_cmp(&val).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    let closest = match pos {
                        Ok(p) => p,
                        Err(p) => {
                            if p == 0 {
                                0
                            } else if p >= 256 {
                                255
                            } else {
                                let d1 = (dict[p - 1] - val).abs();
                                let d2 = (dict[p] - val).abs();
                                if d1 < d2 { p - 1 } else { p }
                            }
                        }
                    };
                    indices[idx] = closest as u8;
                }

                // Combine dict + indices
                let mut combined = Vec::with_capacity(256 * 4 + indices.len());
                combined.extend_from_slice(bytemuck::cast_slice(&dict));
                combined.extend_from_slice(&indices);

                let file_name = format!("{}_columnar.bin", name.replace(".", "_"));
                let file_path = target_path.join(&file_name);
                std::fs::write(&file_path, &combined)?;

                let mut l_meta = LayerMetadata::new(name.clone(), layer.shape.clone());
                l_meta.quantization_bits = Some(8);
                l_meta.compression_format = CompressionFormat::ColumnarDict;
                l_meta.stored_bytes = combined.len() as u64;
                target_manifest.add_layer(l_meta);
                continue;
            }

            if use_svd {
                let mut svd_rank = None;
                if name.contains("mlp") {
                    svd_rank = Some(256);
                } else if name.contains("self_attn") {
                    svd_rank = Some(128);
                }

                if let Some(k) = svd_rank
                    && out_features > k
                    && in_features > k
                {
                    print!(
                        "\r[{}/{}] SVD Factorizing (Randomized) to rank {}: '{}'...",
                        i + 1,
                        source_manifest.layers.len(),
                        k,
                        name
                    );
                    std::io::stdout().flush().unwrap_or_default();

                    let (a, b, actual_rank) = bramha::storage::factorization::randomized_svd(
                        f32_data,
                        out_features,
                        in_features,
                        k,
                    )?;

                    let mut combined = Vec::with_capacity(
                        (out_features * actual_rank + actual_rank * in_features) * 4,
                    );
                    combined.extend_from_slice(bytemuck::cast_slice(&a));
                    combined.extend_from_slice(bytemuck::cast_slice(&b));

                    let file_name = format!("{}_svd.bin", name.replace(".", "_"));
                    let file_path = target_path.join(&file_name);
                    std::fs::write(&file_path, &combined)?;

                    let mut l_meta = LayerMetadata::new(name.clone(), layer.shape.clone());
                    l_meta.quantization_bits = None;
                    l_meta.compression_format = CompressionFormat::Svd;
                    l_meta.svd_rank = Some(actual_rank);
                    l_meta.stored_bytes = combined.len() as u64;
                    target_manifest.add_layer(l_meta);
                    continue;
                }
            }

            if use_differential {
                // Parse layer index from name: "model.layers.1.mlp.down_proj.weight"
                if let Some(rest) = name.strip_prefix("model.layers.") {
                    let parts: Vec<&str> = rest.split('.').collect();
                    if let Some(idx_str) = parts.first()
                        && let Ok(idx) = idx_str.parse::<usize>()
                        && idx > 0
                    {
                        let prev_name = name.replace(
                            &format!("model.layers.{}", idx),
                            &format!("model.layers.{}", idx - 1),
                        );
                        if let Some(prev_page) = model.layers.get(&prev_name) {
                            let prev_f32_data: &[f32] = bytemuck::cast_slice(prev_page.as_bytes());

                            // Calculate delta
                            let mut delta = vec![0.0f32; f32_data.len()];
                            for j in 0..f32_data.len() {
                                delta[j] = f32_data[j] - prev_f32_data[j];
                            }

                            print!(
                                "\r[{}/{}] Differential compress '{}' from '{}'...",
                                i + 1,
                                source_manifest.layers.len(),
                                name,
                                prev_name
                            );
                            std::io::stdout().flush().unwrap_or_default();

                            let file_name = format!("{}_diff.bin", name.replace(".", "_"));
                            let file_path = target_path.join(&file_name);
                            std::fs::write(&file_path, bytemuck::cast_slice(&delta))?;

                            let mut l_meta = LayerMetadata::new(name.clone(), layer.shape.clone());
                            l_meta.quantization_bits = None;
                            l_meta.compression_format = CompressionFormat::Differential {
                                delta_format: Box::new(CompressionFormat::None),
                            };
                            l_meta.reference_tensor = Some(prev_name);
                            l_meta.stored_bytes = (delta.len() * 4) as u64;
                            target_manifest.add_layer(l_meta);
                            continue;
                        }
                    }
                }
            }

            print!(
                "\r[{}/{}] Quantizing 2D weight layer: '{}' (Shape: {:?})...",
                i + 1,
                source_manifest.layers.len(),
                name,
                layer.shape
            );
            std::io::stdout().flush().unwrap_or_default();

            // Perform scale-and-zero-point INT4 quantization
            let (q_bytes, scales) = quantize_to_int4(f32_data, out_features, in_features);

            // Write packed INT4 weights to disk
            let q_file_name = format!("{}_u4.bin", name.replace(".", "_"));
            let q_file_path = target_path.join(&q_file_name);
            std::fs::write(&q_file_path, &q_bytes)?;

            // Write scales to disk
            let scale_file_name = format!("{}_scale.bin", name.replace(".", "_"));
            let scale_file_path = target_path.join(&scale_file_name);
            std::fs::write(&scale_file_path, bytemuck::cast_slice(&scales))?;

            // Add metadata for INT4 layer
            let mut l_meta = LayerMetadata::new(name.clone(), layer.shape.clone());
            l_meta.quantization_bits = Some(4);
            l_meta.compression_format = CompressionFormat::Int4PerChannel;
            target_manifest.add_layer(l_meta);

            // Add metadata for scale layer
            let mut scale_meta = LayerMetadata::new(format!("{}.scale", name), vec![out_features]);
            scale_meta.quantization_bits = None;
            target_manifest.add_layer(scale_meta);
        } else {
            // Copy float layers directly (embedding, norms, lm_head)
            println!(
                "\nCopying float layer directly: '{}' (Shape: {:?})...",
                name, layer.shape
            );
            let safe_name = layer.layer_id.replace(".", "_");
            let file_name = format!("{}.bin", safe_name);
            let src_file = source_path.join(&file_name);
            let tgt_file = target_path.join(&file_name);
            if src_file.exists() {
                std::fs::copy(&src_file, &tgt_file)?;
            }
            target_manifest.add_layer(layer.clone());
        }
    }

    println!("\nWriting target model manifest...");
    target_manifest.update_statistics();

    let target_manifest_json = serde_json::to_string_pretty(&target_manifest)?;
    std::fs::write(target_path.join("manifest.json"), target_manifest_json)?;

    println!(
        "\n🎉 SUCCESS! Quantized 4-bit model created successfully at: {:?}",
        target_path
    );
    Ok(())
}
