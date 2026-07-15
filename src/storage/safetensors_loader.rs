use safetensors::SafeTensors;
use safetensors::tensor::Dtype as ST_Dtype;
use std::fs::File;
use std::path::Path;

use crate::core::tensor::{DType, TensorPage};
use crate::storage::atomic_write::atomic_write_file;
use crate::storage::tensor_db::ModelTable;

/// Helper to convert safetensors dtype to our internal DType
fn convert_dtype(st_dtype: ST_Dtype) -> DType {
    match st_dtype {
        ST_Dtype::F32 => DType::F32,
        ST_Dtype::F16 => DType::F16,
        ST_Dtype::BF16 => DType::BF16,
        ST_Dtype::I8 => DType::I8,
        ST_Dtype::U8 => DType::U8,
        _ => DType::Other,
    }
}

pub fn string_to_dtype(s: &str) -> DType {
    match s {
        "f32" => DType::F32,
        "f16" => DType::F16,
        "bf16" => DType::BF16,
        "i8" => DType::I8,
        "u8" => DType::U8,
        "u4" => DType::U4,
        _ => DType::Other,
    }
}

/// Natively shards an entire `.safetensors` file into 1MB blocks stored in the content-addressed block database
/// and registers them as independent in-memory TensorPages in the given ModelTable.
/// Also writes a `model_view.json` for virtual restoration on restart.
pub fn shard_safetensors_file(
    table: &mut ModelTable,
    file_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::open(file_path)?;
    // Create a memory map of the whole source file
    let mmap = unsafe {
        let mut opts = memmap2::MmapOptions::new();
        #[cfg(target_os = "linux")]
        opts.huge(None);
        opts.populate();
        let m = opts.map(&file)?;
        #[cfg(target_os = "linux")]
        libc::mlock(m.as_ptr() as *const libc::c_void, m.len());
        m
    };
    mmap.advise(memmap2::Advice::Sequential).ok();
    mmap.advise(memmap2::Advice::WillNeed).ok();
    let mmap_arc = std::sync::Arc::new(crate::core::tensor::TensorData::Mmap(mmap));

    // Parse the safetensors header using the safetensors crate
    let st = SafeTensors::deserialize(mmap_arc.as_bytes())?;

    let content_dir = table
        .base_path
        .parent()
        .unwrap_or(Path::new(""))
        .join("content");
    let block_db = std::sync::Mutex::new(crate::storage::block_db::BlockDB::new(&content_dir)?);

    let model_view_path = table.base_path.join("model_view.json");
    let mut model_view = if model_view_path.exists() {
        crate::storage::model_view::ModelView::load(&model_view_path).unwrap_or_default()
    } else {
        crate::storage::model_view::ModelView::new()
    };

    struct TensorTask {
        name: String,
        shape: Vec<usize>,
        dtype: DType,
        start: usize,
        end: usize,
    }

    let mut tasks = Vec::new();
    for (name, view) in st.tensors() {
        let data = view.data();
        let start = (data.as_ptr() as usize) - (mmap_arc.as_bytes().as_ptr() as usize);
        let end = start + data.len();
        tasks.push(TensorTask {
            name: name.to_string(),
            shape: view.shape().to_vec(),
            dtype: convert_dtype(view.dtype()),
            start,
            end,
        });
    }

    let layer_indices_dir = table.base_path.join("layer_indices");
    std::fs::create_dir_all(&layer_indices_dir)?;

    // Process tensors sequentially to drastically reduce peak memory usage
    // and avoid lock contention on the block_db.
    for task in tasks {
        let tensor_data = &mmap_arc.as_bytes()[task.start..task.end];

        let is_originally_f32 = task.dtype == DType::F32;
        let mapped_dtype = DType::F32;
        let is_f32_conversion = matches!(task.dtype, DType::F16 | DType::BF16 | DType::I8);

        let mut block_hashes = Vec::new();
        let mut layer_chunks = Vec::new();

        let chunk_size = crate::storage::chunker::CHUNK_SIZE_BYTES;
        let elem_size = match task.dtype {
            DType::F16 | DType::BF16 => 2,
            DType::I8 | DType::U8 => 1,
            _ => 4,
        };
        let elements_per_chunk = chunk_size / 4;
        let input_bytes_per_chunk = elements_per_chunk * elem_size;

        let mut offset = 0;
        let mut chunk_idx = 0;
        let mut db = block_db.lock().unwrap();

        while offset < tensor_data.len() {
            let end = std::cmp::min(offset + input_bytes_per_chunk, tensor_data.len());
            let input_slice = &tensor_data[offset..end];

            let f32_floats: Vec<f32> = match task.dtype {
                DType::F16 => {
                    let float_data: &[half::f16] = bytemuck::cast_slice(input_slice);
                    float_data.iter().map(|f| f.to_f32()).collect()
                }
                DType::BF16 => {
                    let float_data: &[half::bf16] = bytemuck::cast_slice(input_slice);
                    float_data.iter().map(|f| f.to_f32()).collect()
                }
                DType::I8 => {
                    let i8_data: &[i8] = bytemuck::cast_slice(input_slice);
                    i8_data.iter().map(|&x| x as f32).collect()
                }
                _ => Vec::new(),
            };

            let fallback_bytes = if f32_floats.is_empty() && !is_originally_f32 {
                input_slice.to_vec()
            } else {
                Vec::new()
            };

            let chunk_data_slice = if !f32_floats.is_empty() {
                bytemuck::cast_slice(&f32_floats)
            } else if !fallback_bytes.is_empty() {
                &fallback_bytes
            } else {
                input_slice
            };

            let chunks = crate::storage::chunker::Chunker::chunk_data(chunk_data_slice);
            for chunk in chunks {
                db.store_block(&chunk.hash, &chunk.data).unwrap();
                block_hashes.push(chunk.hash.clone());

                let location = db.get_block_location(&chunk.hash);
                layer_chunks.push(crate::storage::model_view::LayerChunk {
                    chunk_index: chunk_idx,
                    hash: chunk.hash.clone(),
                    location,
                });
                chunk_idx += 1;
            }
            offset = end;
        }

        let layer_index = crate::storage::model_view::LayerIndex {
            layer_name: task.name.clone(),
            chunks: layer_chunks,
        };

        let page = if is_f32_conversion || (!is_originally_f32 && task.dtype != DType::Other) {
            // Remove ingested parts from memory ASAP to prevent RAM accumulation
            TensorPage::new_memory(
                task.name.clone(),
                task.shape.clone(),
                mapped_dtype,
                Vec::new(), // Dummy empty vector to free memory (chunks are safe in DB)
            )
        } else {
            // It's F32 and zero-copy, create a slice over the mmap directly!
            TensorPage::new_slice(
                task.name.clone(),
                mmap_arc.clone(),
                task.shape.clone(),
                mapped_dtype,
                task.start,
                task.end,
            )
        };

        let virtual_tensor = crate::storage::model_view::VirtualTensor {
            name: task.name.clone(),
            shape: task.shape,
            dtype: "f32".to_string(),
            total_bytes: page.as_bytes().len() as u64,
            block_hashes,
        };

        let layer_index_path = layer_indices_dir.join(format!("{}.json", task.name));
        layer_index.save(&layer_index_path).unwrap();
        table.layers.insert(task.name, page);
        model_view.add_tensor(virtual_tensor);
    }

    // S2.4: Perform early-exit calibration
    let thresholds = crate::inference::calibration::calibrate_thresholds(table);
    table.early_exit_thresholds = thresholds;

    block_db.lock().unwrap().save_index()?;

    // 5. Write the model_view.json for virtual view restoration atomically
    let model_view_json = serde_json::to_string_pretty(&model_view)?;
    atomic_write_file(&model_view_path, model_view_json.as_bytes())?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_mock_safetensors() -> Vec<u8> {
        // JSON header specifying a single tensor: "model.layers.0.input_layernorm.weight"
        // shape [2, 2], dtype F32, and data offsets [0, 16] (16 bytes = 4 floats)
        let header_json = r#"{"__metadata__":{},"model.layers.0.input_layernorm.weight":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}"#;
        let header_bytes = header_json.as_bytes();
        let header_len = header_bytes.len() as u64;

        let mut file_bytes = Vec::new();
        file_bytes.extend_from_slice(&header_len.to_le_bytes());
        file_bytes.extend_from_slice(header_bytes);

        let tensor_data = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32];
        file_bytes.extend_from_slice(bytemuck::cast_slice(&tensor_data));
        file_bytes
    }

    #[test]
    fn test_dtype_conversion() {
        assert_eq!(convert_dtype(ST_Dtype::F32), DType::F32);
        assert_eq!(convert_dtype(ST_Dtype::F16), DType::F16);
        assert_eq!(string_to_dtype("f32"), DType::F32);
        assert_eq!(string_to_dtype("bf16"), DType::BF16);
    }

    #[test]
    fn test_shard_safetensors_file() {
        let temp_dir = TempDir::new().unwrap();
        let safetensors_path = temp_dir.path().join("model.safetensors");
        let model_dir = temp_dir.path().join("model");
        std::fs::create_dir_all(&model_dir).unwrap();

        // 1. Write mock safetensors file
        let sf_data = create_mock_safetensors();
        std::fs::write(&safetensors_path, sf_data).unwrap();

        // 2. Initialize ModelTable
        let mut table = ModelTable::new("mock-model".to_string(), model_dir);

        // 3. Shard safetensors file
        let res = shard_safetensors_file(&mut table, &safetensors_path);
        assert!(res.is_ok());

        // 4. Verify results
        assert!(
            table
                .layers
                .contains_key("model.layers.0.input_layernorm.weight")
        );
        assert_eq!(table.early_exit_thresholds, vec![0.95]);

        let model_view_path = table.base_path.join("model_view.json");
        assert!(model_view_path.exists());
    }
}
