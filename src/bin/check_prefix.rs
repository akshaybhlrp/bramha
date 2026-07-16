use bramha::storage::Database;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let db = Database::load("bramha_db.bin").await?;

    let mut tensor_db_write = db.tensor_db.write().await;
    tensor_db_write.ensure_model_loaded("Llama")?;

    let bramha::storage::tensor_db::TensorDB {
        models, block_db, ..
    } = &mut *tensor_db_write;
    let mut block_db_guard = block_db.lock().unwrap();
    if let Some(model) = models.get_mut("Llama") {
        let tensor_name = "model.embed_tokens.weight";
        model.load_tensor_chunks(tensor_name, &mut *block_db_guard)?;
        if let Some(page) = model.layers.get(tensor_name) {
            let bytes = page.as_bytes();
            println!(
                "Loaded tensor '{}': shape={:?}, dtype={:?}, bytes_len={}",
                tensor_name,
                page.shape,
                page.dtype,
                bytes.len()
            );

            // Let's check first few elements if float
            let floats: &[f32] = bytemuck::cast_slice(bytes);
            if floats.len() > 10 {
                println!("First 10 floats: {:?}", &floats[0..10]);
            }
        }
    }
    Ok(())
}
