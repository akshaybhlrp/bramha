use bramha::storage::tensor_db::TensorDB;
use std::path::PathBuf;

fn main() {
    // Always use this path for model ingestion
    let default_tensor_storage = PathBuf::from("/home/akshay-bhalerao/tensor_data");
    let mut db = TensorDB::new(default_tensor_storage);

    let model_id = "tinyllama-1.1b";
    db.create_model(model_id.to_string());

    let model_table = db.models.get_mut(model_id).unwrap();
    println!(
        "⚙️ Ingesting model from local safetensors at {:?}",
        model_table.base_path
    );
    model_table.load_safetensors("model.safetensors").unwrap();
    println!("🎉 Local ingestion complete!");
}
