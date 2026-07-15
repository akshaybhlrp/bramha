use bramha::storage::chunker::Chunker;
use bramha::storage::block_db::BlockDB;
use bramha::storage::model_view::{ModelView, VirtualTensor};
use bramha::storage::storage_manifest::{ModelManifest, CompressionFormat};
use std::path::PathBuf;
use std::fs;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: buts_ingest <manifest.json> <output_dir>");
        return;
    }

    let manifest_path = &args[1];
    let output_dir = PathBuf::from(&args[2]);

    let manifest_data = fs::read_to_string(manifest_path).unwrap();
    let manifest: ModelManifest = serde_json::from_str(&manifest_data).unwrap();

    let mut db = BlockDB::new(&output_dir).unwrap();
    let mut view = ModelView::new();

    let manifest_dir = PathBuf::from(manifest_path).parent().unwrap().to_path_buf();

    for (name, meta) in &manifest.layers {
        // Construct the file path for this layer
        let mut file_path = manifest_dir.join(name.replace(".", "_"));
        
        // Append correct extension based on format
        match meta.compression_format {
            CompressionFormat::None => { file_path.set_extension("bin"); }
            CompressionFormat::Int4PerChannel => { file_path.set_extension("u4.bin"); }
            CompressionFormat::Int8Linear => { file_path.set_extension("u8.bin"); }
            CompressionFormat::Svd => { file_path.set_extension("svd.bin"); }
            CompressionFormat::ColumnarDict => { file_path.set_extension("columnar.bin"); }
            
            _ => { file_path.set_extension("bin"); }
        };

        if !file_path.exists() {
            println!("Warning: Could not find file {:?}", file_path);
            continue;
        }

        println!("Ingesting layer {} from {:?}", name, file_path);
        let data = fs::read(&file_path).unwrap();

        let chunks = Chunker::chunk_data(&data);
        let mut block_hashes = Vec::new();

        for chunk in chunks {
            db.store_block(&chunk.hash, &chunk.data).unwrap();
            block_hashes.push(chunk.hash);
        }

        view.add_tensor(VirtualTensor {
            name: name.clone(),
            shape: meta.shape.clone(),
            dtype: format!("{:?}", meta.compression_format),
            total_bytes: data.len() as u64,
            block_hashes,
        });
    }

    db.save_index().unwrap();
    view.save(output_dir.join("model_view.json")).unwrap();
    println!("Ingestion complete! BUTS ModelView saved.");
}
