use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct BlockLocation {
    pub file_path: String,
    pub offset: u64,
    pub length: usize,
}

#[derive(Serialize, Deserialize, Default)]
pub struct BlockIndex {
    pub blocks: HashMap<String, BlockLocation>,
}

pub struct BlockDB {
    base_dir: PathBuf,
    index: BlockIndex,
    blob_file: File,
    pub mmap: Option<std::sync::Arc<crate::core::tensor::TensorData>>,
}

impl BlockDB {
    pub fn new<P: AsRef<Path>>(base_dir: P) -> std::io::Result<Self> {
        let base_dir = base_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&base_dir)?;

        let index_path = base_dir.join("block_index.json");
        let index = if index_path.exists() {
            let data = std::fs::read_to_string(&index_path)?;
            serde_json::from_str(&data).unwrap_or_default()
        } else {
            BlockIndex::default()
        };

        let blob_path = base_dir.join("blob_store.bin");
        let blob_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&blob_path)?;

        let mmap = unsafe {
            let opts = memmap2::MmapOptions::new();
            if let Ok(m) = opts.map(&blob_file) {
                Some(std::sync::Arc::new(crate::core::tensor::TensorData::Mmap(
                    m,
                )))
            } else {
                None
            }
        };

        Ok(Self {
            base_dir,
            index,
            blob_file,
            mmap,
        })
    }

    /// Stores a block if it does not already exist. Returns true if it was newly added.
    pub fn store_block(&mut self, hash: &str, data: &[u8]) -> std::io::Result<bool> {
        if self.index.blocks.contains_key(hash) {
            return Ok(false); // Deduplicated!
        }

        // Append to blob file
        self.blob_file.seek(SeekFrom::End(0))?;
        let offset = self.blob_file.stream_position()?;
        self.blob_file.write_all(data)?;

        self.index.blocks.insert(
            hash.to_string(),
            BlockLocation {
                file_path: "blob_store.bin".to_string(),
                offset,
                length: data.len(),
            },
        );

        Ok(true)
    }

    pub fn get_block_location(&self, hash: &str) -> Option<BlockLocation> {
        self.index.blocks.get(hash).cloned()
    }

    pub fn read_block(&mut self, location: &BlockLocation) -> std::io::Result<Vec<u8>> {
        let path = self.base_dir.join(&location.file_path);
        let mut file = File::open(path)?;
        file.seek(SeekFrom::Start(location.offset))?;
        let mut buffer = vec![0u8; location.length];
        file.read_exact(&mut buffer)?;
        Ok(buffer)
    }

    pub fn save_index(&self) -> std::io::Result<()> {
        let index_path = self.base_dir.join("block_index.json");
        let data = serde_json::to_string_pretty(&self.index)
            .map_err(|e| std::io::Error::other(e))?;
        std::fs::write(index_path, data)
    }
}
