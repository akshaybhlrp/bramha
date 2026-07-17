use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::storage::block_db::BlockLocation;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct LayerChunk {
    pub chunk_index: usize,
    pub hash: String,
    pub location: Option<BlockLocation>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct LayerIndex {
    pub layer_name: String,
    pub chunks: Vec<LayerChunk>,
}

impl LayerIndex {
    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> std::io::Result<()> {
        let data = serde_json::to_string_pretty(self).map_err(|e| std::io::Error::other(e))?;
        std::fs::write(path, data)
    }

    pub fn load<P: AsRef<std::path::Path>>(path: P) -> std::io::Result<Self> {
        let data = std::fs::read_to_string(path)?;
        let index = serde_json::from_str(&data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        Ok(index)
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct VirtualTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub total_bytes: u64,
    pub block_hashes: Vec<String>,
}

#[derive(Serialize, Deserialize, Default)]
pub struct ModelView {
    pub tensors: HashMap<String, VirtualTensor>,
}

impl ModelView {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_tensor(&mut self, tensor: VirtualTensor) {
        self.tensors.insert(tensor.name.clone(), tensor);
    }

    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> std::io::Result<()> {
        let data = serde_json::to_string_pretty(self).map_err(|e| std::io::Error::other(e))?;
        std::fs::write(path, data)
    }

    pub fn load<P: AsRef<std::path::Path>>(path: P) -> std::io::Result<Self> {
        let data = std::fs::read_to_string(path)?;
        let view = serde_json::from_str(&data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        Ok(view)
    }
}
