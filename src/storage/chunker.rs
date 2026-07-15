use sha2::{Digest, Sha256};
use std::fmt::Write;

pub const CHUNK_SIZE_BYTES: usize = 1 * 1024 * 1024; // 1MB

pub struct Chunk {
    pub hash: String,
    pub data: Vec<u8>,
}

pub struct Chunker;

impl Chunker {
    /// Slices raw bytes into fixed-size blocks (e.g. 4MB) and returns a list of chunks with their hashes.
    pub fn chunk_data(data: &[u8]) -> Vec<Chunk> {
        let mut chunks = Vec::new();
        for chunk_data in data.chunks(CHUNK_SIZE_BYTES) {
            let hash = Self::compute_hash(chunk_data);
            chunks.push(Chunk {
                hash,
                data: chunk_data.to_vec(),
            });
        }
        chunks
    }

    pub fn compute_hash(data: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(data);
        let result = hasher.finalize();

        let mut hash_string = String::with_capacity(result.len() * 2);
        for byte in result {
            write!(&mut hash_string, "{:02x}", byte).unwrap();
        }
        hash_string
    }
}
