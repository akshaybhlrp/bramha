use crate::storage::disk::load_from_file;
use blake3;
use serde::{Deserialize, Serialize};
/// Content-Addressed Storage: Deduplication layer for model weights
///
/// Core idea: Instead of storing raw weight files, store chunks by their hash.
/// Multiple models can reference the same weight chunk if it's identical.
/// This enables:
/// - Cross-model deduplication (e.g., tinyllama base + quantized variants)
/// - Layer deduplication (identical layers across depths)
/// - Block-level efficiency (256-element chunks hashed independently)
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

const CHUNK_SIZE: usize = 262144; // Elements per chunk (1MB)

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkHash {
    pub hash: String, // blake3 hash as hex string
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkReference {
    pub chunk_hash: String,
    pub byte_offset: usize,
    pub byte_length: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StorageLocation {
    pub path: PathBuf,
    pub byte_offset: u64,
    pub byte_length: u64,
}

/// Deduplication index: maps chunk hash to storage location + reference count
#[derive(Debug, Serialize, Deserialize)]
pub struct DedupIndex {
    /// chunk_hash -> StorageLocation (where this chunk is stored)
    pub chunks: HashMap<String, StorageLocation>,

    /// chunk_hash -> reference count (how many models/layers reference this chunk)
    pub ref_counts: HashMap<String, u32>,

    /// model_name -> set of chunk hashes it uses
    pub model_chunks: HashMap<String, HashSet<String>>,

    /// Bloom filter for quick negative checks (prevents unnecessary hash lookups)
    // TODO: Replace HashSet with a proper Bloom filter implementation for better memory efficiency.
    pub bloom_cache: HashSet<String>,
}

impl Default for DedupIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl DedupIndex {
    pub fn new() -> Self {
        DedupIndex {
            chunks: HashMap::new(),
            ref_counts: HashMap::new(),
            model_chunks: HashMap::new(),
            bloom_cache: HashSet::new(),
        }
    }

    /// Check if chunk likely exists (false positives ok, no false negatives)
    pub fn maybe_exists(&self, chunk_hash: &str) -> bool {
        self.bloom_cache.contains(chunk_hash)
    }

    /// Register a chunk in the dedup index
    pub fn register_chunk(
        &mut self,
        chunk_hash: String,
        location: StorageLocation,
        model_name: &str,
    ) {
        // Update chunk location mapping ONLY if it doesn't already exist or if the new path is non-empty
        if !self.chunks.contains_key(&chunk_hash) || location.path != PathBuf::new() {
            self.chunks.insert(chunk_hash.clone(), location);
        }

        // Increment reference count
        *self.ref_counts.entry(chunk_hash.clone()).or_insert(0) += 1;

        // Track which model uses this chunk
        self.model_chunks
            .entry(model_name.to_string())
            .or_default()
            .insert(chunk_hash.clone());

        // Update Bloom filter
        self.bloom_cache.insert(chunk_hash);
    }

    /// Check if model owns this chunk exclusively (ref count == 1)
    pub fn is_exclusive(&self, chunk_hash: &str) -> bool {
        self.ref_counts
            .get(chunk_hash)
            .is_some_and(|count| *count == 1)
    }

    /// Get dedup savings statistics
    pub fn get_stats(&self) -> DedupStats {
        let total_chunks = self.chunks.len();
        let total_refs: u32 = self.ref_counts.values().sum();
        let unique_chunks_with_dupes = self.ref_counts.values().filter(|&&count| count > 1).count();
        let total_duplicate_refs = total_refs.saturating_sub(total_chunks as u32);

        let bytes_deduplicated = self
            .chunks
            .values()
            .filter(|loc| {
                self.ref_counts
                    .get(
                        &self
                            .chunks
                            .iter()
                            .find(|(_, l)| *l == *loc)
                            .map(|(h, _)| h)
                            .unwrap()
                            .to_string(),
                    )
                    .is_some_and(|&count| count > 1)
            })
            .map(|loc| loc.byte_length)
            .sum::<u64>();

        DedupStats {
            total_unique_chunks: total_chunks,
            total_references: total_refs as usize,
            chunks_with_dupes: unique_chunks_with_dupes,
            duplicate_refs: total_duplicate_refs as usize,
            bytes_saved: bytes_deduplicated,
        }
    }

    /// Clean up: remove chunks unreferenced by any model
    pub fn gc(&mut self, active_models: &HashSet<String>) -> usize {
        let mut removed = 0;
        let mut chunks_to_remove = Vec::new();

        for (model_name, chunks) in &self.model_chunks {
            if !active_models.contains(model_name) {
                chunks_to_remove.extend(chunks.iter().cloned());
            }
        }

        for hash in chunks_to_remove {
            if let Some(count) = self.ref_counts.get_mut(&hash) {
                *count -= 1;
                if *count == 0 {
                    self.chunks.remove(&hash);
                    self.ref_counts.remove(&hash);
                    self.bloom_cache.remove(&hash);
                    removed += 1;
                }
            }
        }

        // Remove dead model entries
        self.model_chunks
            .retain(|model_name, _| active_models.contains(model_name));

        removed
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DedupStats {
    pub total_unique_chunks: usize,
    pub total_references: usize,
    pub chunks_with_dupes: usize,
    pub duplicate_refs: usize,
    pub bytes_saved: u64,
}

/// Content-Addressed Storage backend
pub struct ContentAddressedStorage {
    pub data_dir: PathBuf,
    pub index: Arc<Mutex<DedupIndex>>,
}

impl ContentAddressedStorage {
    pub fn new(data_dir: PathBuf) -> std::io::Result<Self> {
        std::fs::create_dir_all(&data_dir)?;

        let index_path = data_dir.join("dedup_index.json");
        let index = if index_path.exists() {
            match load_from_file::<DedupIndex>(index_path.to_str().unwrap()) {
                Ok(idx) => idx,
                Err(e) => {
                    eprintln!("⚠️ Failed to load dedup index: {}. Creating a new one.", e);
                    DedupIndex::new()
                }
            }
        } else {
            DedupIndex::new()
        };

        Ok(ContentAddressedStorage {
            data_dir,
            index: Arc::new(Mutex::new(index)),
        })
    }

    /// Save the deduplication index atomically to disk
    pub fn save_index(&self) -> std::io::Result<()> {
        let index = self.index.lock().unwrap();
        let index_path = self.data_dir.join("dedup_index.json");

        let bytes =
            serde_json::to_vec(&*index).map_err(|e| std::io::Error::other(e.to_string()))?;

        let tmp_path = self.data_dir.join("dedup_index.json.tmp");

        use std::io::Write;
        let mut file = std::fs::File::create(&tmp_path)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        std::fs::rename(&tmp_path, &index_path)?;

        Ok(())
    }

    /// Hash a chunk (sequence of f32 values)
    pub fn hash_chunk(chunk: &[f32]) -> String {
        let mut hasher = blake3::Hasher::new();
        let bytes: &[u8] = bytemuck::cast_slice(chunk);
        hasher.update(bytes);
        hasher.finalize().to_hex().to_string()
    }

    /// Store a weight tensor, detecting deduplication opportunities
    /// Returns (total_stored_bytes, dedup_savings_bytes)
    pub fn store_tensor(
        &self,
        model_name: &str,
        _layer_name: &str,
        data: &[f32],
    ) -> std::io::Result<(u64, u64)> {
        let mut index = self.index.lock().unwrap();
        let mut total_bytes_stored = 0u64;
        let mut dedup_savings = 0u64;

        // Chunked storage: hash each chunk and check dedup
        let chunk_iter: Box<dyn Iterator<Item = &[f32]>> = if !data.len().is_multiple_of(CHUNK_SIZE)
        {
            Box::new(data.chunks_exact(CHUNK_SIZE).chain(std::iter::once(
                &data[data.len() - (data.len() % CHUNK_SIZE)..],
            )))
        } else {
            Box::new(data.chunks_exact(CHUNK_SIZE))
        };

        // Open/create unified container store
        let container_path = self.data_dir.join("chunk_store.bin");
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .open(&container_path)?;

        let mut current_offset = file.metadata()?.len();

        // Wrap in BufWriter for high-performance buffered sequential writes
        let mut writer = std::io::BufWriter::with_capacity(65536, file);

        // Seek to the end of the container once before starting the chunk loop
        use std::io::Seek;
        writer.seek(std::io::SeekFrom::Start(current_offset))?;

        for chunk in chunk_iter {
            let chunk_hash = Self::hash_chunk(chunk);

            if index.maybe_exists(&chunk_hash) && index.chunks.contains_key(&chunk_hash) {
                // Chunk already exists! Register reference without storing
                dedup_savings += (chunk.len() * 4) as u64;
                index.register_chunk(
                    chunk_hash,
                    StorageLocation {
                        path: PathBuf::new(),
                        byte_offset: 0,
                        byte_length: 0,
                    },
                    model_name,
                );
            } else {
                let byte_offset = current_offset;
                let byte_length = (chunk.len() * 4) as u64;

                // Write chunk to file sequentially using BufWriter and zero-copy byte casting
                use std::io::Write;
                let byte_slice: &[u8] = bytemuck::cast_slice(chunk);
                writer.write_all(byte_slice)?;

                current_offset += byte_length;
                total_bytes_stored += byte_length;

                index.register_chunk(
                    chunk_hash,
                    StorageLocation {
                        path: container_path.clone(),
                        byte_offset,
                        byte_length,
                    },
                    model_name,
                );
            }
        }

        // Explicitly flush our buffered writer to ensure all data is written to disk
        use std::io::Write;
        writer.flush()?;

        // Save index is no longer called automatically inside store_tensor.
        // Callers must call save_index() when they are done with bulk ingestion.

        Ok((total_bytes_stored, dedup_savings))
    }

    /// Get deduplication statistics
    pub fn stats(&self) -> DedupStats {
        self.index.lock().unwrap().get_stats()
    }

    /// Load a chunk from storage by hash
    pub fn load_chunk(&self, chunk_hash: &str) -> std::io::Result<Vec<f32>> {
        let index = self.index.lock().unwrap();
        let location = index
            .chunks
            .get(chunk_hash)
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "Chunk not found"))?;

        if location.path == PathBuf::new() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "Chunk path is empty",
            ));
        }

        use std::io::{Read, Seek, SeekFrom};
        let mut file = std::fs::File::open(&location.path)?;
        file.seek(SeekFrom::Start(location.byte_offset))?;
        let mut data = vec![0u8; location.byte_length as usize];
        file.read_exact(&mut data)?;

        let result = data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        Ok(result)
    }

    /// Report dedup efficiency
    pub fn report(&self) {
        let stats = self.stats();
        println!("\n📊 Content-Addressed Storage Deduplication Report");
        println!("───────────────────────────────────────────────────");
        println!("Total unique chunks: {}", stats.total_unique_chunks);
        println!("Total references: {}", stats.total_references);
        println!("Chunks with duplicates: {}", stats.chunks_with_dupes);
        println!("Duplicate references: {}", stats.duplicate_refs);
        println!(
            "Bytes saved by deduplication: {:.2} MB",
            stats.bytes_saved as f64 / 1024.0 / 1024.0
        );
        if stats.total_references > 0 {
            let dedup_ratio = stats.duplicate_refs as f64 / stats.total_references as f64;
            println!(
                "Deduplication ratio: {:.1}% of storage reused",
                dedup_ratio * 100.0
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_hashing() {
        let chunk1 = vec![1.0, 2.0, 3.0, 4.0];
        let chunk2 = vec![1.0, 2.0, 3.0, 4.0];
        let chunk3 = vec![1.0, 2.0, 3.0, 5.0];

        let hash1 = ContentAddressedStorage::hash_chunk(&chunk1);
        let hash2 = ContentAddressedStorage::hash_chunk(&chunk2);
        let hash3 = ContentAddressedStorage::hash_chunk(&chunk3);

        assert_eq!(hash1, hash2, "Identical chunks must hash the same");
        assert_ne!(hash1, hash3, "Different chunks must hash differently");
    }

    #[test]
    fn test_dedup_index() {
        let mut index = DedupIndex::new();
        let location = StorageLocation {
            path: PathBuf::from("/tmp/chunk.dat"),
            byte_offset: 0,
            byte_length: 1024,
        };

        // Register same chunk for two models
        index.register_chunk("hash1".to_string(), location.clone(), "model1");
        index.register_chunk("hash1".to_string(), location.clone(), "model2");

        let stats = index.get_stats();
        assert_eq!(stats.total_unique_chunks, 1);
        assert_eq!(stats.total_references, 2);
        assert_eq!(stats.duplicate_refs, 1);
    }

    #[test]
    fn test_content_addressed_storage_container() {
        let temp_dir = tempfile::tempdir().unwrap();
        let data_dir = temp_dir.path().to_path_buf();

        // 1. Initialize storage and store tensor
        let storage = ContentAddressedStorage::new(data_dir.clone()).unwrap();

        // 262144 f32 elements (exactly 1 chunk)
        let chunk1 = vec![42.0f32; 262144];
        let (stored_bytes, savings) = storage.store_tensor("modelA", "layer0", &chunk1).unwrap();
        assert_eq!(stored_bytes, 1048576);
        assert_eq!(savings, 0);

        // Store a second tensor with a duplicate chunk and a new chunk
        // 524288 f32 elements (2 chunks: one duplicate, one new)
        let mut chunk2 = vec![42.0f32; 262144];
        chunk2.extend(vec![99.0f32; 262144]);
        let (stored_bytes2, savings2) = storage.store_tensor("modelB", "layer0", &chunk2).unwrap();
        assert_eq!(stored_bytes2, 1048576); // only the new chunk is stored
        assert_eq!(savings2, 1048576); // the duplicate chunk is saved

        // Check stats
        let stats = storage.stats();
        assert_eq!(stats.total_unique_chunks, 2);
        assert_eq!(stats.total_references, 3);
        assert_eq!(stats.chunks_with_dupes, 1);
        assert_eq!(stats.duplicate_refs, 1);
        assert_eq!(stats.bytes_saved, 1048576);

        // Save index manually since store_tensor no longer auto-saves on every call
        storage.save_index().unwrap();

        // Check files on disk
        let container_path = data_dir.join("chunk_store.bin");
        let index_path = data_dir.join("dedup_index.json");
        assert!(container_path.exists());
        assert!(index_path.exists());

        // Verify size of container file (2 chunks * 1048576 bytes = 2097152 bytes)
        assert_eq!(container_path.metadata().unwrap().len(), 2097152);

        // 2. Shut down and recreate to verify persistence
        drop(storage);

        let storage_reloaded = ContentAddressedStorage::new(data_dir.clone()).unwrap();
        let stats_reloaded = storage_reloaded.stats();
        assert_eq!(stats_reloaded.total_unique_chunks, 2);
        assert_eq!(stats_reloaded.total_references, 3);

        // Load chunks and verify content
        let hash1 = ContentAddressedStorage::hash_chunk(&chunk1);
        let hash2 = ContentAddressedStorage::hash_chunk(&vec![99.0f32; 262144]);

        let loaded1 = storage_reloaded.load_chunk(&hash1).unwrap();
        let loaded2 = storage_reloaded.load_chunk(&hash2).unwrap();

        assert_eq!(loaded1, vec![42.0f32; 262144]);
        assert_eq!(loaded2, vec![99.0f32; 262144]);
    }
}
