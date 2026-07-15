use std::fs::{OpenOptions, File};
use std::io::{Write, BufReader, BufRead};
use serde::{Serialize, Deserialize};
use crate::core::vector::Vector;
use crate::core::collection::Collection;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum WalOp {
    Upsert { vector: Vector },
    Delete { id: String },
}

pub struct WalManager {
    file_path: String,
}

impl WalManager {
    pub fn new(collection_name: &str) -> Self {
        let dir = std::env::current_dir().unwrap_or_default().join("wal_data");
        std::fs::create_dir_all(&dir).unwrap();
        let file_path = dir.join(format!("{}.wal", collection_name)).to_string_lossy().to_string();
        WalManager { file_path }
    }

    /// Appends a transaction operation to the Write-Ahead Log on disk and flushes it immediately.
    pub fn append(&self, op: WalOp) -> Result<(), String> {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.file_path)
            .map_err(|e| format!("Failed to open WAL: {}", e))?;
        
        let json_line = serde_json::to_string(&op)
            .map_err(|e| format!("Failed to serialize WAL entry: {}", e))? + "\n";
        
        file.write_all(json_line.as_bytes())
            .map_err(|e| format!("Failed to write WAL: {}", e))?;
        
        file.sync_all()
            .map_err(|e| format!("Failed to sync WAL to disk: {}", e))?;
            
        Ok(())
    }

    /// Clears/truncates the Write-Ahead Log file.
    pub fn clear(&self) -> Result<(), String> {
        if std::path::Path::new(&self.file_path).exists() {
            std::fs::remove_file(&self.file_path)
                .map_err(|e| format!("Failed to clear WAL file: {}", e))?;
        }
        Ok(())
    }

    /// Replays all transactions in the WAL to recover the collection's state.
    pub fn replay(&self, collection: &mut Collection) -> Result<usize, String> {
        let path = std::path::Path::new(&self.file_path);
        if !path.exists() {
            return Ok(0);
        }

        let file = File::open(path).map_err(|e| format!("Failed to open WAL for replay: {}", e))?;
        let reader = BufReader::new(file);
        let mut count = 0;

        for line in reader.lines() {
            let line_str = line.map_err(|e| format!("WAL read line err: {}", e))?;
            if line_str.trim().is_empty() {
                continue;
            }
            let op: WalOp = serde_json::from_str(&line_str)
                .map_err(|e| format!("WAL deserialize err (potential corruption): {}", e))?;
            
            match op {
                WalOp::Upsert { vector } => {
                    let _ = collection.insert(vector);
                }
                WalOp::Delete { id } => {
                    let _ = collection.delete(&id);
                }
            }
            count += 1;
        }

        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::vector::Metric;
    use crate::core::collection::CollectionStatus;

    #[test]
    fn test_wal_crash_recovery_and_blocking() {
        let name = "test_wal_col";
        let manager = WalManager::new(name);
        let _ = manager.clear();

        // 1. Setup clean collection
        let mut collection = Collection::new(name.to_string(), 3, Metric::L2);
        
        // 2. Perform upserts writing to WAL
        let v1 = Vector {
            id: "vec_1".to_string(),
            values: vec![1.0, 2.0, 3.0],
            metadata: None,
        };
        let v2 = Vector {
            id: "vec_2".to_string(),
            values: vec![4.0, 5.0, 6.0],
            metadata: None,
        };

        // Simulated WAL log appends
        manager.append(WalOp::Upsert { vector: v1.clone() }).unwrap();
        manager.append(WalOp::Upsert { vector: v2.clone() }).unwrap();

        // Replay onto clean collection to verify recovery
        let count = manager.replay(&mut collection).unwrap();
        assert_eq!(count, 2);
        assert_eq!(collection.vectors.len(), 2);
        assert!(collection.vectors.contains_key("vec_1"));
        assert!(collection.vectors.contains_key("vec_2"));

        // 3. Simulated WAL deletion recovery
        manager.append(WalOp::Delete { id: "vec_1".to_string() }).unwrap();
        let count2 = manager.replay(&mut collection).unwrap();
        // Since we replayed from the beginning, it's 3 actions now
        assert_eq!(count2, 3);
        assert_eq!(collection.vectors.len(), 1);
        assert!(!collection.vectors.contains_key("vec_1"));
        assert!(collection.vectors.contains_key("vec_2"));

        // 4. Test query blocking on CORRUPT status
        collection.status = CollectionStatus::CORRUPT;
        let res = collection.search(&[4.0, 5.0, 6.0], 1, None, false);
        assert!(res.is_empty(), "Queries should be rejected on CORRUPT collections!");

        // Clean up WAL file
        manager.clear().unwrap();
    }
}
