use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;

/// Writes data to a target path atomically and crash-safely by first writing to a `.tmp` file,
/// executing a full hardware flush (`sync_all`), and then performing an atomic rename.
pub fn atomic_write_file(path: &Path, data: &[u8]) -> std::io::Result<()> {
    // 1. Ensure the parent directory exists
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // 2. Generate path for the temporary file
    let file_name = path.file_name().ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, "Invalid file path")
    })?;
    let mut tmp_file_name = file_name.to_os_string();
    tmp_file_name.push(".tmp");
    let tmp_path = path.with_file_name(tmp_file_name);

    // 3. Write data to the temp file
    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&tmp_path)?;

    file.write_all(data)?;

    // 4. Force synchronization of both data and metadata to disk platters/chips
    file.sync_all()?;

    // Drop the file handle to ensure locks/handles are freed before rename
    drop(file);

    // 5. Atomically rename the temp file to the final target path
    std::fs::rename(&tmp_path, path)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atomic_write_file_correctness() {
        let temp_dir = std::env::temp_dir();
        let target_path = temp_dir.join("test_atomic_write_final.bin");

        let test_data = b"Bramha Neural Engine Crash-Safe Shard Writing";
        atomic_write_file(&target_path, test_data).unwrap();

        assert!(target_path.exists());
        let read_bytes = std::fs::read(&target_path).unwrap();
        assert_eq!(read_bytes, test_data);

        let _ = std::fs::remove_file(target_path);
    }
}
