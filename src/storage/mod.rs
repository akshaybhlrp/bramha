use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::core::collection::Collection;
use crate::core::vector::Metric;
use crate::storage::caching::SemanticCache;
use crate::storage::disk::{load_from_file, save_to_file};

pub mod activation_view;
pub mod answer_cache;
pub mod atomic_write;
pub mod block_db;
pub mod cache_db;
pub mod caching;
pub mod chunker;
pub mod content_addressing;
pub mod crud;
pub mod data_organization;
pub mod disk;
pub mod factorization;
pub mod indexing;
pub mod kv_persistence;
pub mod metadata_sql;
pub mod model_view;
pub mod multi_tier;
pub mod pull;
pub mod query_execution;
pub mod safetensors_loader;
pub mod session_store;
pub mod sparse_pager;
pub mod storage_manifest;
pub mod tensor_db;
pub mod wal;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ModelMetadata {
    pub name: String,
    pub base_path: String,
    pub early_exit_thresholds: Vec<f32>,
    #[serde(default)]
    pub active_device: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DatabaseState {
    pub collections: HashMap<String, Collection>,
    pub cache: SemanticCache,
    #[serde(default)]
    pub tensor_storage_dir: Option<String>,
    #[serde(default)]
    pub model_registry: HashMap<String, ModelMetadata>,
    #[serde(default)]
    pub preserved_collections: HashMap<String, String>,
    #[serde(skip)]
    pub ingestion_tasks: HashMap<String, String>,
}

pub struct Database {
    pub state: RwLock<DatabaseState>,
    pub tensor_db: RwLock<crate::storage::tensor_db::TensorDB>,
    pub inference_queue: crate::middleware::queue::InferenceQueue,
    pub receiver:
        RwLock<Option<tokio::sync::mpsc::Receiver<crate::middleware::queue::InferenceTask>>>,
    pub file_path: Option<String>,
    pub planner_cache_path: Option<std::path::PathBuf>,
}

impl Database {
    /// Creates a new, empty database.
    pub fn new(file_path: Option<String>, cache_dim: usize) -> Self {
        let default_tensor_storage =
            if std::path::Path::new("/home/akshay-bhalerao/tensor_data").exists() {
                std::path::PathBuf::from("/home/akshay-bhalerao/tensor_data")
            } else {
                std::env::current_dir()
                    .unwrap_or_default()
                    .join("tensor_data")
            };
        Self::new_with_dir(file_path, cache_dim, default_tensor_storage)
    }

    /// Creates a new database with a custom tensor storage directory.
    pub fn new_with_dir(
        file_path: Option<String>,
        cache_dim: usize,
        tensor_storage_dir: std::path::PathBuf,
    ) -> Self {
        let tensor_storage_str = tensor_storage_dir.to_string_lossy().to_string();

        let state = DatabaseState {
            collections: HashMap::new(),
            cache: SemanticCache::new(cache_dim, Metric::Cosine, 0.92),
            tensor_storage_dir: Some(tensor_storage_str),
            model_registry: HashMap::new(),
            preserved_collections: HashMap::new(),
            ingestion_tasks: HashMap::new(),
        };

        let tensor_db = crate::storage::tensor_db::TensorDB::new(tensor_storage_dir);
        let (inference_queue, rx) = crate::middleware::queue::InferenceQueue::new(5);

        Database {
            state: RwLock::new(state),
            tensor_db: RwLock::new(tensor_db),
            inference_queue,
            receiver: RwLock::new(Some(rx)),
            file_path,
            planner_cache_path: None,
        }
    }

    /// Saves the database state to disk if a file path is configured.
    pub async fn save(&self) -> Result<(), String> {
        if let Some(ref path) = self.file_path {
            // Update the model registry from the current active tensor_db models
            {
                let tensor_db_guard = self.tensor_db.read().await;
                let mut state_guard = self.state.write().await;
                state_guard.model_registry.clear();
                for (name, model) in &tensor_db_guard.models {
                    state_guard.model_registry.insert(
                        name.clone(),
                        ModelMetadata {
                            name: name.clone(),
                            base_path: model.base_path.to_string_lossy().to_string(),
                            early_exit_thresholds: model.early_exit_thresholds.clone(),
                            active_device: model.active_device.clone(),
                        },
                    );
                }
            }
            let state_guard = self.state.read().await;
            save_to_file(&*state_guard, path)?;

            // S4.4: Clear the WAL for all collections since state is fully persisted
            for collection in state_guard.collections.values() {
                let wal = crate::storage::wal::WalManager::new(&collection.name);
                let _ = wal.clear();
            }
        }
        Ok(())
    }

    /// Loads the database state from a binary file.
    pub async fn load(path: &str) -> Result<Self, String> {
        let mut state: DatabaseState = load_from_file(path)?;

        // Recreate any missing preserved collections
        for (model_name, col_name) in &state.preserved_collections {
            if !state.collections.contains_key(col_name) {
                println!(
                    "🔄 Recreating missing preserved collection '{}' for model '{}' on startup...",
                    col_name, model_name
                );
                let mut collection = Collection::new(col_name.clone(), 384, Metric::Cosine);
                if let Err(e) = collection.init_sqlite_index() {
                    println!(
                        "⚠️ Failed to rebuild SQLite index for recreated collection '{}': {}",
                        col_name, e
                    );
                }
                state.collections.insert(col_name.clone(), collection);
            }
        }

        // S4.4: Rebuild SQLite in-memory metadata indices and replay WAL for all loaded collections
        for collection in state.collections.values_mut() {
            let wal = crate::storage::wal::WalManager::new(&collection.name);
            if let Err(e) = wal.replay(collection) {
                println!(
                    "⚠️ WAL Replay failed for collection '{}' (marking degraded/corrupt): {}",
                    collection.name, e
                );
                collection.status = crate::core::collection::CollectionStatus::CORRUPT;
            } else {
                if let Err(e) = collection.init_sqlite_index() {
                    println!(
                        "⚠️ Failed to rebuild SQLite index for collection '{}': {}",
                        collection.name, e
                    );
                }
            }
        }

        let tensor_storage = if let Some(ref dir) = state.tensor_storage_dir {
            std::path::PathBuf::from(dir)
        } else {
            if std::path::Path::new("/home/akshay-bhalerao/tensor_data").exists() {
                std::path::PathBuf::from("/home/akshay-bhalerao/tensor_data")
            } else {
                std::env::current_dir()
                    .unwrap_or_default()
                    .join("tensor_data")
            }
        };
        let mut tensor_db = crate::storage::tensor_db::TensorDB::new(tensor_storage);

        // Reconstruct any registered models from the model_registry to make sure they are fully populated in TensorDB
        for (name, meta) in &state.model_registry {
            let meta_path = std::path::Path::new(&meta.base_path);
            tensor_db.restore_model_at_path(name.clone(), meta_path);
            if let Some(model) = tensor_db.models.get_mut(name) {
                model.active_device = meta.active_device.clone();
                if model.active_device.to_lowercase() == "cpu" {
                    crate::inference::set_cpu_only(true);
                }
            }
        }

        let (inference_queue, rx) = crate::middleware::queue::InferenceQueue::new(5);

        let db = Database {
            state: RwLock::new(state),
            tensor_db: RwLock::new(tensor_db),
            inference_queue,
            receiver: RwLock::new(Some(rx)),
            file_path: Some(path.to_string()),
            planner_cache_path: None,
        };
        let _ = db.save().await;
        Ok(db)
    }

    /// Starts the background inference worker loop to process sequential tasks on WGPU.
    pub fn start_worker(self: &Arc<Self>) {
        let db_clone = self.clone();
        tokio::spawn(async move {
            let rx = {
                let mut rx_guard = db_clone.receiver.write().await;
                rx_guard.take()
            };
            if let Some(mut receiver) = rx {
                println!("🧠 Bramha Background Inference Worker spawned successfully!");
                while let Some(task) = receiver.recv().await {
                    let cpu_only = if let Some(ref dev) = task.device {
                        dev.to_lowercase() == "cpu"
                    } else {
                        let state_guard = db_clone.state.read().await;
                        if let Some(meta) = state_guard.model_registry.get(&task.model_name) {
                            meta.active_device.to_lowercase() == "cpu"
                        } else {
                            false
                        }
                    };

                    let db_clone_inner = db_clone.clone();
                    let task_model = task.model_name.clone();
                    let task_prompt = task.prompt.clone();
                    let task_max_tokens = task.max_new_tokens;
                    let task_temp = task.temperature;
                    let task_workflow = task.workflow_id.clone();
                    let task_branch = task.branch_id.clone();

                    let result_fut = crate::inference::CPU_ONLY_TASK.scope(cpu_only, async move {
                        crate::inference::engine::InferenceEngine::new(None)
                            .generate(
                                db_clone_inner,
                                &task_model,
                                &task_prompt,
                                task_max_tokens,
                                task_temp,
                                task_workflow,
                                task_branch,
                            )
                            .await
                    });
                    let result = result_fut.await;
                    let _ = task.response_tx.send(result);
                }
            }
        });
    }
}
