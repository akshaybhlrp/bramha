use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ActivationMaterializedView {
    pub workflow_id: String,
    pub branch_id: String,
    pub token_hash: String,
    pub token_length: usize,
    pub disk_path: String,
    pub created_at: u64,
}
