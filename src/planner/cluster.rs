use std::collections::HashMap;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NodeInfo {
    pub node_id: String,
    pub address: String,
    pub role: NodeRole,
    pub hardware: HardwareConfig,
    pub available_vram: u64,
    pub is_healthy: bool,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum NodeRole {
    ControlPlane,
    DataWorker,
    Mixed,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HardwareConfig {
    pub gpu_count: usize,
    pub total_vram: u64,
    pub has_tensor_cores: bool,
}

pub struct ClusterPlanner {
    nodes: HashMap<String, NodeInfo>,
}

impl ClusterPlanner {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
        }
    }

    pub fn register_node(&mut self, node: NodeInfo) {
        self.nodes.insert(node.node_id.clone(), node);
    }

    pub fn remove_node(&mut self, node_id: &str) {
        self.nodes.remove(node_id);
    }

    pub fn get_healthy_workers(&self) -> Vec<&NodeInfo> {
        self.nodes.values().filter(|n| n.is_healthy && n.role != NodeRole::ControlPlane).collect()
    }

    /// Sprint 13: Cache-aware cluster planner
    pub fn route_query_to_node(&self, _model: &str, _required_vram: u64) -> Option<String> {
        // Find the worker with the most available VRAM
        let mut best_node = None;
        let mut max_vram = 0;
        
        for worker in self.get_healthy_workers() {
            if worker.available_vram > max_vram {
                max_vram = worker.available_vram;
                best_node = Some(worker.node_id.clone());
            }
        }
        
        best_node
    }
}
