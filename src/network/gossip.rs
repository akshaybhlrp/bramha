use crate::network::proto::GossipRequest;
use crate::network::proto::compute_node_client::ComputeNodeClient;
use std::time::Duration;
use tokio::time;

/// A simple Gossip Worker that broadcasts semantic memory updates to known peers.
pub struct GossipWorker {
    node_id: String,
    peers: Vec<String>,
}

impl GossipWorker {
    pub fn new(node_id: String, peers: Vec<String>) -> Self {
        Self { node_id, peers }
    }

    pub async fn start_background_sync(self) {
        tokio::spawn(async move {
            let mut interval = time::interval(Duration::from_secs(10));
            loop {
                interval.tick().await;
                // Periodic sync of KV Cache states
                for peer in &self.peers {
                    if let Ok(mut client) = ComputeNodeClient::connect(peer.clone()).await {
                        let request = tonic::Request::new(GossipRequest {
                            node_id: self.node_id.clone(),
                            memory_payload: vec![], // In reality, this would contain serialized bincode updates
                        });
                        if let Err(e) = client.gossip_memory(request).await {
                            eprintln!("⚠️ [Hyperscale] Failed to gossip with {}: {}", peer, e);
                        }
                    }
                }
            }
        });
    }
}
