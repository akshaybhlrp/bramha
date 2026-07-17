use crate::network::proto::compute_node_client::ComputeNodeClient;
use crate::network::proto::{HeartbeatRequest, LayerExecutionRequest};
use tonic::transport::Channel;

pub struct RemoteBackend {
    client: ComputeNodeClient<Channel>,
}

impl RemoteBackend {
    pub async fn connect(addr: String) -> Result<Self, Box<dyn std::error::Error>> {
        let client = ComputeNodeClient::connect(addr).await?;
        Ok(Self { client })
    }

    pub async fn ping(&mut self, node_id: String) -> Result<f32, Box<dyn std::error::Error>> {
        let request = tonic::Request::new(HeartbeatRequest { node_id });
        let response = self.client.heartbeat(request).await?.into_inner();
        Ok(response.vram_usage_percent)
    }

    pub async fn execute_remote_layer(
        &mut self,
        model_name: String,
        layer_name: String,
        input_tensor: Vec<u8>,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let request = tonic::Request::new(LayerExecutionRequest {
            model_name,
            layer_name,
            input_tensor,
            batch_size: 1,
            seq_len: 1,
            hidden_dim: 1,
        });

        let response = self.client.execute_layer(request).await?.into_inner();
        
        if !response.success {
            return Err(format!("Remote execution failed: {}", response.error_message).into());
        }
        
        Ok(response.output_tensor)
    }
}
