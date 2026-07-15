use crate::inference::engine::InferenceResult;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::{mpsc, oneshot};

/// A singular heavy inference task containing parameters and oneshot response backchannel
pub struct InferenceTask {
    pub model_name: String,
    pub prompt: String,
    pub max_new_tokens: usize,
    pub temperature: f64,
    pub device: Option<String>,
    pub workflow_id: Option<String>,
    pub branch_id: Option<String>,
    pub response_tx: oneshot::Sender<Result<InferenceResult, String>>,
}

/// Thread-safe Bounded Asynchronous Inference Queue
#[derive(Clone)]
pub struct InferenceQueue {
    sender: mpsc::Sender<InferenceTask>,
    queue_depth: Arc<AtomicUsize>,
    max_depth: usize,
}

impl InferenceQueue {
    /// Creates a new Inference Queue and returns its transmitter along with the receiver channel
    pub fn new(max_depth: usize) -> (Self, mpsc::Receiver<InferenceTask>) {
        let (tx, rx) = mpsc::channel(max_depth);
        let queue = InferenceQueue {
            sender: tx,
            queue_depth: Arc::new(AtomicUsize::new(0)),
            max_depth,
        };
        (queue, rx)
     }

     /// Submits a task to the queue and awaits the result asynchronously from the worker
     pub async fn submit(
         &self,
         model_name: String,
         prompt: String,
         max_new_tokens: usize,
         temperature: f64,
         device: Option<String>,
         workflow_id: Option<String>,
         branch_id: Option<String>,
     ) -> Result<InferenceResult, String> {
         let (tx, rx) = oneshot::channel();
         let task = InferenceTask {
             model_name,
             prompt,
             max_new_tokens,
             temperature,
             device,
             workflow_id,
             branch_id,
             response_tx: tx,
         };

        // Increment depth atomic
        self.queue_depth.fetch_add(1, Ordering::SeqCst);

        // try_send returns immediately if capacity is saturated
        if let Err(_) = self.sender.try_send(task) {
            self.queue_depth.fetch_sub(1, Ordering::SeqCst);
            return Err("429: Bramha Engine is busy. Inference queue is saturated.".to_string());
        }

        // Await the response from oneshot
        let res = rx
            .await
            .map_err(|e| format!("Inference worker task cancelled: {}", e))?;

        // Decrement depth atomic
        self.queue_depth.fetch_sub(1, Ordering::SeqCst);

        res
    }

    /// Exposes the current number of queued tasks
    pub fn queue_depth(&self) -> usize {
        self.queue_depth.load(Ordering::Relaxed)
    }

    /// Exposes the maximum queue capacity
    pub fn max_capacity(&self) -> usize {
        self.max_depth
    }
}
