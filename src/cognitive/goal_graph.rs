use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub enum TaskStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SubTask {
    pub id: String,
    pub description: String,
    pub query: Option<String>,
    pub status: TaskStatus,
    pub result: String,
    pub hop_usefulness: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GoalGraph {
    pub original_prompt: String,
    pub tasks: Vec<SubTask>,
    pub max_hops: usize,
    pub current_hop: usize,
}

impl GoalGraph {
    /// Create a new GoalGraph and dynamically decompose the user prompt into structured subtasks
    pub fn new(prompt: &str, max_hops: usize) -> Self {
        let mut tasks = Vec::new();
        
        // Dynamic Heuristic Decomposition:
        // Detect complex requests asking to compare, summarize, or analyze multiple things
        let is_comparative = prompt.to_lowercase().contains("compare") || prompt.to_lowercase().contains("difference");
        let is_multi_part = prompt.to_lowercase().contains("first") || prompt.to_lowercase().contains("then") || prompt.contains(';');

        if is_comparative {
            tasks.push(SubTask {
                id: "subtask_1".to_string(),
                description: "Retrieve context and analyze the first target entity".to_string(),
                query: Some(prompt.to_string()),
                status: TaskStatus::Pending,
                result: String::new(),
                hop_usefulness: 0.0,
            });
            tasks.push(SubTask {
                id: "subtask_2".to_string(),
                description: "Analyze the second target entity and outline similarities/differences".to_string(),
                query: Some(prompt.to_string()),
                status: TaskStatus::Pending,
                result: String::new(),
                hop_usefulness: 0.0,
            });
        } else if is_multi_part {
            tasks.push(SubTask {
                id: "subtask_1".to_string(),
                description: "Identify and resolve the primary question component".to_string(),
                query: Some(prompt.to_string()),
                status: TaskStatus::Pending,
                result: String::new(),
                hop_usefulness: 0.0,
            });
            tasks.push(SubTask {
                id: "subtask_2".to_string(),
                description: "Extend context to compile secondary insights and dependencies".to_string(),
                query: Some(prompt.to_string()),
                status: TaskStatus::Pending,
                result: String::new(),
                hop_usefulness: 0.0,
            });
        } else {
            // Default single-hop fallback task
            tasks.push(SubTask {
                id: "subtask_main".to_string(),
                description: "Execute standard direct semantic retrieval and answer synthesis".to_string(),
                query: Some(prompt.to_string()),
                status: TaskStatus::Pending,
                result: String::new(),
                hop_usefulness: 0.0,
            });
        }

        GoalGraph {
            original_prompt: prompt.to_string(),
            tasks,
            max_hops: max_hops.min(3), // Enforce upper bound of 3 hops maximum
            current_hop: 0,
        }
    }

    /// Execute sequential multi-hop subtask planning using a simulated vector search context hook
    pub fn execute_next_hop<F>(&mut self, retrieve_context_fn: F) -> Result<String, String>
    where
        F: Fn(&str) -> Vec<String>,
    {
        if self.current_hop >= self.max_hops {
            return Err("Reached maximum permitted retrieval reasoning hops".to_string());
        }

        // Find the first pending task
        if let Some(task) = self.tasks.iter_mut().find(|t| t.status == TaskStatus::Pending) {
            task.status = TaskStatus::InProgress;
            self.current_hop += 1;

            // 1. Retrieval Hop
            let query_str = task.query.as_deref().unwrap_or(&self.original_prompt);
            let contexts = retrieve_context_fn(query_str);
            
            // 2. Synthesize subtask intermediate result
            if contexts.is_empty() {
                task.result = format!("No direct evidence found for: {}", task.description);
                task.hop_usefulness = 0.1;
                task.status = TaskStatus::Completed;
            } else {
                task.result = format!(
                    "[Step Output for: {}] Resolved using contexts: {}",
                    task.description,
                    contexts.join(" | ")
                );
                task.hop_usefulness = 0.95; // High utility metric
                task.status = TaskStatus::Completed;
            }
            
            Ok(task.result.clone())
        } else {
            Err("All tasks in the goal graph have already completed".to_string())
        }
    }

    /// Checks if all subtasks are finished, or if we can terminate early safely
    pub fn should_stop_early(&self) -> bool {
        // If all tasks are completed or failed, we stop.
        let all_done = self.tasks.iter().all(|t| t.status == TaskStatus::Completed || t.status == TaskStatus::Failed);
        
        // Early stop guard: if the first task succeeded with extremely high usefulness score,
        // we can early-out to prevent runaway reasoning loops!
        let high_usefulness = self.tasks.first()
            .map(|t| t.status == TaskStatus::Completed && t.hop_usefulness > 0.9)
            .unwrap_or(false);

        all_done || (self.current_hop >= 1 && high_usefulness)
    }

    /// Combine sequential subtask result logs into a unified context block
    pub fn merge_graph_outputs(&self) -> String {
        let mut final_context = String::new();
        for task in &self.tasks {
            if task.status == TaskStatus::Completed && !task.result.is_empty() {
                final_context.push_str(&task.result);
                final_context.push_str("\n");
            }
        }
        final_context
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_goal_graph_decomposition_and_execution() {
        // Comparative prompt
        let prompt = "Compare Rust and C++ speed; which is faster?";
        let mut graph = GoalGraph::new(prompt, 3);
        
        assert_eq!(graph.tasks.len(), 2);
        assert_eq!(graph.tasks[0].status, TaskStatus::Pending);

        // Simulated context retrieval
        let retrieve_fn = |_query: &str| -> Vec<String> {
            vec!["Rust has zero-cost abstractions".to_string()]
        };

        // Execute Hop 1
        let res1 = graph.execute_next_hop(retrieve_fn).unwrap();
        assert!(res1.contains("Resolved using contexts"));
        assert_eq!(graph.current_hop, 1);
        assert_eq!(graph.tasks[0].status, TaskStatus::Completed);

        // Check if should stop early: since task 1 was completed and has a usefulness score of 0.95, it can stop early!
        assert!(graph.should_stop_early());

        // Merge outputs
        let merged = graph.merge_graph_outputs();
        assert!(merged.contains("Resolved using contexts"));
    }

    #[test]
    fn test_goal_graph_max_hops_bound() {
        let prompt = "Compare Rust and C++ speed; which is faster?";
        // max hops = 1
        let mut graph = GoalGraph::new(prompt, 1);
        
        let retrieve_fn = |_query: &str| -> Vec<String> {
            vec!["Rust is extremely fast".to_string()]
        };

        // Hop 1
        let _ = graph.execute_next_hop(retrieve_fn).unwrap();
        
        // Hop 2 (should fail because max_hops = 1)
        let res2 = graph.execute_next_hop(retrieve_fn);
        assert!(res2.is_err());
    }
}

