use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct SubGoal {
    pub id: String,
    pub description: String,
    pub dependencies: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MetadataFilter {
    pub key: String,
    pub value: String,
}

/// Result produced by a single resolved goal during multi-hop execution.
#[derive(Debug, Clone)]
pub struct HopResult {
    pub goal_id: String,
    pub retrieved_contexts: Vec<String>,
}

pub struct ResearchGraph {
    goals: HashMap<String, SubGoal>,
}

impl Default for ResearchGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl ResearchGraph {
    pub fn new() -> Self {
        ResearchGraph {
            goals: HashMap::new(),
        }
    }

    pub fn add_goal(&mut self, goal: SubGoal) {
        self.goals.insert(goal.id.clone(), goal);
    }

    /// Executes multi-hop retrieval by resolving goals in dependency order.
    ///
    /// `retrieve_fn` is called for each resolvable goal with its description;
    /// it returns a list of retrieved context strings. `filters` are used to
    /// skip contexts that do not match all key=value pairs.
    pub fn execute_multi_hop<F>(
        &self,
        filters: &[MetadataFilter],
        retrieve_fn: F,
    ) -> Result<Vec<HopResult>, String>
    where
        F: Fn(&str) -> Vec<String>,
    {
        let mut resolved_ids: Vec<String> = Vec::new();
        let mut hop_results: Vec<HopResult> = Vec::new();
        let mut pending: Vec<&SubGoal> = self.goals.values().collect();
        let mut max_hops = 10;

        while !pending.is_empty() && max_hops > 0 {
            let mut progressed = false;
            let mut next_pending = Vec::new();

            for goal in pending {
                let can_resolve = goal
                    .dependencies
                    .iter()
                    .all(|dep| resolved_ids.contains(dep));

                if can_resolve {
                    let raw_contexts = retrieve_fn(&goal.description);
                    let filtered = apply_filters(&raw_contexts, filters);

                    hop_results.push(HopResult {
                        goal_id: goal.id.clone(),
                        retrieved_contexts: filtered,
                    });
                    resolved_ids.push(goal.id.clone());
                    progressed = true;
                } else {
                    next_pending.push(goal);
                }
            }

            if !progressed {
                // Deadlock or unresolvable cycle — stop early
                break;
            }
            pending = next_pending;
            max_hops -= 1;
        }

        Ok(hop_results)
    }
}

/// Retain contexts that match ALL provided key=value filters.
/// Since raw context strings are plain text, we check that each filter's
/// value appears as a substring of the context (case-insensitive).
fn apply_filters(contexts: &[String], filters: &[MetadataFilter]) -> Vec<String> {
    if filters.is_empty() {
        return contexts.to_vec();
    }
    contexts
        .iter()
        .filter(|ctx| {
            let lower = ctx.to_lowercase();
            filters
                .iter()
                .all(|f| lower.contains(&f.value.to_lowercase()))
        })
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_hop_single_goal_no_deps() {
        let mut graph = ResearchGraph::new();
        graph.add_goal(SubGoal {
            id: "g1".to_string(),
            description: "Find information about Rust".to_string(),
            dependencies: vec![],
        });

        let results = graph
            .execute_multi_hop(&[], |_query| {
                vec![
                    "Rust is a systems language".to_string(),
                    "Rust has zero-cost abstractions".to_string(),
                ]
            })
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].goal_id, "g1");
        assert_eq!(results[0].retrieved_contexts.len(), 2);
    }

    #[test]
    fn test_multi_hop_dependency_ordering() {
        let mut graph = ResearchGraph::new();
        graph.add_goal(SubGoal {
            id: "g1".to_string(),
            description: "Identify entity A".to_string(),
            dependencies: vec![],
        });
        graph.add_goal(SubGoal {
            id: "g2".to_string(),
            description: "Retrieve details about entity A from context".to_string(),
            dependencies: vec!["g1".to_string()],
        });

        let results = graph
            .execute_multi_hop(&[], |_| vec!["context data".to_string()])
            .unwrap();

        // Both goals resolved, g1 before g2
        assert_eq!(results.len(), 2);
        let ids: Vec<&str> = results.iter().map(|r| r.goal_id.as_str()).collect();
        let g1_pos = ids.iter().position(|&id| id == "g1").unwrap();
        let g2_pos = ids.iter().position(|&id| id == "g2").unwrap();
        assert!(g1_pos < g2_pos, "g1 must be resolved before g2");
    }

    #[test]
    fn test_multi_hop_filter_applied() {
        let mut graph = ResearchGraph::new();
        graph.add_goal(SubGoal {
            id: "g1".to_string(),
            description: "Find Rust articles".to_string(),
            dependencies: vec![],
        });

        let filter = MetadataFilter {
            key: "topic".to_string(),
            value: "memory".to_string(),
        };

        let results = graph
            .execute_multi_hop(&[filter], |_| {
                vec![
                    "Rust memory safety guarantees".to_string(), // matches "memory"
                    "Rust compile-time checks".to_string(),      // filtered out
                ]
            })
            .unwrap();

        assert_eq!(results[0].retrieved_contexts.len(), 1);
        assert!(results[0].retrieved_contexts[0].contains("memory"));
    }

    #[test]
    fn test_multi_hop_unresolvable_cycle_stops_cleanly() {
        let mut graph = ResearchGraph::new();
        // g1 depends on g2, g2 depends on g1 — deadlock
        graph.add_goal(SubGoal {
            id: "g1".to_string(),
            description: "First".to_string(),
            dependencies: vec!["g2".to_string()],
        });
        graph.add_goal(SubGoal {
            id: "g2".to_string(),
            description: "Second".to_string(),
            dependencies: vec!["g1".to_string()],
        });

        let results = graph
            .execute_multi_hop(&[], |_| vec!["ctx".to_string()])
            .unwrap();

        // Neither resolves — clean empty result, no panic
        assert!(results.is_empty());
    }
}
