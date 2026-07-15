use crate::core::collection::{Collection, SearchResult};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

#[derive(Debug, Clone)]
struct CompareDistDesc {
    id: String,
    dist: f32,
}

impl Eq for CompareDistDesc {}
impl PartialEq for CompareDistDesc {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist
    }
}

impl Ord for CompareDistDesc {
    fn cmp(&self, other: &Self) -> Ordering {
        // Normal ordering: largest distance is at the top (Max-Heap)
        self.dist
            .partial_cmp(&other.dist)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for CompareDistDesc {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Clone)]
struct CompareDistAsc {
    id: String,
    dist: f32,
}

impl Eq for CompareDistAsc {}
impl PartialEq for CompareDistAsc {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist
    }
}

impl Ord for CompareDistAsc {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reversed ordering: smallest distance is at the top (Min-Heap)
        other
            .dist
            .partial_cmp(&self.dist)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for CompareDistAsc {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswIndex {
    pub m: usize,
    pub m0: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub enter_node: Option<String>,
    pub max_level: usize,
    // levels[level][node_id] = Vec<neighbor_ids>
    pub levels: Vec<HashMap<String, Vec<String>>>,
    pub level_mult: f64,
}

impl HnswIndex {
    pub fn new(m: usize, ef_construction: usize, ef_search: usize) -> Self {
        let level_mult = 1.0 / (m as f64).ln();
        HnswIndex {
            m,
            m0: m * 2,
            ef_construction,
            ef_search,
            enter_node: None,
            max_level: 0,
            levels: vec![HashMap::new()],
            level_mult,
        }
    }

    /// Rebuilds the HNSW index from all vectors currently in the collection.
    pub fn build(
        collection: &Collection,
        m: usize,
        ef_construction: usize,
        ef_search: usize,
    ) -> Self {
        let mut index = HnswIndex::new(m, ef_construction, ef_search);

        for (id, _) in &collection.vectors {
            // Generate exponential level assignment
            let r: f64 = rand::random();
            let level = (-r.ln() * index.level_mult).floor() as usize;
            index.insert(collection, id.clone(), level);
        }
        index
    }

    /// Inserts a new node into the HNSW graph index.
    pub fn insert(&mut self, collection: &Collection, id: String, level: usize) {
        let target_vec = match collection.vectors.get(&id) {
            Some(v) => v,
            None => return,
        };

        // Grow levels representation if needed
        while self.levels.len() <= level {
            self.levels.push(HashMap::new());
        }

        let curr_enter_node = match &self.enter_node {
            Some(node) => node.clone(),
            None => {
                // First element insertion
                self.enter_node = Some(id.clone());
                self.max_level = level;
                for l in 0..=level {
                    self.levels[l].insert(id.clone(), vec![]);
                }
                return;
            }
        };

        let mut curr_node = curr_enter_node;
        let mut curr_dist = collection.metric.distance(
            &target_vec.values,
            &collection.vectors.get(&curr_node).unwrap().values,
        );

        // 1. Greedy routing down to target insertion level + 1
        let start_level = self.max_level;
        for l in (level + 1..=start_level).rev() {
            let mut changed = true;
            while changed {
                changed = false;
                if let Some(neighbors) = self.levels[l].get(&curr_node) {
                    for neighbor in neighbors {
                        let d = collection.metric.distance(
                            &target_vec.values,
                            &collection.vectors.get(neighbor).unwrap().values,
                        );
                        if d < curr_dist {
                            curr_dist = d;
                            curr_node = neighbor.clone();
                            changed = true;
                        }
                    }
                }
            }
        }

        // 2. Multi-layer beam search insertion down to Layer 0
        let mut enter_nodes = vec![curr_node];
        for l in (0..=std::cmp::min(level, start_level)).rev() {
            let candidates = self.search_layer(
                collection,
                &target_vec.values,
                &enter_nodes,
                self.ef_construction,
                l,
            );

            // Connect the new node to the best neighbors
            let m_limit = if l == 0 { self.m0 } else { self.m };
            let neighbors = self.select_neighbors(&candidates, m_limit);

            self.levels[l].insert(id.clone(), neighbors.clone());

            // Bidirectional linking
            for neighbor in &neighbors {
                if let Some(links) = self.levels[l].get_mut(neighbor) {
                    links.push(id.clone());

                    // Shrink neighbor links if they exceed limits
                    if links.len() > m_limit {
                        let mut temp_candidates = Vec::new();
                        let n_vec = &collection.vectors.get(neighbor).unwrap().values;
                        for link in links.iter() {
                            let d = collection
                                .metric
                                .distance(n_vec, &collection.vectors.get(link).unwrap().values);
                            temp_candidates.push(CompareDistAsc {
                                id: link.clone(),
                                dist: d,
                            });
                        }
                        temp_candidates.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
                        temp_candidates.truncate(m_limit);
                        *links = temp_candidates.into_iter().map(|c| c.id).collect();
                    }
                }
            }

            enter_nodes = neighbors;
        }

        // Update maximum graph entry level
        if level > self.max_level {
            self.max_level = level;
            self.enter_node = Some(id);
        }
    }

    /// Search a single layer of HNSW using a search budget (ef).
    fn search_layer(
        &self,
        collection: &Collection,
        query: &[f32],
        enter_nodes: &[String],
        ef: usize,
        level: usize,
    ) -> Vec<CompareDistAsc> {
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new(); // Min-Heap
        let mut results = BinaryHeap::new(); // Max-Heap

        for node in enter_nodes {
            visited.insert(node.clone());
            let d = collection
                .metric
                .distance(query, &collection.vectors.get(node).unwrap().values);
            candidates.push(CompareDistAsc {
                id: node.clone(),
                dist: d,
            });
            results.push(CompareDistDesc {
                id: node.clone(),
                dist: d,
            });
        }

        while let Some(curr) = candidates.pop() {
            let worst_result = results.peek().unwrap();
            if curr.dist > worst_result.dist {
                break;
            }

            if let Some(neighbors) = self.levels[level].get(&curr.id) {
                for neighbor in neighbors {
                    if visited.insert(neighbor.clone()) {
                        let d = collection
                            .metric
                            .distance(query, &collection.vectors.get(neighbor).unwrap().values);
                        let worst = results.peek().unwrap();

                        if d < worst.dist || results.len() < ef {
                            candidates.push(CompareDistAsc {
                                id: neighbor.clone(),
                                dist: d,
                            });
                            results.push(CompareDistDesc {
                                id: neighbor.clone(),
                                dist: d,
                            });

                            if results.len() > ef {
                                results.pop();
                            }
                        }
                    }
                }
            }
        }

        let mut sorted_results = Vec::new();
        while let Some(node) = results.pop() {
            sorted_results.push(CompareDistAsc {
                id: node.id,
                dist: node.dist,
            });
        }
        sorted_results.reverse();
        sorted_results
    }

    /// Heuristic to select neighbors. Currently grabs the top closest elements.
    fn select_neighbors(&self, candidates: &[CompareDistAsc], limit: usize) -> Vec<String> {
        candidates
            .iter()
            .take(limit)
            .map(|c| c.id.clone())
            .collect()
    }

    /// Entry point for HNSW retrieval.
    pub fn search(
        &self,
        collection: &Collection,
        query: &[f32],
        k: usize,
        allowed_ids: Option<&HashSet<String>>,
    ) -> Vec<SearchResult> {
        let enter_node = match &self.enter_node {
            Some(node) => node.clone(),
            None => return vec![],
        };

        let mut curr_node = enter_node;
        let mut curr_dist = collection
            .metric
            .distance(query, &collection.vectors.get(&curr_node).unwrap().values);

        // 1. Route greedily down to Level 1
        for l in (1..=self.max_level).rev() {
            let mut changed = true;
            while changed {
                changed = false;
                if let Some(neighbors) = self.levels[l].get(&curr_node) {
                    for neighbor in neighbors {
                        let d = collection
                            .metric
                            .distance(query, &collection.vectors.get(neighbor).unwrap().values);
                        if d < curr_dist {
                            curr_dist = d;
                            curr_node = neighbor.clone();
                            changed = true;
                        }
                    }
                }
            }
        }

        // 2. Comprehensive beam search at Layer 0
        let ef = std::cmp::max(self.ef_search, k);
        let candidates = self.search_layer(collection, query, &[curr_node], ef, 0);

        // Filter and map candidates to SearchResult
        let mut results = Vec::new();
        for candidate in candidates {
            if let Some(ref allowed) = allowed_ids {
                if !allowed.contains(&candidate.id) {
                    continue;
                }
            }

            let v = collection.vectors.get(&candidate.id).unwrap();
            results.push(SearchResult {
                id: candidate.id,
                score: candidate.dist,
                metadata: v.metadata.clone(),
                ..Default::default()
            });
        }

        results.truncate(k);
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::vector::{Metric, Vector};

    #[test]
    fn test_hnsw_correctness() {
        let mut collection = Collection::new("test_hnsw".to_string(), 4, Metric::L2);

        // Add 20 deterministic vectors
        for i in 0..20 {
            let val = i as f32;
            let vec = Vector {
                id: format!("doc_{}", i),
                values: vec![val, val + 1.0, val + 2.0, val + 3.0],
                metadata: None,
            };
            collection.insert(vec).unwrap();
        }

        // Build HNSW index: M = 4, ef_construction = 10, ef_search = 10
        let hnsw = HnswIndex::build(&collection, 4, 10, 10);
        assert!(hnsw.enter_node.is_some());
        assert!(!hnsw.levels.is_empty());

        // Query vector close to doc_5: [5.0, 6.0, 7.0, 8.0]
        let query = vec![5.1, 5.9, 7.0, 8.1];
        let results = hnsw.search(&collection, &query, 3, None);

        assert!(results.len() >= 1);
        // The closest document should be doc_5
        assert_eq!(results[0].id, "doc_5");
    }
}
