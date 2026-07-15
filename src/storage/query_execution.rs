//! # Database-Native Query Execution for Bramha
//!
//! Implements three query execution strategies for neural database workloads:
//!
//! - **QueryOptimizer**: Cost-based execution path selection extending the planner
//! - **ParallelQuery**: Rayon-based parallel search across partitions/shard/collections
//! - **ConnectionPool**: Reusable SQLite and inference connection management
//!
//! These integrate with the existing planner (scheduler, policy, cost_model, optimizer)
//! and concurrency (rayon_bridge) infrastructure.

use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

// ─── Query Optimizer ────────────────────────────────────────────────────────

/// Types of query operations that can be optimized.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum QueryOp {
    /// Vector similarity search (ANN)
    VectorSearch {
        collection: String,
        k: usize,
        ef_search: Option<usize>,
    },
    /// Exact nearest neighbor
    ExactSearch {
        collection: String,
        k: usize,
    },
    /// Metadata filter + vector search
    FilteredSearch {
        collection: String,
        k: usize,
        filter: String,
    },
    /// Range scan on an indexed field
    RangeScan {
        collection: String,
        field: String,
        low: Option<f64>,
        high: Option<f64>,
    },
    /// Full collection scan (fallback)
    FullScan {
        collection: String,
    },
    /// Aggregation (count, avg, etc.)
    Aggregate {
        collection: String,
        operation: String,
        field: String,
    },
}

/// Estimated cost of a query execution plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryCost {
    /// Estimated CPU cycles
    pub cpu_cost: f64,
    /// Estimated I/O operations
    pub io_cost: f64,
    /// Estimated memory usage in bytes
    pub memory_cost: f64,
    /// Estimated latency in milliseconds
    pub estimated_latency_ms: f64,
    /// Overall cost score (lower = better)
    pub total_score: f64,
}

impl QueryCost {
    pub fn new(cpu: f64, io: f64, memory: f64, latency: f64) -> Self {
        let total = cpu * 1.0 + io * 2.0 + memory * 0.5 + latency * 3.0;
        QueryCost {
            cpu_cost: cpu,
            io_cost: io,
            memory_cost: memory,
            estimated_latency_ms: latency,
            total_score: total,
        }
    }
}

/// A single step in a query execution plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PlanStep {
    /// Use an index (B-Tree, Hash, etc.)
    IndexScan {
        index_name: String,
        estimated_selectivity: f64,
    },
    /// Sequential scan
    SeqScan {
        collection: String,
        estimated_rows: usize,
    },
    /// Vector similarity search
    AnnSearch {
        index_type: String, // "hnsw", "ivf", "flat"
        k: usize,
        estimated_cost: f64,
    },
    /// Filter rows
    Filter {
        predicate: String,
        estimated_selectivity: f64,
    },
    /// Sort results
    Sort {
        field: String,
        descending: bool,
        estimated_rows: usize,
    },
    /// Limit results
    Limit {
        n: usize,
    },
    /// Parallel execution of sub-steps
    Parallel {
        steps: Vec<Vec<PlanStep>>,
        description: String,
    },
    /// Merge results from parallel branches
    Merge {
        strategy: MergeStrategy,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MergeStrategy {
    /// Union all results
    Union,
    /// Intersection
    Intersection,
    /// Top-K from each branch
    TopK(usize),
    /// Round-robin merge
    RoundRobin,
}

/// A complete query execution plan with cost estimates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionPlan {
    pub id: u64,
    pub description: String,
    pub steps: Vec<PlanStep>,
    pub estimated_cost: QueryCost,
    pub parallelizable: bool,
}

/// Cost-based query optimizer that selects the best execution plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryOptimizer {
    /// Statistics about collections (row counts, index sizes, etc.)
    collection_stats: HashMap<String, CollectionStats>,
    /// Available indexes per collection
    available_indexes: HashMap<String, Vec<IndexInfo>>,
    /// Whether to use parallel execution
    parallel_enabled: bool,
    /// Number of plans evaluated
    plans_evaluated: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionStats {
    pub name: String,
    pub total_rows: usize,
    pub avg_vector_dim: usize,
    pub avg_document_size_bytes: usize,
    pub last_analyzed: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexInfo {
    pub name: String,
    pub index_type: String, // "btree", "hash", "hnsw", "ivf"
    pub field: String,
    pub size_bytes: usize,
    pub selectivity: f64, // 0.0 = very selective, 1.0 = not selective
}

impl QueryOptimizer {
    pub fn new(parallel_enabled: bool) -> Self {
        QueryOptimizer {
            collection_stats: HashMap::new(),
            available_indexes: HashMap::new(),
            parallel_enabled,
            plans_evaluated: 0,
        }
    }

    /// Update collection statistics (called by ANALYZE).
    pub fn update_stats(&mut self, stats: CollectionStats) {
        self.collection_stats
            .insert(stats.name.clone(), stats);
    }

    /// Register an available index.
    pub fn register_index(&mut self, collection: &str, index: IndexInfo) {
        self.available_indexes
            .entry(collection.to_string())
            .or_default()
            .push(index);
    }

    /// Optimize a query operation into the best execution plan.
    pub fn optimize(&mut self, op: &QueryOp) -> ExecutionPlan {
        self.plans_evaluated += 1;
        let id = self.plans_evaluated;

        match op {
            QueryOp::VectorSearch {
                collection,
                k,
                ef_search,
            } => self.plan_vector_search(collection, *k, *ef_search, id),
            QueryOp::ExactSearch {
                collection,
                k,
            } => self.plan_exact_search(collection, *k, id),
            QueryOp::FilteredSearch {
                collection,
                k,
                filter: _,
            } => self.plan_filtered_search(collection, *k, id),
            QueryOp::RangeScan {
                collection,
                field,
                low,
                high,
            } => self.plan_range_scan(collection, field, *low, *high, id),
            QueryOp::FullScan { collection } => self.plan_full_scan(collection, id),
            QueryOp::Aggregate {
                collection,
                operation,
                field,
            } => self.plan_aggregate(collection, operation, field, id),
        }
    }

    fn plan_vector_search(
        &self,
        collection: &str,
        k: usize,
        ef_search: Option<usize>,
        id: u64,
    ) -> ExecutionPlan {
        let stats = self.collection_stats.get(collection);
        let total_rows = stats.map(|s| s.total_rows).unwrap_or(100_000) as f64;

        // Check for ANN indexes
        let has_hnsw = self
            .available_indexes
            .get(collection)
            .map(|indexes| indexes.iter().any(|i| i.index_type == "hnsw"))
            .unwrap_or(false);

        let has_ivf = self
            .available_indexes
            .get(collection)
            .map(|indexes| indexes.iter().any(|i| i.index_type == "ivf"))
            .unwrap_or(false);

        let (steps, cost) = if has_hnsw {
            // HNSW: O(log n) search
            let _ef = ef_search.unwrap_or(k * 2);
            let steps = vec![
                PlanStep::AnnSearch {
                    index_type: "hnsw".into(),
                    k,
                    estimated_cost: total_rows.ln() * 10.0,
                },
                PlanStep::Limit { n: k },
            ];
            let cost = QueryCost::new(
                total_rows.ln() * 10.0,
                1.0,
                (k * 768 * 4) as f64,
                (total_rows.ln() * 0.5) as f64,
            );
            (steps, cost)
        } else if has_ivf {
            // IVF: O(nprobe * (n/list_size))
            let steps = vec![
                PlanStep::AnnSearch {
                    index_type: "ivf".into(),
                    k,
                    estimated_cost: total_rows.sqrt() * 5.0,
                },
                PlanStep::Limit { n: k },
            ];
            let cost = QueryCost::new(
                total_rows.sqrt() * 5.0,
                2.0,
                (k * 768 * 4) as f64,
                total_rows.sqrt() * 0.3,
            );
            (steps, cost)
        } else {
            // Fall back to exact search
            let steps = vec![
                PlanStep::SeqScan {
                    collection: collection.to_string(),
                    estimated_rows: total_rows as usize,
                },
                PlanStep::Sort {
                    field: "similarity".into(),
                    descending: true,
                    estimated_rows: total_rows as usize,
                },
                PlanStep::Limit { n: k },
            ];
            let cost = QueryCost::new(total_rows * 100.0, total_rows, total_rows * 768.0 * 4.0, total_rows * 0.1);
            (steps, cost)
        };

        ExecutionPlan {
            id,
            description: format!("Vector search on '{}' (k={})", collection, k),
            steps,
            estimated_cost: cost,
            parallelizable: true,
        }
    }

    fn plan_exact_search(
        &self,
        collection: &str,
        k: usize,
        id: u64,
    ) -> ExecutionPlan {
        let stats = self.collection_stats.get(collection);
        let total_rows = stats.map(|s| s.total_rows).unwrap_or(100_000) as f64;

        let steps = vec![
            PlanStep::SeqScan {
                collection: collection.to_string(),
                estimated_rows: total_rows as usize,
            },
            PlanStep::Sort {
                field: "similarity".into(),
                descending: true,
                estimated_rows: total_rows as usize,
            },
            PlanStep::Limit { n: k },
        ];

        let cost = QueryCost::new(
            total_rows * 100.0,
            total_rows,
            total_rows * 768.0 * 4.0,
            total_rows * 0.1,
        );

        ExecutionPlan {
            id,
            description: format!("Exact search on '{}' (k={})", collection, k),
            steps,
            estimated_cost: cost,
            parallelizable: true,
        }
    }

    fn plan_filtered_search(
        &self,
        collection: &str,
        k: usize,
        id: u64,
    ) -> ExecutionPlan {
        let stats = self.collection_stats.get(collection);
        let total_rows = stats.map(|s| s.total_rows).unwrap_or(100_000) as f64;

        // Check for B-Tree index on metadata fields
        let has_btree = self
            .available_indexes
            .get(collection)
            .map(|indexes| indexes.iter().any(|i| i.index_type == "btree"))
            .unwrap_or(false);

        let steps = if has_btree {
            vec![
                PlanStep::IndexScan {
                    index_name: format!("{}_metadata_btree", collection),
                    estimated_selectivity: 0.1,
                },
                PlanStep::AnnSearch {
                    index_type: "hnsw".into(),
                    k,
                    estimated_cost: (total_rows * 0.1).ln() * 10.0,
                },
                PlanStep::Limit { n: k },
            ]
        } else {
            vec![
                PlanStep::SeqScan {
                    collection: collection.to_string(),
                    estimated_rows: total_rows as usize,
                },
                PlanStep::Filter {
                    predicate: "metadata_match".into(),
                    estimated_selectivity: 0.1,
                },
                PlanStep::Sort {
                    field: "similarity".into(),
                    descending: true,
                    estimated_rows: (total_rows * 0.1) as usize,
                },
                PlanStep::Limit { n: k },
            ]
        };

        let cost = QueryCost::new(
            total_rows * 20.0,
            total_rows * 0.1,
            (k * 768 * 4) as f64,
            total_rows * 0.05,
        );

        ExecutionPlan {
            id,
            description: format!("Filtered search on '{}' (k={})", collection, k),
            steps,
            estimated_cost: cost,
            parallelizable: true,
        }
    }

    fn plan_range_scan(
        &self,
        collection: &str,
        field: &str,
        low: Option<f64>,
        high: Option<f64>,
        id: u64,
    ) -> ExecutionPlan {
        let stats = self.collection_stats.get(collection);
        let total_rows = stats.map(|s| s.total_rows).unwrap_or(100_000) as f64;

        // Check for B-Tree index on this field
        let has_index = self
            .available_indexes
            .get(collection)
            .map(|indexes| indexes.iter().any(|i| i.field == field))
            .unwrap_or(false);

        let estimated_selectivity = match (low, high) {
            (Some(_), Some(_)) => 0.2,  // bounded range
            (Some(_), None) => 0.5,     // one-sided
            (None, Some(_)) => 0.5,
            (None, None) => 1.0,        // full scan
        };

        let steps = if has_index {
            vec![PlanStep::IndexScan {
                index_name: format!("{}_{}_btree", collection, field),
                estimated_selectivity,
            }]
        } else {
            vec![
                PlanStep::SeqScan {
                    collection: collection.to_string(),
                    estimated_rows: total_rows as usize,
                },
                PlanStep::Filter {
                    predicate: format!("{} IN range", field),
                    estimated_selectivity,
                },
            ]
        };

        let cost = QueryCost::new(
            total_rows * estimated_selectivity * 10.0,
            total_rows * estimated_selectivity,
            total_rows * estimated_selectivity * 100.0,
            total_rows * estimated_selectivity * 0.05,
        );

        ExecutionPlan {
            id,
            description: format!("Range scan on '{}'.{}", collection, field),
            steps,
            estimated_cost: cost,
            parallelizable: true,
        }
    }

    fn plan_full_scan(&self, collection: &str, id: u64) -> ExecutionPlan {
        let stats = self.collection_stats.get(collection);
        let total_rows = stats.map(|s| s.total_rows).unwrap_or(100_000) as f64;

        let steps = vec![PlanStep::SeqScan {
            collection: collection.to_string(),
            estimated_rows: total_rows as usize,
        }];

        let cost = QueryCost::new(
            total_rows * 50.0,
            total_rows,
            total_rows * 768.0 * 4.0,
            total_rows * 0.05,
        );

        ExecutionPlan {
            id,
            description: format!("Full scan of '{}'", collection),
            steps,
            estimated_cost: cost,
            parallelizable: true,
        }
    }

    fn plan_aggregate(
        &self,
        collection: &str,
        operation: &str,
        field: &str,
        id: u64,
    ) -> ExecutionPlan {
        let stats = self.collection_stats.get(collection);
        let total_rows = stats.map(|s| s.total_rows).unwrap_or(100_000) as f64;

        let steps = vec![
            PlanStep::SeqScan {
                collection: collection.to_string(),
                estimated_rows: total_rows as usize,
            },
            PlanStep::Filter {
                predicate: format!("{} IS NOT NULL", field),
                estimated_selectivity: 0.9,
            },
        ];

        let cost = QueryCost::new(
            total_rows * 5.0,
            total_rows * 0.5,
            1024.0,
            total_rows * 0.01,
        );

        ExecutionPlan {
            id,
            description: format!("{} of {}.{}", operation, collection, field),
            steps,
            estimated_cost: cost,
            parallelizable: true,
        }
    }

    /// Generate a parallel execution plan by splitting work across shards/partitions.
    pub fn parallelize(&self, plan: &ExecutionPlan, num_workers: usize) -> ExecutionPlan {
        if !self.parallel_enabled || num_workers <= 1 {
            return plan.clone();
        }

        let parallel_steps: Vec<Vec<PlanStep>> = (0..num_workers)
            .map(|i| {
                let mut worker_steps = plan.steps.clone();
                worker_steps.insert(
                    0,
                    PlanStep::Filter {
                        predicate: format!("hash_mod({}, {})", i, num_workers),
                        estimated_selectivity: 1.0 / num_workers as f64,
                    },
                );
                worker_steps
            })
            .collect();

        let steps = vec![
            PlanStep::Parallel {
                steps: parallel_steps,
                description: format!("Parallel execution across {} workers", num_workers),
            },
            PlanStep::Merge {
                strategy: MergeStrategy::TopK(100),
            },
        ];

        ExecutionPlan {
            id: plan.id,
            description: format!("{} [parallel x{}]", plan.description, num_workers),
            steps,
            estimated_cost: QueryCost::new(
                plan.estimated_cost.cpu_cost / num_workers as f64,
                plan.estimated_cost.io_cost / num_workers as f64,
                plan.estimated_cost.memory_cost,
                plan.estimated_cost.estimated_latency_ms / num_workers as f64,
            ),
            parallelizable: true,
        }
    }

    pub fn plans_evaluated(&self) -> u64 {
        self.plans_evaluated
    }

    pub fn parallel_enabled(&self) -> bool {
        self.parallel_enabled
    }

    pub fn set_parallel_enabled(&mut self, enabled: bool) {
        self.parallel_enabled = enabled;
    }
}

// ─── Parallel Query ─────────────────────────────────────────────────────────

/// A task that can be executed in parallel across workers.
pub struct ParallelTask {
    pub id: usize,
    pub description: String,
    pub work_fn: Arc<dyn Fn() -> ParallelTaskResult + Send + Sync>,
}

/// Result from a parallel task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelTaskResult {
    pub task_id: usize,
    pub success: bool,
    pub rows_processed: usize,
    pub duration_ms: f64,
    pub error: Option<String>,
}

/// Manages parallel query execution across multiple workers.
#[derive(Debug, Clone)]
pub struct ParallelQueryExecutor {
    /// Number of worker threads
    num_workers: usize,
    /// Minimum rows per worker before parallelization is beneficial
    min_rows_per_worker: usize,
    /// Statistics
    total_tasks: u64,
    total_parallel_time_ms: f64,
}

impl ParallelQueryExecutor {
    pub fn new(num_workers: usize, min_rows_per_worker: usize) -> Self {
        ParallelQueryExecutor {
            num_workers,
            min_rows_per_worker,
            total_tasks: 0,
            total_parallel_time_ms: 0.0,
        }
    }

    pub fn num_workers(&self) -> usize {
        self.num_workers
    }

    /// Determine the optimal number of workers for a given workload.
    pub fn optimal_workers(&self, total_rows: usize) -> usize {
        if total_rows < self.min_rows_per_worker {
            1 // Sequential is faster for small workloads
        } else {
            let workers = total_rows / self.min_rows_per_worker;
            workers.min(self.num_workers).max(1)
        }
    }

    /// Execute multiple tasks in parallel using rayon.
    /// Returns results in task order.
    pub fn execute(&mut self, tasks: Vec<ParallelTask>) -> Vec<ParallelTaskResult> {
        let start = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();

        let num_tasks = tasks.len();
        let results: Vec<ParallelTaskResult> = if num_tasks <= 1 {
            // Sequential for single task
            tasks
                .into_iter()
                .map(|t| {
                    let t_start = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs_f64();
                    let result = (t.work_fn)();
                    ParallelTaskResult {
                        duration_ms: (SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs_f64()
                            - t_start)
                            * 1000.0,
                        ..result
                    }
                })
                .collect()
        } else {
            // Parallel using rayon
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(self.num_workers)
                .build()
                .unwrap_or_else(|_| rayon::ThreadPoolBuilder::new().build().unwrap());

            pool.install(|| {
                use rayon::prelude::*;
                tasks
                    .into_par_iter()
                    .map(|t| {
                        let t_start = SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs_f64();
                        let result = (t.work_fn)();
                        ParallelTaskResult {
                            duration_ms: (SystemTime::now()
                                .duration_since(UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_secs_f64()
                                - t_start)
                                * 1000.0,
                            ..result
                        }
                    })
                    .collect()
            })
        };

        let elapsed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64()
            - start;

        self.total_tasks += num_tasks as u64;
        self.total_parallel_time_ms += elapsed * 1000.0;

        results
    }

    /// Execute a partitioned query in parallel.
    /// Each partition is processed by one worker.
    pub fn execute_partitioned<F>(
        &mut self,
        num_partitions: usize,
        work_fn: Arc<F>,
    ) -> Vec<ParallelTaskResult>
    where
        F: Fn(usize) -> ParallelTaskResult + Send + Sync + 'static,
    {
        let tasks: Vec<ParallelTask> = (0..num_partitions)
            .map(|i| {
                let wf = Arc::clone(&work_fn);
                ParallelTask {
                    id: i,
                    description: format!("Partition {}", i),
                    work_fn: Arc::new(move || wf(i)),
                }
            })
            .collect();

        self.execute(tasks)
    }

    pub fn total_tasks(&self) -> u64 {
        self.total_tasks
    }

    pub fn avg_parallel_time_ms(&self) -> f64 {
        if self.total_tasks == 0 {
            0.0
        } else {
            self.total_parallel_time_ms / self.total_tasks as f64
        }
    }
}

// ─── Connection Pool ────────────────────────────────────────────────────────

/// A pooled connection to a database or service.
#[derive(Debug, Clone)]
pub struct PooledConnection {
    pub id: usize,
    pub connection_type: ConnectionType,
    pub created_at: u64,
    pub last_used: u64,
    pub borrow_count: u64,
    pub is_in_use: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Hash, Eq)]
pub enum ConnectionType {
    /// SQLite connection for metadata queries
    Sqlite(String), // path to database
    /// Inference engine connection
    Inference(String), // model name
    /// Tensor database connection
    TensorDb,
    /// External service connection
    External(String),
}

/// Configuration for a connection pool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolConfig {
    pub min_connections: usize,
    pub max_connections: usize,
    pub connection_timeout_ms: u64,
    pub idle_timeout_secs: u64,
    pub max_lifetime_secs: u64,
}

impl Default for PoolConfig {
    fn default() -> Self {
        PoolConfig {
            min_connections: 2,
            max_connections: 10,
            connection_timeout_ms: 5000,
            idle_timeout_secs: 300,
            max_lifetime_secs: 3600,
        }
    }
}

/// Manages a pool of reusable connections.
#[derive(Debug, Clone)]
pub struct ConnectionPool {
    pub name: String,
    pub connection_type: ConnectionType,
    pub config: PoolConfig,
    connections: Vec<PooledConnection>,
    next_id: usize,
    /// Statistics
    total_borrows: u64,
    total_waits: u64,
    total_timeouts: u64,
    total_errors: u64,
}

impl ConnectionPool {
    pub fn new(
        name: impl Into<String>,
        connection_type: ConnectionType,
        config: PoolConfig,
    ) -> Self {
        let min_connections = config.min_connections;
        let mut pool = ConnectionPool {
            name: name.into(),
            connection_type,
            config,
            connections: Vec::with_capacity(min_connections),
            next_id: 1,
            total_borrows: 0,
            total_waits: 0,
            total_timeouts: 0,
            total_errors: 0,
        };

        // Pre-create minimum connections
        for _ in 0..min_connections {
            let conn = pool.create_connection();
            pool.connections.push(conn);
        }

        pool
    }

    fn create_connection(&mut self) -> PooledConnection {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let conn = PooledConnection {
            id: self.next_id,
            connection_type: self.connection_type.clone(),
            created_at: now,
            last_used: now,
            borrow_count: 0,
            is_in_use: false,
        };
        self.next_id += 1;
        conn
    }

    /// Borrow a connection from the pool.
    pub fn borrow(&mut self) -> Result<usize, PoolError> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Clean up expired connections
        self.connections.retain(|c| {
            let age = now - c.created_at;
            let idle = now - c.last_used;
            age < self.config.max_lifetime_secs && (c.is_in_use || idle < self.config.idle_timeout_secs)
        });

        // Find an available connection
        for conn in &mut self.connections {
            if !conn.is_in_use {
                conn.is_in_use = true;
                conn.last_used = now;
                conn.borrow_count += 1;
                self.total_borrows += 1;
                return Ok(conn.id);
            }
        }

        // Create new connection if under max
        if self.connections.len() < self.config.max_connections {
            let mut conn = self.create_connection();
            conn.is_in_use = true;
            conn.last_used = now;
            conn.borrow_count += 1;
            let id = conn.id;
            self.connections.push(conn);
            self.total_borrows += 1;
            return Ok(id);
        }

        // Pool exhausted
        self.total_waits += 1;
        Err(PoolError::PoolExhausted {
            pool_name: self.name.clone(),
            max_connections: self.config.max_connections,
        })
    }

    /// Return a connection to the pool.
    pub fn release(&mut self, id: usize) -> Result<(), PoolError> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        for conn in &mut self.connections {
            if conn.id == id {
                if !conn.is_in_use {
                    return Err(PoolError::NotAcquired { connection_id: id });
                }
                conn.is_in_use = false;
                conn.last_used = now;
                return Ok(());
            }
        }

        Err(PoolError::ConnectionNotFound { connection_id: id })
    }

    /// Get pool statistics.
    pub fn stats(&self) -> PoolStats {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        PoolStats {
            pool_name: self.name.clone(),
            connection_type: format!("{:?}", self.connection_type),
            active_connections: self.connections.iter().filter(|c| c.is_in_use).count(),
            idle_connections: self.connections.iter().filter(|c| !c.is_in_use).count(),
            total_connections: self.connections.len(),
            min_connections: self.config.min_connections,
            max_connections: self.config.max_connections,
            total_borrows: self.total_borrows,
            total_waits: self.total_waits,
            total_timeouts: self.total_timeouts,
            total_errors: self.total_errors,
            oldest_connection_age_secs: self
                .connections
                .iter()
                .map(|c| now - c.created_at)
                .max()
                .unwrap_or(0),
        }
    }

    pub fn len(&self) -> usize {
        self.connections.len()
    }

    pub fn is_empty(&self) -> bool {
        self.connections.is_empty()
    }

    pub fn available(&self) -> usize {
        self.connections.iter().filter(|c| !c.is_in_use).count()
    }
}

/// Errors that can occur during pool operations.
#[derive(Debug, Clone)]
pub enum PoolError {
    PoolExhausted {
        pool_name: String,
        max_connections: usize,
    },
    ConnectionNotFound {
        connection_id: usize,
    },
    NotAcquired {
        connection_id: usize,
    },
    Timeout {
        pool_name: String,
        timeout_ms: u64,
    },
}

impl std::fmt::Display for PoolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PoolError::PoolExhausted { pool_name, max_connections } => {
                write!(
                    f,
                    "Pool '{}' exhausted (max {})",
                    pool_name, max_connections
                )
            }
            PoolError::ConnectionNotFound { connection_id } => {
                write!(f, "Connection {} not found in pool", connection_id)
            }
            PoolError::NotAcquired { connection_id } => {
                write!(f, "Connection {} was not acquired", connection_id)
            }
            PoolError::Timeout { pool_name, timeout_ms } => {
                write!(
                    f,
                    "Pool '{}' timed out after {}ms",
                    pool_name, timeout_ms
                )
            }
        }
    }
}

/// Statistics for a connection pool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolStats {
    pub pool_name: String,
    pub connection_type: String,
    pub active_connections: usize,
    pub idle_connections: usize,
    pub total_connections: usize,
    pub min_connections: usize,
    pub max_connections: usize,
    pub total_borrows: u64,
    pub total_waits: u64,
    pub total_timeouts: u64,
    pub total_errors: u64,
    pub oldest_connection_age_secs: u64,
}

/// Manages multiple connection pools.
#[derive(Debug, Clone)]
pub struct ConnectionPoolManager {
    pools: HashMap<String, ConnectionPool>,
}

impl ConnectionPoolManager {
    pub fn new() -> Self {
        ConnectionPoolManager {
            pools: HashMap::new(),
        }
    }

    /// Register a new connection pool.
    pub fn register_pool(&mut self, pool: ConnectionPool) {
        self.pools.insert(pool.name.clone(), pool);
    }

    /// Get a pool by name.
    pub fn get_pool(&self, name: &str) -> Option<&ConnectionPool> {
        self.pools.get(name)
    }

    /// Get a mutable pool by name.
    pub fn get_pool_mut(&mut self, name: &str) -> Option<&mut ConnectionPool> {
        self.pools.get_mut(name)
    }

    /// Borrow a connection from a named pool.
    pub fn borrow(&mut self, pool_name: &str) -> Result<usize, PoolError> {
        self.pools
            .get_mut(pool_name)
            .ok_or_else(|| PoolError::ConnectionNotFound {
                connection_id: 0,
            })
            .and_then(|pool| pool.borrow())
    }

    /// Release a connection back to its pool.
    pub fn release(&mut self, pool_name: &str, conn_id: usize) -> Result<(), PoolError> {
        self.pools
            .get_mut(pool_name)
            .ok_or_else(|| PoolError::ConnectionNotFound {
                connection_id: conn_id,
            })
            .and_then(|pool| pool.release(conn_id))
    }

    /// Get stats for all pools.
    pub fn all_stats(&self) -> Vec<PoolStats> {
        self.pools.values().map(|p| p.stats()).collect()
    }

    pub fn len(&self) -> usize {
        self.pools.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pools.is_empty()
    }
}

impl Default for ConnectionPoolManager {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Query Execution Manager ────────────────────────────────────────────────

/// Central coordinator for all query execution strategies.
#[derive(Debug, Clone)]
pub struct QueryExecutionManager {
    pub optimizer: QueryOptimizer,
    pub parallel_executor: ParallelQueryExecutor,
    pub pool_manager: ConnectionPoolManager,
}

impl QueryExecutionManager {
    pub fn new(
        optimizer: QueryOptimizer,
        parallel_executor: ParallelQueryExecutor,
        pool_manager: ConnectionPoolManager,
    ) -> Self {
        QueryExecutionManager {
            optimizer,
            parallel_executor,
            pool_manager,
        }
    }

    /// Report comprehensive query execution statistics.
    pub fn report(&self) -> QueryExecutionReport {
        QueryExecutionReport {
            plans_evaluated: self.optimizer.plans_evaluated(),
            parallel_enabled: self.optimizer.parallel_enabled(),
            num_workers: self.parallel_executor.num_workers(),
            total_parallel_tasks: self.parallel_executor.total_tasks(),
            avg_parallel_time_ms: self.parallel_executor.avg_parallel_time_ms(),
            num_connection_pools: self.pool_manager.len(),
            pool_stats: self.pool_manager.all_stats(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryExecutionReport {
    pub plans_evaluated: u64,
    pub parallel_enabled: bool,
    pub num_workers: usize,
    pub total_parallel_tasks: u64,
    pub avg_parallel_time_ms: f64,
    pub num_connection_pools: usize,
    pub pool_stats: Vec<PoolStats>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_optimizer_vector_search() {
        let mut opt = QueryOptimizer::new(true);
        opt.update_stats(CollectionStats {
            name: "vectors".into(),
            total_rows: 1_000_000,
            avg_vector_dim: 768,
            avg_document_size_bytes: 1024,
            last_analyzed: 0,
        });
        opt.register_index(
            "vectors",
            IndexInfo {
                name: "vectors_hnsw".into(),
                index_type: "hnsw".into(),
                field: "embedding".into(),
                size_bytes: 50_000_000,
                selectivity: 0.001,
            },
        );

        let plan = opt.optimize(&QueryOp::VectorSearch {
            collection: "vectors".into(),
            k: 10,
            ef_search: Some(20),
        });

        assert!(plan.estimated_cost.total_score > 0.0);
        assert!(plan.parallelizable);
        // Should prefer HNSW index
        assert!(plan.description.contains("Vector search"));
    }

    #[test]
    fn test_query_optimizer_range_scan() {
        let mut opt = QueryOptimizer::new(true);
        opt.update_stats(CollectionStats {
            name: "docs".into(),
            total_rows: 500_000,
            avg_vector_dim: 0,
            avg_document_size_bytes: 2048,
            last_analyzed: 0,
        });

        let plan = opt.optimize(&QueryOp::RangeScan {
            collection: "docs".into(),
            field: "score".into(),
            low: Some(0.5),
            high: Some(0.9),
        });

        assert!(plan.estimated_cost.total_score > 0.0);
    }

    #[test]
    fn test_parallel_executor_basic() {
        let mut executor = ParallelQueryExecutor::new(4, 100);
        let tasks: Vec<ParallelTask> = (0..4)
            .map(|i| {
                ParallelTask {
                    id: i,
                    description: format!("Task {}", i),
                    work_fn: Arc::new(move || ParallelTaskResult {
                        task_id: i,
                        success: true,
                        rows_processed: 100,
                        duration_ms: 0.0,
                        error: None,
                    }),
                }
            })
            .collect();

        let results = executor.execute(tasks);
        assert_eq!(results.len(), 4);
        assert!(results.iter().all(|r| r.success));
    }

    #[test]
    fn test_connection_pool_borrow_release() {
        let config = PoolConfig {
            min_connections: 2,
            max_connections: 5,
            connection_timeout_ms: 1000,
            idle_timeout_secs: 300,
            max_lifetime_secs: 3600,
        };

        let mut pool = ConnectionPool::new(
            "sqlite_main",
            ConnectionType::Sqlite("/data/main.db".into()),
            config,
        );

        // Borrow a connection
        let id = pool.borrow().unwrap();
        assert!(pool.available() == 1); // 2 min - 1 borrowed = 1 idle

        // Release it
        pool.release(id).unwrap();
        assert!(pool.available() == 2);
    }

    #[test]
    fn test_connection_pool_exhaustion() {
        let config = PoolConfig {
            min_connections: 1,
            max_connections: 2,
            connection_timeout_ms: 1000,
            idle_timeout_secs: 300,
            max_lifetime_secs: 3600,
        };

        let mut pool = ConnectionPool::new(
            "small_pool",
            ConnectionType::Sqlite("/data/test.db".into()),
            config,
        );

        // Borrow both connections
        let _id1 = pool.borrow().unwrap();
        let _id2 = pool.borrow().unwrap();

        // Third borrow should fail
        let result = pool.borrow();
        assert!(result.is_err());
    }

    #[test]
    fn test_query_execution_manager_report() {
        let opt = QueryOptimizer::new(true);
        let executor = ParallelQueryExecutor::new(4, 100);
        let pool_manager = ConnectionPoolManager::new();

        let qem = QueryExecutionManager::new(opt, executor, pool_manager);
        let report = qem.report();
        assert_eq!(report.plans_evaluated, 0);
        assert!(report.parallel_enabled);
        assert_eq!(report.num_connection_pools, 0);
    }
}