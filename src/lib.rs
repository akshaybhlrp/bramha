//! # Bramha Engine
//!
//! Bramha is a database-native intelligence execution system. It provides high-performance
//! LLM inference, retrieval, memory, adaptive learning, and multi-model orchestration.
//!
//! Note: The core inference backend will eventually delegate to the `spanda-engine` crate
//! which provides query-conditional sparse paging and RAM offloading (v7 Architecture).
//!

pub mod api;
pub mod cognitive;
pub mod compute;
pub mod concurrency;
pub mod core;
pub mod index;
pub mod inference;
pub mod middleware;
pub mod models;
pub mod planner;
pub mod storage;
