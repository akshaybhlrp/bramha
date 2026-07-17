//! # Bramha Engine
//!
//! Bramha is a database-native intelligence execution system. It provides high-performance
//! LLM inference, retrieval, memory, adaptive learning, and multi-model orchestration.
//!
//! Note: The core inference backend will eventually delegate to the `spanda-engine` crate
//! which provides query-conditional sparse paging and RAM offloading (v7 Architecture).
//!

#![allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::mut_mutex_lock,
    clippy::len_zero,
    clippy::manual_checked_ops,
    clippy::ptr_arg,
    clippy::suspicious_open_options,
    clippy::if_same_then_else,
    clippy::unnecessary_unwrap,
    clippy::collapsible_if,
    clippy::new_without_default,
    clippy::manual_strip,
    clippy::redundant_closure,
    clippy::field_reassign_with_default,
    clippy::explicit_auto_deref,
    clippy::manual_is_multiple_of,
    clippy::map_entry,
    clippy::manual_div_ceil,
    clippy::unwrap_or_default,
    clippy::unnecessary_sort_by,
    clippy::redundant_pattern_matching,
    clippy::needless_borrows_for_generic_args,
    clippy::unnecessary_get_then_check,
    clippy::single_range_in_vec_init,
    clippy::manual_flatten,
    clippy::await_holding_lock,
    clippy::assertions_on_constants,
    clippy::useless_vec,
    clippy::while_let_loop
)]

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
