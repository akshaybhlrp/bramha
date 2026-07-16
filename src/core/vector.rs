use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Vector {
    pub id: String,
    pub values: Vec<f32>,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Metric {
    L2,
    Cosine,
    DotProduct,
}

impl Metric {
    pub fn distance(&self, u: &[f32], v: &[f32]) -> f32 {
        match self {
            Metric::L2 => crate::core::distance::l2_distance(u, v),
            Metric::Cosine => crate::core::distance::cosine_similarity(u, v),
            Metric::DotProduct => crate::core::distance::dot_product(u, v),
        }
    }

    /// Helper to sort query results. L2 smaller is better (ascending), Cosine and Dot Product larger is better (descending).
    pub fn is_ascending(&self) -> bool {
        match self {
            Metric::L2 => true,
            Metric::Cosine | Metric::DotProduct => false,
        }
    }
}
