use crate::core::tensor::Tensor;
use crate::core::value::Value;
use std::fmt::Debug;

/// Types of supported indices
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IndexType {
    /// Exact match index (hash map based)
    Hash,
    /// Vector similarity index (linear scan for MVP, HNSW later)
    Vector,
}

/// Core trait for all index implementations
/// Indices store a mapping from values/vectors to row IDs
pub trait Index: Send + Sync + Debug {
    /// Add a new entry to the index
    fn add(&mut self, row_id: usize, value: &Value) -> Result<(), String>;

    /// Find row IDs that exactly match the given value
    /// Returns empty vector if no match or if index doesn't support exact lookup
    fn lookup(&self, value: &Value) -> Result<Vec<usize>, String>;

    /// Find k nearest neighbors to the query vector
    /// Returns vector of (row_id, score) tuples
    fn search(&self, query: &Tensor, k: usize) -> Result<Vec<(usize, f32)>, String>;

    /// Get the type of this index
    fn index_type(&self) -> IndexType;

    /// Clone the index box
    fn box_clone(&self) -> Box<dyn Index>;
}

impl Clone for Box<dyn Index> {
    fn clone(&self) -> Box<dyn Index> {
        self.box_clone()
    }
}

// Re-export specific implementations
pub mod hash;
pub mod vector;
