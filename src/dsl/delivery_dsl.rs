use serde::{Deserialize, Serialize};

/// Delivery DSL for read-only projections of datasets.
/// This allows clients to request specific views without executing complex math.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeliveryProjection {
    /// The name of the dataset to project.
    pub dataset: String,

    /// Columns to select.
    pub select: Option<Vec<String>>,

    /// Dimensional filtering (e.g., specific indices for tensors).
    pub shape: Option<HashMap<String, Vec<usize>>>,

    /// Project into a specific format (json, toon, csv).
    pub format: Option<String>,
}

use std::collections::HashMap;

impl DeliveryProjection {
    pub fn new(dataset: String) -> Self {
        Self {
            dataset,
            select: None,
            shape: None,
            format: None,
        }
    }
}

/// Simple parser/executor for the Delivery DSL
pub struct DeliveryEngine;

impl DeliveryEngine {
    /// Compiles a projection against a manifest or schema (Stub for now)
    pub fn execute(&self, _projection: DeliveryProjection) -> String {
        "Projected View (Stub)".to_string()
    }
}
