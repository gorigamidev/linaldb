use crate::core::tensor::TensorId;
use serde::{Deserialize, Serialize};

/// Represents a reference to a data resource.
/// This enables datasets to point to tensors directly or to columns in other datasets.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResourceReference {
    /// A direct reference to a tensor in the TensorStore.
    Tensor { id: TensorId },
    /// A reference to a column in another dataset.
    /// This allows creating views or virtual datasets.
    Column { dataset: String, column: String },
}

impl ResourceReference {
    /// Create a new tensor reference.
    pub fn tensor(id: TensorId) -> Self {
        Self::Tensor { id }
    }

    /// Create a new column reference.
    pub fn column(dataset: impl Into<String>, column: impl Into<String>) -> Self {
        Self::Column {
            dataset: dataset.into(),
            column: column.into(),
        }
    }
}
