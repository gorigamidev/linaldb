use super::schema::DatasetSchema;
use crate::core::tensor::TensorId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Dataset serves as a structured view over existing tensors in the TensorStore.
/// It does not own the actual data, but references it via TensorId.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dataset {
    pub name: String,
    pub schema: DatasetSchema,
    pub columns: HashMap<String, TensorId>,
}

impl Dataset {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            schema: DatasetSchema::default(),
            columns: HashMap::new(),
        }
    }

    /// Add an existing tensor as a column to this dataset.
    /// This is a zero-copy operation as it only stores the TensorId.
    pub fn add_column(
        &mut self,
        name: String,
        tensor_id: TensorId,
        schema: super::schema::ColumnSchema,
    ) {
        self.columns.insert(name, tensor_id);
        self.schema.add_column(schema);
    }

    pub fn get_tensor_id(&self, column_name: &str) -> Option<TensorId> {
        self.columns.get(column_name).copied()
    }
}
