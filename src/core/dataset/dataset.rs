use super::reference::ResourceReference;
use super::schema::DatasetSchema;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Dataset serves as a structured view over existing tensors or other dataset columns.
/// It does not own the actual data, but references it via ResourceReference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dataset {
    pub name: String,
    pub schema: DatasetSchema,
    pub columns: HashMap<String, ResourceReference>,
}

impl Dataset {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            schema: DatasetSchema::default(),
            columns: HashMap::new(),
        }
    }

    /// Add an existing resource as a column to this dataset.
    /// This is a zero-copy operation as it only stores the ResourceReference.
    pub fn add_column(
        &mut self,
        name: String,
        reference: ResourceReference,
        schema: super::schema::ColumnSchema,
    ) {
        self.columns.insert(name, reference);
        self.schema.add_column(schema);
    }

    pub fn get_reference(&self, column_name: &str) -> Option<&ResourceReference> {
        self.columns.get(column_name)
    }
}
