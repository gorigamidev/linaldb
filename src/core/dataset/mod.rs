use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub mod graph;
pub mod lineage;
pub mod manifest;
pub mod metadata;
pub mod reference;
pub mod registry;
pub mod schema;
pub mod schema_evolution;
pub mod stats;

pub use graph::DatasetGraph;
pub use lineage::{DatasetLineage, LineageNode};
pub use manifest::DatasetManifest;
pub use metadata::{DatasetMetadata, DatasetOrigin};
pub use reference::ResourceReference;
pub use registry::DatasetRegistry;
pub use schema::{ColumnRole, ColumnSchema, DatasetSchema};
pub use schema_evolution::{Migration, SchemaVersion};
pub use stats::{ColumnStats, DatasetStats};

/// Dataset serves as a structured view over existing tensors or other dataset columns.
/// It does not own the actual data, but references it via ResourceReference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dataset {
    pub name: String,
    pub schema: DatasetSchema,
    pub columns: HashMap<String, ResourceReference>,
    pub metadata: Option<DatasetMetadata>,
}

impl Dataset {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            schema: DatasetSchema::default(),
            columns: HashMap::new(),
            metadata: None,
        }
    }

    /// Add an existing resource as a column to this dataset.
    /// This is a zero-copy operation as it only stores the ResourceReference.
    pub fn add_column(&mut self, name: String, reference: ResourceReference, schema: ColumnSchema) {
        self.columns.insert(name, reference);
        self.schema.add_column(schema);
    }

    pub fn get_reference(&self, column_name: &str) -> Option<&ResourceReference> {
        self.columns.get(column_name)
    }
}
