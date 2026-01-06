pub mod dataset;
pub mod graph;
pub mod metadata;
pub mod reference;
pub mod registry;
pub mod schema;
pub mod schema_evolution;

pub use dataset::Dataset;
pub use graph::DatasetGraph;
pub use metadata::{DatasetMetadata, DatasetOrigin};
pub use reference::ResourceReference;
pub use registry::DatasetRegistry;
pub use schema::{ColumnSchema, DatasetSchema};
pub use schema_evolution::{Migration, SchemaVersion};
