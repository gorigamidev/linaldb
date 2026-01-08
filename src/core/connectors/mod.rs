pub mod csv_connector;
pub mod hdf5_connector;
pub mod numpy_connector;
pub mod zarr_connector;

use crate::core::dataset::{DatasetLineage, DatasetSchema};
use arrow::record_batch::RecordBatch;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ConnectorError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),

    #[error("Parse error: {0}")]
    Parse(String),

    #[error("Feature not supported: {0}")]
    Unsupported(String),

    #[error("Other error: {0}")]
    Other(String),
}

/// Trait for dataset connectors that can read various scientific and standard formats.
pub trait Connector: Send + Sync {
    /// Unique name of the connector (e.g., "csv", "hdf5")
    fn name(&self) -> &str;

    /// Check if this connector can handle the given path/URI
    fn can_handle(&self, path: &str) -> bool;

    /// Read the dataset from the given path
    fn read_dataset(&self, path: &str) -> Result<(RecordBatch, DatasetLineage), ConnectorError>;

    /// Inspect the dataset to get its schema without reading all data
    fn inspect(&self, path: &str) -> Result<DatasetSchema, ConnectorError>;
}

/// Registry for managing available connectors
pub struct ConnectorRegistry {
    connectors: Vec<Box<dyn Connector>>,
}

impl ConnectorRegistry {
    pub fn new() -> Self {
        Self {
            connectors: Vec::new(),
        }
    }

    pub fn register(&mut self, connector: Box<dyn Connector>) {
        self.connectors.push(connector);
    }

    pub fn find_connector(&self, path: &str) -> Option<&dyn Connector> {
        self.connectors
            .iter()
            .find(|c| c.can_handle(path))
            .map(|c| c.as_ref())
    }

    pub fn list_connectors(&self) -> Vec<&str> {
        self.connectors.iter().map(|c| c.name()).collect()
    }
}

impl Default for ConnectorRegistry {
    fn default() -> Self {
        Self::new()
    }
}
