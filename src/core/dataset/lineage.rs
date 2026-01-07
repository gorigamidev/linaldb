use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Represents a node in the dataset derivation DAG.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineageNode {
    pub id: Uuid,
    pub dataset_name: String,
    pub dataset_hash: String,
    pub operation: String,
    pub parents: Vec<Uuid>,
    pub engine_version: String,
}

/// A DAG representing the full derivation and dependency history of a dataset.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DatasetLineage {
    pub nodes: Vec<LineageNode>,
}

impl DatasetLineage {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    pub fn add_node(&mut self, node: LineageNode) {
        self.nodes.push(node);
    }
}
