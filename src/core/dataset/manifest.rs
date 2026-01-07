use serde::{Deserialize, Serialize};
use std::time::SystemTime;

/// The entrypoint delivery contract for a LINAL dataset package.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetManifest {
    pub name: String,
    pub version: String,
    pub hash: String,
    pub created_at: SystemTime,

    /// Entrypoints into the dataset (e.g., default view name)
    pub entrypoints: HashMap<String, String>,

    /// Supported export formats and their paths relative to the package root
    pub formats: HashMap<String, String>,

    /// Compatibility information (e.g., engine version)
    pub compatibility: HashMap<String, String>,
}

use std::collections::HashMap;

impl DatasetManifest {
    pub fn new(name: String, version: String, hash: String) -> Self {
        Self {
            name,
            version,
            hash,
            created_at: SystemTime::now(),
            entrypoints: HashMap::new(),
            formats: HashMap::new(),
            compatibility: HashMap::new(),
        }
    }
}
