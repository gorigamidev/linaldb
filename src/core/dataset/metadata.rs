use super::schema_evolution::SchemaVersion;
use serde::{Deserialize, Serialize};
use std::time::SystemTime;

/// Metadata for a dataset, tracking its identity, origin, and lifecycle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetMetadata {
    /// Dataset name
    pub name: String,

    /// Version number (incremented on each save)
    pub version: u64,

    /// Content hash for identity verification
    pub hash: String,

    /// How this dataset was created
    pub origin: DatasetOrigin,

    /// Current schema version for evolution tracking
    pub schema_version: u64,

    /// History of schema changes
    pub schema_history: Vec<SchemaVersion>,

    /// Creation timestamp
    #[serde(with = "systemtime_serde")]
    pub created_at: SystemTime,

    /// Last update timestamp
    #[serde(with = "systemtime_serde")]
    pub updated_at: SystemTime,

    /// Optional author/creator
    pub author: Option<String>,

    /// Optional description
    pub description: Option<String>,

    /// Tags for categorization
    pub tags: Vec<String>,
}

/// Tracks how a dataset was created
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DatasetOrigin {
    /// Created from scratch via DATASET command
    Created,

    /// Imported from external source
    Imported { source: String },

    /// Derived from another dataset via transformation
    Derived { parent: String, operation: String },

    /// Bound to existing tensor(s)
    Bound { source: String },

    /// Attached from multiple sources
    Attached { sources: Vec<String> },
}

impl DatasetMetadata {
    /// Create new metadata for a dataset
    pub fn new(name: String, origin: DatasetOrigin) -> Self {
        let now = SystemTime::now();
        Self {
            name,
            version: 1,
            hash: String::new(), // Will be computed on save
            origin,
            schema_version: 1,
            schema_history: Vec::new(),
            created_at: now,
            updated_at: now,
            author: None,
            description: None,
            tags: Vec::new(),
        }
    }

    /// Increment version and update timestamp
    pub fn increment_version(&mut self) {
        self.version += 1;
        self.updated_at = SystemTime::now();
    }

    /// Update hash based on content
    pub fn update_hash(&mut self, content_hash: String) {
        self.hash = content_hash;
    }

    /// Set author
    pub fn with_author(mut self, author: String) -> Self {
        self.author = Some(author);
        self
    }

    /// Set description
    pub fn with_description(mut self, description: String) -> Self {
        self.description = Some(description);
        self
    }

    /// Add tag
    pub fn add_tag(&mut self, tag: String) {
        if !self.tags.contains(&tag) {
            self.tags.push(tag);
        }
    }

    /// Record a schema version if it differs from the last one
    pub fn record_schema(&mut self, schema: crate::core::dataset::DatasetSchema) {
        if let Some(last) = self.schema_history.last() {
            if *last.schema == schema {
                return;
            }
        }

        self.schema_version += 1;
        self.schema_history
            .push(SchemaVersion::new(self.schema_version, schema));
    }
}

/// Custom serialization for SystemTime
mod systemtime_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    pub fn serialize<S>(time: &SystemTime, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let duration = time.duration_since(UNIX_EPOCH).unwrap_or(Duration::ZERO);
        duration.as_secs().serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<SystemTime, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs = u64::deserialize(deserializer)?;
        Ok(UNIX_EPOCH + Duration::from_secs(secs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metadata_creation() {
        let meta = DatasetMetadata::new("test_dataset".to_string(), DatasetOrigin::Created);

        assert_eq!(meta.name, "test_dataset");
        assert_eq!(meta.version, 1);
        assert_eq!(meta.schema_version, 1);
    }

    #[test]
    fn test_version_increment() {
        let mut meta = DatasetMetadata::new("test".to_string(), DatasetOrigin::Created);

        let initial_version = meta.version;
        meta.increment_version();

        assert_eq!(meta.version, initial_version + 1);
    }

    #[test]
    fn test_origin_serialization() {
        let origins = vec![
            DatasetOrigin::Created,
            DatasetOrigin::Imported {
                source: "file.csv".to_string(),
            },
            DatasetOrigin::Derived {
                parent: "parent_ds".to_string(),
                operation: "FILTER".to_string(),
            },
        ];

        for origin in origins {
            let json = serde_json::to_string(&origin).unwrap();
            let _: DatasetOrigin = serde_json::from_str(&json).unwrap();
        }
    }
}
