use super::schema::DatasetSchema;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Represents a versioned schema for a dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaVersion {
    pub version: u64,
    pub schema: Arc<DatasetSchema>,
    pub migration: Option<Migration>,
}

/// Supported non-breaking schema migrations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Migration {
    /// Add a new column with an optional default value (stored as a string for now).
    AddColumn { name: String, default_value: String },
    /// Rename an existing column.
    RenameColumn { old_name: String, new_name: String },
}

impl Migration {
    /// Apply this migration to a schema.
    pub fn apply(&self, schema: &mut DatasetSchema) -> Result<(), String> {
        match self {
            Migration::AddColumn {
                name,
                default_value: _,
            } => {
                if schema.get_column(name).is_some() {
                    return Err(format!("Column '{}' already exists", name));
                }
                // We don't have enough info here to know the type/shape,
                // but this enum will be used when we have that info.
                // For now, this is just a placeholder for the logic.
            }
            Migration::RenameColumn { old_name, new_name } => {
                if let Some(col) = schema.columns.iter_mut().find(|c| &c.name == old_name) {
                    col.name = new_name.clone();
                } else {
                    return Err(format!("Column '{}' not found", old_name));
                }
            }
        }
        Ok(())
    }
}

impl SchemaVersion {
    pub fn new(version: u64, schema: DatasetSchema) -> Self {
        Self {
            version,
            schema: Arc::new(schema),
            migration: None,
        }
    }

    /// Check if this schema version is compatible with another.
    /// Currently, we only support non-breaking changes.
    pub fn is_compatible_with(&self, other: &SchemaVersion) -> bool {
        // A very basic check: if all columns in 'other' exist in 'self' with the same type.
        for field in &other.schema.columns {
            if let Some(my_field) = self.schema.columns.iter().find(|f| f.name == field.name) {
                if my_field.value_type != field.value_type {
                    return false;
                }
            } else {
                // Column missing in new schema - technically a breaking change if not handled.
                return false;
            }
        }
        true
    }
}
