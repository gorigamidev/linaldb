use crate::core::tensor::Shape;
use crate::core::value::ValueType;
use serde::{Deserialize, Serialize};

/// Semantic role of a column within a dataset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ColumnRole {
    /// Input feature for model or search.
    Feature,
    /// Target variable for training or evaluation.
    Target,
    /// Sample weight.
    Weight,
    /// Unique identifier for each row.
    Guid,
    /// Any other metadata or data.
    Generic,
}

impl Default for ColumnRole {
    fn default() -> Self {
        Self::Generic
    }
}

/// Metadata about a single column in a dataset
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ColumnSchema {
    pub name: String,
    pub value_type: ValueType,
    pub shape: Shape,
    #[serde(default)]
    pub role: ColumnRole,
    #[serde(default)]
    pub nullable: bool,
}

impl ColumnSchema {
    pub fn new(name: String, value_type: ValueType, shape: Shape) -> Self {
        Self {
            name,
            value_type,
            shape,
            role: ColumnRole::Generic,
            nullable: false,
        }
    }

    pub fn with_role(mut self, role: ColumnRole) -> Self {
        self.role = role;
        self
    }

    pub fn with_nullable(mut self, nullable: bool) -> Self {
        self.nullable = nullable;
        self
    }
}

/// Defines the structure of a dataset
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct DatasetSchema {
    pub columns: Vec<ColumnSchema>,
}

impl DatasetSchema {
    pub fn new(columns: Vec<ColumnSchema>) -> Self {
        Self { columns }
    }

    pub fn add_column(&mut self, col: ColumnSchema) {
        self.columns.push(col);
    }

    pub fn get_column(&self, name: &str) -> Option<&ColumnSchema> {
        self.columns.iter().find(|c| c.name == name)
    }
}

impl From<crate::core::tuple::Schema> for DatasetSchema {
    fn from(legacy: crate::core::tuple::Schema) -> Self {
        let columns = legacy
            .fields
            .into_iter()
            .map(|f| {
                ColumnSchema::new(f.name, f.value_type, Shape::new(vec![])) // Default shape for non-tensor columns
                    .with_nullable(f.nullable)
            })
            .collect();
        Self { columns }
    }
}
