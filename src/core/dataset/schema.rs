use crate::core::tensor::Shape;
use crate::core::value::ValueType;
use serde::{Deserialize, Serialize};

/// Metadata about a single column in a dataset
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ColumnSchema {
    pub name: String,
    pub value_type: ValueType,
    pub shape: Shape,
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
