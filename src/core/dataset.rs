// src/dataset.rs

use super::tuple::{Schema, Tuple};
use super::value::{Value, ValueType};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::SystemTime;

/// Unique identifier for datasets
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DatasetId(pub u64);

/// Statistics for a single column
#[derive(Debug, Clone)]
pub struct ColumnStats {
    pub value_type: ValueType,
    pub null_count: usize,
    pub min: Option<Value>,
    pub max: Option<Value>,
}

/// Metadata about a dataset
#[derive(Debug, Clone)]
pub struct DatasetMetadata {
    pub name: Option<String>,
    pub created_at: SystemTime,
    pub row_count: usize,
    pub column_stats: HashMap<String, ColumnStats>,
}

impl DatasetMetadata {
    pub fn new(name: Option<String>) -> Self {
        Self {
            name,
            created_at: SystemTime::now(),
            row_count: 0,
            column_stats: HashMap::new(),
        }
    }

    /// Update statistics based on current rows
    pub fn update_stats(&mut self, schema: &Schema, rows: &[Tuple]) {
        self.row_count = rows.len();
        self.column_stats.clear();

        for field in &schema.fields {
            let mut stats = ColumnStats {
                value_type: field.value_type.clone(),
                null_count: 0,
                min: None,
                max: None,
            };

            for row in rows {
                if let Some(value) = row.get(&field.name) {
                    if value.is_null() {
                        stats.null_count += 1;
                    } else {
                        // Update min
                        if let Some(ref current_min) = stats.min {
                            if let Some(ord) = value.compare(current_min) {
                                if ord == std::cmp::Ordering::Less {
                                    stats.min = Some(value.clone());
                                }
                            }
                        } else {
                            stats.min = Some(value.clone());
                        }

                        // Update max
                        if let Some(ref current_max) = stats.max {
                            if let Some(ord) = value.compare(current_max) {
                                if ord == std::cmp::Ordering::Greater {
                                    stats.max = Some(value.clone());
                                }
                            }
                        } else {
                            stats.max = Some(value.clone());
                        }
                    }
                }
            }

            self.column_stats.insert(field.name.clone(), stats);
        }
    }
}

/// Dataset represents a table-like collection of tuples
#[derive(Debug, Clone)]
pub struct Dataset {
    pub id: DatasetId,
    pub schema: Arc<Schema>,
    pub rows: Vec<Tuple>,
    pub metadata: DatasetMetadata,
}

impl Dataset {
    /// Create a new empty dataset
    pub fn new(id: DatasetId, schema: Arc<Schema>, name: Option<String>) -> Self {
        let mut metadata = DatasetMetadata::new(name);
        metadata.update_stats(&schema, &[]);

        Self {
            id,
            schema,
            rows: Vec::new(),
            metadata,
        }
    }

    /// Create a dataset with initial rows
    pub fn with_rows(
        id: DatasetId,
        schema: Arc<Schema>,
        rows: Vec<Tuple>,
        name: Option<String>,
    ) -> Result<Self, String> {
        // Validate all rows match schema
        for (i, row) in rows.iter().enumerate() {
            if !Arc::ptr_eq(&row.schema, &schema) {
                return Err(format!("Row {} has incompatible schema", i));
            }
        }

        let mut metadata = DatasetMetadata::new(name);
        metadata.update_stats(&schema, &rows);

        Ok(Self {
            id,
            schema,
            rows,
            metadata,
        })
    }

    /// Add a row to the dataset
    pub fn add_row(&mut self, row: Tuple) -> Result<(), String> {
        if !Arc::ptr_eq(&row.schema, &self.schema) {
            return Err("Row schema does not match dataset schema".to_string());
        }

        self.rows.push(row);
        self.metadata.update_stats(&self.schema, &self.rows);
        Ok(())
    }

    /// Get number of rows
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Filter rows based on a predicate
    pub fn filter<F>(&self, predicate: F) -> Self
    where
        F: Fn(&Tuple) -> bool,
    {
        let filtered_rows: Vec<Tuple> =
            self.rows.iter().filter(|r| predicate(r)).cloned().collect();

        let mut new_dataset = Self {
            id: self.id,
            schema: self.schema.clone(),
            rows: filtered_rows,
            metadata: self.metadata.clone(),
        };

        new_dataset
            .metadata
            .update_stats(&self.schema, &new_dataset.rows);
        new_dataset
    }

    /// Select specific columns (projection)
    pub fn select(&self, column_names: &[&str]) -> Result<Self, String> {
        // Build new schema with selected fields
        let mut new_fields = Vec::new();
        let mut field_indices = Vec::new();

        for &col_name in column_names {
            let idx = self
                .schema
                .get_field_index(col_name)
                .ok_or_else(|| format!("Column '{}' not found", col_name))?;
            new_fields.push(self.schema.fields[idx].clone());
            field_indices.push(idx);
        }

        let new_schema = Arc::new(Schema::new(new_fields));

        // Project rows
        let mut new_rows = Vec::new();
        for row in &self.rows {
            let new_values: Vec<Value> = field_indices
                .iter()
                .map(|&idx| row.values[idx].clone())
                .collect();

            new_rows.push(Tuple::new(new_schema.clone(), new_values)?);
        }

        let mut new_dataset = Self {
            id: self.id,
            schema: new_schema.clone(),
            rows: new_rows,
            metadata: self.metadata.clone(),
        };

        new_dataset
            .metadata
            .update_stats(&new_schema, &new_dataset.rows);
        Ok(new_dataset)
    }

    /// Take first N rows
    pub fn take(&self, n: usize) -> Self {
        let taken_rows: Vec<Tuple> = self.rows.iter().take(n).cloned().collect();

        let mut new_dataset = Self {
            id: self.id,
            schema: self.schema.clone(),
            rows: taken_rows,
            metadata: self.metadata.clone(),
        };

        new_dataset
            .metadata
            .update_stats(&self.schema, &new_dataset.rows);
        new_dataset
    }

    /// Skip first N rows
    pub fn skip(&self, n: usize) -> Self {
        let skipped_rows: Vec<Tuple> = self.rows.iter().skip(n).cloned().collect();

        let mut new_dataset = Self {
            id: self.id,
            schema: self.schema.clone(),
            rows: skipped_rows,
            metadata: self.metadata.clone(),
        };

        new_dataset
            .metadata
            .update_stats(&self.schema, &new_dataset.rows);
        new_dataset
    }

    /// Sort by a column
    pub fn sort_by(&self, column_name: &str, ascending: bool) -> Result<Self, String> {
        let col_idx = self
            .schema
            .get_field_index(column_name)
            .ok_or_else(|| format!("Column '{}' not found", column_name))?;

        let mut sorted_rows = self.rows.clone();
        sorted_rows.sort_by(|a, b| {
            let val_a = &a.values[col_idx];
            let val_b = &b.values[col_idx];

            let cmp = val_a.compare(val_b).unwrap_or(std::cmp::Ordering::Equal);

            if ascending {
                cmp
            } else {
                cmp.reverse()
            }
        });

        Ok(Self {
            id: self.id,
            schema: self.schema.clone(),
            rows: sorted_rows,
            metadata: self.metadata.clone(),
        })
    }

    /// Map over rows to transform them
    pub fn map<F>(&self, f: F) -> Self
    where
        F: Fn(&Tuple) -> Tuple,
    {
        let mapped_rows: Vec<Tuple> = self.rows.iter().map(f).collect();

        let mut new_dataset = Self {
            id: self.id,
            schema: self.schema.clone(),
            rows: mapped_rows,
            metadata: self.metadata.clone(),
        };

        new_dataset
            .metadata
            .update_stats(&self.schema, &new_dataset.rows);
        new_dataset
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tuple::Field;

    fn create_test_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("id", ValueType::Int),
            Field::new("name", ValueType::String),
            Field::new("age", ValueType::Int),
            Field::new("score", ValueType::Float),
        ]))
    }

    fn create_test_rows(schema: Arc<Schema>) -> Vec<Tuple> {
        vec![
            Tuple::new(
                schema.clone(),
                vec![
                    Value::Int(1),
                    Value::String("Alice".to_string()),
                    Value::Int(30),
                    Value::Float(0.95),
                ],
            )
            .unwrap(),
            Tuple::new(
                schema.clone(),
                vec![
                    Value::Int(2),
                    Value::String("Bob".to_string()),
                    Value::Int(25),
                    Value::Float(0.85),
                ],
            )
            .unwrap(),
            Tuple::new(
                schema.clone(),
                vec![
                    Value::Int(3),
                    Value::String("Carol".to_string()),
                    Value::Int(35),
                    Value::Float(0.90),
                ],
            )
            .unwrap(),
        ]
    }

    #[test]
    fn test_dataset_creation() {
        let schema = create_test_schema();
        let dataset = Dataset::new(DatasetId(1), schema.clone(), Some("test".to_string()));

        assert_eq!(dataset.len(), 0);
        assert_eq!(dataset.metadata.name, Some("test".to_string()));
        assert_eq!(dataset.metadata.row_count, 0);
    }

    #[test]
    fn test_dataset_with_rows() {
        let schema = create_test_schema();
        let rows = create_test_rows(schema.clone());

        let dataset =
            Dataset::with_rows(DatasetId(1), schema, rows, Some("users".to_string())).unwrap();

        assert_eq!(dataset.len(), 3);
        assert_eq!(dataset.metadata.row_count, 3);
    }

    #[test]
    fn test_add_row() {
        let schema = create_test_schema();
        let mut dataset = Dataset::new(DatasetId(1), schema.clone(), None);

        let row = Tuple::new(
            schema.clone(),
            vec![
                Value::Int(1),
                Value::String("Alice".to_string()),
                Value::Int(30),
                Value::Float(0.95),
            ],
        )
        .unwrap();

        assert!(dataset.add_row(row).is_ok());
        assert_eq!(dataset.len(), 1);
    }

    #[test]
    fn test_filter() {
        let schema = create_test_schema();
        let rows = create_test_rows(schema.clone());
        let dataset = Dataset::with_rows(DatasetId(1), schema, rows, None).unwrap();

        // Filter age > 25
        let filtered = dataset.filter(|row| {
            if let Some(Value::Int(age)) = row.get("age") {
                *age > 25
            } else {
                false
            }
        });

        assert_eq!(filtered.len(), 2); // Alice (30) and Carol (35)
    }

    #[test]
    fn test_select() {
        let schema = create_test_schema();
        let rows = create_test_rows(schema.clone());
        let dataset = Dataset::with_rows(DatasetId(1), schema, rows, None).unwrap();

        let selected = dataset.select(&["name", "age"]).unwrap();

        assert_eq!(selected.schema.len(), 2);
        assert_eq!(selected.len(), 3);
        assert!(selected.schema.get_field("name").is_some());
        assert!(selected.schema.get_field("age").is_some());
        assert!(selected.schema.get_field("score").is_none());
    }

    #[test]
    fn test_take_and_skip() {
        let schema = create_test_schema();
        let rows = create_test_rows(schema.clone());
        let dataset = Dataset::with_rows(DatasetId(1), schema, rows, None).unwrap();

        let taken = dataset.take(2);
        assert_eq!(taken.len(), 2);

        let skipped = dataset.skip(1);
        assert_eq!(skipped.len(), 2);
    }

    #[test]
    fn test_sort_by() {
        let schema = create_test_schema();
        let rows = create_test_rows(schema.clone());
        let dataset = Dataset::with_rows(DatasetId(1), schema, rows, None).unwrap();

        // Sort by age ascending
        let sorted_asc = dataset.sort_by("age", true).unwrap();
        if let Some(Value::Int(age)) = sorted_asc.rows[0].get("age") {
            assert_eq!(*age, 25); // Bob is youngest
        }

        // Sort by age descending
        let sorted_desc = dataset.sort_by("age", false).unwrap();
        if let Some(Value::Int(age)) = sorted_desc.rows[0].get("age") {
            assert_eq!(*age, 35); // Carol is oldest
        }
    }

    #[test]
    fn test_metadata_stats() {
        let schema = create_test_schema();
        let rows = create_test_rows(schema.clone());
        let dataset = Dataset::with_rows(DatasetId(1), schema, rows, None).unwrap();

        // Check age stats
        let age_stats = dataset.metadata.column_stats.get("age").unwrap();
        assert_eq!(age_stats.min, Some(Value::Int(25)));
        assert_eq!(age_stats.max, Some(Value::Int(35)));
        assert_eq!(age_stats.null_count, 0);
    }
}
