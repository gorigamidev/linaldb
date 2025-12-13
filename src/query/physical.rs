use crate::core::tuple::{Schema, Tuple};
use crate::engine::EngineError;
use crate::engine::TensorDb;
use std::sync::Arc;

/// Trait for physical execution plan nodes
pub trait PhysicalPlan: Send + Sync + std::fmt::Debug {
    /// Get the schema of the output
    fn schema(&self) -> Arc<Schema>;

    /// Execute the plan and return the result rows
    fn execute(&self, db: &TensorDb) -> Result<Vec<Tuple>, EngineError>;
}

/// Sequential Scan Executor
#[derive(Debug)]
pub struct SeqScanExec {
    pub dataset_name: String,
    pub schema: Arc<Schema>,
}

impl PhysicalPlan for SeqScanExec {
    fn schema(&self) -> Arc<Schema> {
        self.schema.clone()
    }

    fn execute(&self, db: &TensorDb) -> Result<Vec<Tuple>, EngineError> {
        let dataset = db.get_dataset(&self.dataset_name)?;
        // Clone all rows (Seq Scan)
        Ok(dataset.rows.clone())
    }
}

/// Filter Executor
pub struct FilterExec {
    pub input: Box<dyn PhysicalPlan>,
    pub predicate: Box<dyn Fn(&Tuple) -> bool + Send + Sync>,
}

impl std::fmt::Debug for FilterExec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FilterExec")
            .field("input", &self.input)
            .field("predicate", &"<closure>")
            .finish()
    }
}

impl PhysicalPlan for FilterExec {
    fn schema(&self) -> Arc<Schema> {
        self.input.schema()
    }

    fn execute(&self, db: &TensorDb) -> Result<Vec<Tuple>, EngineError> {
        let input_rows = self.input.execute(db)?;
        let filtered = input_rows
            .into_iter()
            .filter(|row| (self.predicate)(row))
            .collect();
        Ok(filtered)
    }
}

/// Index Scan Executor (Optimization)
#[derive(Debug)]
pub struct IndexScanExec {
    pub dataset_name: String,
    pub schema: Arc<Schema>,
    pub column: String,
    pub value: crate::core::value::Value,
}

impl PhysicalPlan for IndexScanExec {
    fn schema(&self) -> Arc<Schema> {
        self.schema.clone()
    }

    fn execute(&self, db: &TensorDb) -> Result<Vec<Tuple>, EngineError> {
        let dataset = db.get_dataset(&self.dataset_name)?;

        // Use Index!
        let index = dataset.get_index(&self.column).ok_or_else(|| {
            EngineError::InvalidOp(format!("Index not found on column '{}'", self.column))
        })?;

        let row_ids = index
            .lookup(&self.value)
            .map_err(|e| EngineError::InvalidOp(e))?;

        Ok(dataset.get_rows_by_ids(&row_ids))
    }
}

/// Vector Search Executor
#[derive(Debug)]
pub struct VectorSearchExec {
    pub dataset_name: String,
    pub schema: Arc<Schema>,
    pub column: String,
    pub query: crate::core::tensor::Tensor,
    pub k: usize,
}

impl PhysicalPlan for VectorSearchExec {
    fn schema(&self) -> Arc<Schema> {
        self.schema.clone()
    }

    fn execute(&self, db: &TensorDb) -> Result<Vec<Tuple>, EngineError> {
        let dataset = db.get_dataset(&self.dataset_name)?;
        let index = dataset.get_index(&self.column).ok_or_else(|| {
            EngineError::InvalidOp(format!(
                "Vector index not found on column '{}'",
                self.column
            ))
        })?;

        if index.index_type() != crate::core::index::IndexType::Vector {
            return Err(EngineError::InvalidOp(format!(
                "Index on '{}' is not a VECTOR index",
                self.column
            )));
        }

        let results = index
            .search(&self.query, self.k)
            .map_err(|e| EngineError::InvalidOp(e))?;
        let row_ids: Vec<usize> = results.iter().map(|(id, _)| *id).collect();

        Ok(dataset.get_rows_by_ids(&row_ids))
    }
}

/// Projection Executor
#[derive(Debug)]
pub struct ProjectionExec {
    pub input: Box<dyn PhysicalPlan>,
    pub output_schema: Arc<Schema>,
    pub column_indices: Vec<usize>,
}

impl PhysicalPlan for ProjectionExec {
    fn schema(&self) -> Arc<Schema> {
        self.output_schema.clone()
    }

    fn execute(&self, db: &TensorDb) -> Result<Vec<Tuple>, EngineError> {
        let input_rows = self.input.execute(db)?;
        let mut output_rows = Vec::with_capacity(input_rows.len());

        for row in input_rows {
            let new_values: Vec<_> = self
                .column_indices
                .iter()
                .map(|&idx| row.values[idx].clone())
                .collect();
            output_rows.push(
                Tuple::new(self.output_schema.clone(), new_values)
                    .map_err(|e| EngineError::InvalidOp(e))?,
            );
        }
        Ok(output_rows)
    }
}

/// Limit Executor
#[derive(Debug)]
pub struct LimitExec {
    pub input: Box<dyn PhysicalPlan>,
    pub n: usize,
}

impl PhysicalPlan for LimitExec {
    fn schema(&self) -> Arc<Schema> {
        self.input.schema()
    }

    fn execute(&self, db: &TensorDb) -> Result<Vec<Tuple>, EngineError> {
        let input_rows = self.input.execute(db)?;
        Ok(input_rows.into_iter().take(self.n).collect())
    }
}

/// Sort Executor
#[derive(Debug)]
pub struct SortExec {
    pub input: Box<dyn PhysicalPlan>,
    pub column: String,
    pub ascending: bool,
}

impl PhysicalPlan for SortExec {
    fn schema(&self) -> Arc<Schema> {
        self.input.schema()
    }

    fn execute(&self, db: &TensorDb) -> Result<Vec<Tuple>, EngineError> {
        let rows = self.input.execute(db)?;
        let schema = self.schema();
        let col_idx = schema.get_field_index(&self.column).ok_or_else(|| {
            EngineError::InvalidOp(format!("Column not found for sorting: {}", self.column))
        })?;

        let mut sorted_rows = rows;
        sorted_rows.sort_by(|a, b| {
            let val_a = &a.values[col_idx];
            let val_b = &b.values[col_idx];
            let cmp = val_a.compare(val_b).unwrap_or(std::cmp::Ordering::Equal);
            if self.ascending {
                cmp
            } else {
                cmp.reverse()
            }
        });

        Ok(sorted_rows)
    }
}
