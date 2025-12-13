use crate::core::tensor::Tensor;
use crate::core::tuple::Schema;
use crate::core::value::Value;
use std::sync::Arc;

/// Represents a filter expression
#[derive(Debug, Clone)]
pub enum Expr {
    /// Column reference
    Column(String),
    /// Constants
    Literal(Value),
    /// Binary operation (e.g. =, >, <)
    BinaryExpr {
        left: Box<Expr>,
        op: String,
        right: Box<Expr>,
    },
}

#[derive(Debug, Clone)]
pub enum LogicalPlan {
    /// Scan a dataset
    Scan {
        dataset_name: String,
        schema: Arc<Schema>,
    },
    /// Filter rows
    Filter {
        input: Box<LogicalPlan>,
        predicate: Expr,
    },
    /// Projection (Select columns)
    Project {
        input: Box<LogicalPlan>,
        columns: Vec<String>,
    },
    /// Vector Search (K-NN)
    VectorSearch {
        input: Box<LogicalPlan>,
        column: String,
        query: Tensor,
        k: usize,
    },
    /// Sort rows
    Sort {
        input: Box<LogicalPlan>,
        column: String,
        ascending: bool,
    },
    /// Limit rows
    Limit { input: Box<LogicalPlan>, n: usize },
}

impl LogicalPlan {
    pub fn schema(&self) -> Arc<Schema> {
        match self {
            LogicalPlan::Scan { schema, .. } => schema.clone(),
            LogicalPlan::Filter { input, .. } => input.schema(),
            LogicalPlan::Project { input, columns } => {
                let input_schema = input.schema();
                // Construct new schema from selected columns
                // This is a simplification; normally we'd validate here or during construction
                let fields = columns
                    .iter()
                    .filter_map(|name| input_schema.get_field(name).cloned())
                    .collect();
                Arc::new(Schema::new(fields))
            }
            LogicalPlan::VectorSearch { input, .. } => input.schema(),
            LogicalPlan::Sort { input, .. } => input.schema(),
            LogicalPlan::Limit { input, .. } => input.schema(),
        }
    }
}
