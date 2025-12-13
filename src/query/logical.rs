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
    /// Aggregation function
    AggregateExpr {
        func: AggregateFunction,
        expr: Box<Expr>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum AggregateFunction {
    Sum,
    Avg,
    Count,
    Min,
    Max,
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
    /// Aggregate rows
    Aggregate {
        input: Box<LogicalPlan>,
        group_expr: Vec<Expr>,
        aggr_expr: Vec<Expr>,
    },
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
            LogicalPlan::Aggregate {
                input,
                group_expr,
                aggr_expr,
            } => {
                // Schema consists of Group keys + Aggregation results
                let mut fields = Vec::new();
                // 1. Group keys
                for expr in group_expr {
                    if let Expr::Column(name) = expr {
                        // We assume specific types for now or can lookup in input schema if available.
                        // But we don't have input schema here easily without evaluating expr schema.
                        // For MVP: Aggregate variant usually provides output schema or we infer it.
                        // Let's infer name at least. field type is tricky without full type system.
                        // Hack: for now use placeholder ValueType::String or try to get from input?
                        // Better: We should probably store schema in the LogicalPlan node like Scan does.
                        // But to proceed without major refactor, let's create a minimal schema.
                        // Actually, we can get Name from expr.
                        fields.push(crate::core::tuple::Field::new(
                            name,
                            crate::core::value::ValueType::String,
                        )); // Placeholder type
                    }
                }
                // 2. Aggregates
                for expr in aggr_expr {
                    if let Expr::AggregateExpr { func, expr: inner } = expr {
                        let name = format!("{:?}", func); // e.g., "Count"
                        let mut typ = crate::core::value::ValueType::Int; // Default

                        // Infer for SUM/MIN/MAX if inner is likely Vector (not perfect, but MVP)
                        match func {
                            super::logical::AggregateFunction::Sum
                            | super::logical::AggregateFunction::Min
                            | super::logical::AggregateFunction::Max => {
                                // If inner expr is Column, try to lookup in input schema?
                                // We need access to input schema here!
                                // self.input.schema() is available as `input.schema()`

                                let input_schema = input.schema();
                                typ = infer_expr_type_full(inner.as_ref(), &input_schema);
                            }
                            super::logical::AggregateFunction::Avg => {
                                typ = crate::core::value::ValueType::Float;
                            }
                            _ => {}
                        }

                        fields.push(crate::core::tuple::Field::new(&name, typ));
                    }
                }
                Arc::new(Schema::new(fields))
            }
        }
    }
}

// Helper to fix BinaryExpr destructuring in infer_expr_type
fn infer_expr_type_full(expr: &Expr, schema: &Schema) -> crate::core::value::ValueType {
    use crate::core::value::ValueType;
    match expr {
        Expr::Column(name) => schema
            .get_field(name)
            .map(|f| f.value_type.clone())
            .unwrap_or(ValueType::Null),
        Expr::Literal(val) => val.value_type(),
        Expr::BinaryExpr { left, right, .. } => {
            let l = infer_expr_type_full(left, schema);
            let r = infer_expr_type_full(right, schema);

            match (l, r) {
                (ValueType::Matrix(r, c), _) => ValueType::Matrix(r, c),
                (_, ValueType::Matrix(r, c)) => ValueType::Matrix(r, c),
                (ValueType::Vector(d), _) => ValueType::Vector(d),
                (_, ValueType::Vector(d)) => ValueType::Vector(d),
                (ValueType::Float, _) | (_, ValueType::Float) => ValueType::Float,
                (ValueType::Int, ValueType::Int) => ValueType::Int,
                _ => ValueType::Int,
            }
        }
        Expr::AggregateExpr { .. } => ValueType::Int, // Nested aggregations? Should not happen in logical plan simple exprs
    }
}
