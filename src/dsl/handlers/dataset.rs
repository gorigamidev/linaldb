use crate::core::tuple::{Field, Schema, Tuple};
use crate::core::value::{Value, ValueType};
use crate::engine::TensorDb;
use std::sync::Arc;

use crate::dsl::{DslError, DslOutput};

/// DATASET name COLUMNS (col1: TYPE1, col2: TYPE2, ...)
/// or
/// DATASET name FROM source ...
pub fn handle_dataset(
    db: &mut TensorDb,
    line: &str,
    line_no: usize,
) -> Result<DslOutput, DslError> {
    if line.contains(" COLUMNS ") {
        handle_dataset_creation(db, line, line_no)
    } else if line.contains(" FROM ") {
        handle_dataset_query(db, line, line_no)
    } else if line.contains(" ADD COLUMN ") {
        handle_add_column(db, line, line_no)
    } else {
        Err(DslError::Parse {
            line: line_no,
            msg: "Expected DATASET ... COLUMNS ... or DATASET ... FROM ... or DATASET ... ADD COLUMN ...".into(),
        })
    }
}

fn handle_dataset_creation(
    db: &mut TensorDb,
    line: &str,
    line_no: usize,
) -> Result<DslOutput, DslError> {
    let rest = line.trim_start_matches("DATASET").trim();

    // Split into name and columns part
    let parts: Vec<&str> = rest.splitn(2, "COLUMNS").collect();
    if parts.len() != 2 {
        return Err(DslError::Parse {
            line: line_no,
            msg: "Expected: DATASET name COLUMNS (col1: TYPE1, col2: TYPE2, ...)".into(),
        });
    }

    let name = parts[0].trim().to_string();
    let columns_str = parts[1].trim();

    // Parse column definitions: (col1: TYPE1, col2: TYPE2, ...)
    let fields = parse_column_definitions(columns_str, line_no)?;
    let schema = Arc::new(Schema::new(fields));

    db.create_dataset(name.clone(), schema)
        .map_err(|e| DslError::Engine {
            line: line_no,
            source: e,
        })?;

    Ok(DslOutput::Message(format!("Created dataset: {}", name)))
}

use crate::query::logical::{Expr, LogicalPlan};
use crate::query::planner::Planner;

/// DATASET target FROM source [FILTER col > val] [SELECT col1, col2] [ORDER BY col [DESC]] [LIMIT n]
fn handle_dataset_query(
    db: &mut TensorDb,
    line: &str,
    line_no: usize,
) -> Result<DslOutput, DslError> {
    let (target_name, current_plan) = build_dataset_query_plan(db, line, line_no)?;

    // Plan & Execute
    let planner = Planner::new(db);
    let physical_plan =
        planner
            .create_physical_plan(&current_plan)
            .map_err(|e| DslError::Engine {
                line: line_no,
                source: e,
            })?;

    let result_rows = physical_plan.execute(db).map_err(|e| DslError::Engine {
        line: line_no,
        source: e,
    })?;
    let result_schema = physical_plan.schema();

    // Create target dataset
    db.create_dataset(target_name.to_string(), result_schema)
        .map_err(|e| DslError::Engine {
            line: line_no,
            source: e,
        })?;

    // Insert rows into target
    let target_ds = db
        .get_dataset_mut(&target_name)
        .map_err(|e| DslError::Engine {
            line: line_no,
            source: e,
        })?;
    target_ds.rows = result_rows;
    // Update metadata/stats
    target_ds
        .metadata
        .update_stats(&target_ds.schema, &target_ds.rows);

    Ok(DslOutput::None)
}

/// SELECT ... FROM ...
pub fn handle_select(db: &mut TensorDb, line: &str, line_no: usize) -> Result<DslOutput, DslError> {
    let working_plan = build_select_query_plan(db, line, line_no)?;

    // Execution
    let planner = Planner::new(db);
    let physical_plan =
        planner
            .create_physical_plan(&working_plan)
            .map_err(|e| DslError::Engine {
                line: line_no,
                source: e,
            })?;
    let result_rows = physical_plan.execute(db).map_err(|e| DslError::Engine {
        line: line_no,
        source: e,
    })?;

    // Construct Dataset for Output
    let result_schema = physical_plan.schema();
    let ds = crate::core::dataset::Dataset {
        id: crate::core::dataset::DatasetId(0),
        schema: result_schema.clone(),
        rows: result_rows.clone(),
        metadata: crate::core::dataset::DatasetMetadata {
            name: Some("Query Result".into()),
            created_at: std::time::SystemTime::now(),
            row_count: result_rows.len(),
            column_stats: std::collections::HashMap::new(),
            schema: result_schema.as_ref().clone(), // Clone the full schema
        },
        indices: std::collections::HashMap::new(),
        lazy_expressions: std::collections::HashMap::new(),
    };

    Ok(DslOutput::Table(ds))
}

pub fn build_select_query_plan(
    db: &mut TensorDb,
    line: &str,
    line_no: usize,
) -> Result<LogicalPlan, DslError> {
    // Parse: SELECT col1, col2, ... FROM source [FILTER ...] [GROUP BY ...]

    // Find FROM
    let from_idx = line.find(" FROM ").ok_or_else(|| DslError::Parse {
        line: line_no,
        msg: "Expected SELECT ... FROM source ...".into(),
    })?;

    // cols part: "SELECT col1, ..."
    let cols_part = line[..from_idx].trim();
    // rest part: "source [FILTER ...]"
    let rest_part = line[from_idx + 6..].trim(); // skip " FROM "

    // Extract source name (first word of rest_part)
    let parts: Vec<&str> = rest_part.splitn(2, ' ').collect();
    let source_name = parts[0];
    let clauses_str = if parts.len() > 1 { parts[1] } else { "" };

    // Build Plan
    let source_ds = db.get_dataset(source_name).map_err(|e| DslError::Engine {
        line: line_no,
        source: e,
    })?;
    let source_schema = source_ds.schema.clone();

    let mut working_plan = LogicalPlan::Scan {
        dataset_name: source_name.to_string(),
        schema: source_schema.clone(),
    };

    let mut pending_group_by: Option<Vec<Expr>> = None;
    let mut remaining_clauses = clauses_str.to_string();
    let keywords = ["FILTER", "WHERE", "ORDER BY", "LIMIT", "GROUP BY", "HAVING"];

    // We process clauses from `clauses_str`
    while !remaining_clauses.is_empty() {
        let clauses_trimmed = remaining_clauses.trim();
        if clauses_trimmed.is_empty() {
            break;
        }

        if clauses_trimmed.starts_with("FILTER ") || clauses_trimmed.starts_with("WHERE ") {
            let kw = if clauses_trimmed.starts_with("WHERE ") {
                "WHERE"
            } else {
                "FILTER"
            };
            let (cond_str, rem) = split_clause(clauses_trimmed, kw, &keywords);
            let cond_string = cond_str.to_string();
            remaining_clauses = rem.to_string();
            let (col, op, val) = parse_filter_condition(&cond_string, line_no)?;
            working_plan = LogicalPlan::Filter {
                input: Box::new(working_plan),
                predicate: Expr::BinaryExpr {
                    left: Box::new(Expr::Column(col)),
                    op,
                    right: Box::new(Expr::Literal(val)),
                },
            };
        } else if clauses_trimmed.starts_with("GROUP BY ") {
            let (group_str, rem) = split_clause(clauses_trimmed, "GROUP BY", &keywords);
            let group_string = group_str.to_string();
            remaining_clauses = rem.to_string();
            let cols: Vec<String> = group_string
                .split(',')
                .map(|s| s.trim().to_string())
                .collect();
            let exprs: Vec<Expr> = cols.into_iter().map(Expr::Column).collect();
            pending_group_by = Some(exprs);
        } else if clauses_trimmed.starts_with("HAVING ") {
            let (cond_str, rem) = split_clause(clauses_trimmed, "HAVING", &keywords);
            let cond_string = cond_str.to_string();
            remaining_clauses = rem.to_string();
            let (col, op, val) = parse_filter_condition(&cond_string, line_no)?;

            working_plan = LogicalPlan::Filter {
                input: Box::new(working_plan),
                predicate: Expr::BinaryExpr {
                    left: Box::new(Expr::Column(col)),
                    op,
                    right: Box::new(Expr::Literal(val)),
                },
            };
        } else if clauses_trimmed.starts_with("limit ") || clauses_trimmed.starts_with("LIMIT ") {
            let (limit_str, rem) = split_clause(clauses_trimmed, "LIMIT", &keywords);
            let limit_string = limit_str.to_string();
            remaining_clauses = rem.to_string();
            let n: usize = limit_string.parse().map_err(|_| DslError::Parse {
                line: line_no,
                msg: "Invalid limit".into(),
            })?;
            working_plan = LogicalPlan::Limit {
                input: Box::new(working_plan),
                n,
            };
        } else {
            if clauses_trimmed.starts_with("ORDER BY ") {
                let (order_str, rem) = split_clause(clauses_trimmed, "ORDER BY", &keywords);
                let order_string = order_str.to_string();
                remaining_clauses = rem.to_string();
                let parts: Vec<&str> = order_string.split_whitespace().collect();
                let col = parts[0].to_string();
                let desc = parts.len() > 1 && parts[1].eq_ignore_ascii_case("DESC");
                working_plan = LogicalPlan::Sort {
                    input: Box::new(working_plan),
                    column: col,
                    ascending: !desc,
                };
            } else {
                return Err(DslError::Parse {
                    line: line_no,
                    msg: format!("Unknown clause in SELECT: {}", clauses_trimmed),
                });
            }
        }
    }

    // Finally apply Projection/Aggregation from the initial SELECT `cols_part`
    let select_exprs_str = cols_part.trim_start_matches("SELECT ").trim();
    let exprs = parse_select_items(select_exprs_str, line_no)?;

    // Check for Aggregates
    let has_aggr = exprs
        .iter()
        .any(|e| matches!(e, Expr::AggregateExpr { .. }));

    if pending_group_by.is_some() || has_aggr {
        let group_expr = pending_group_by.unwrap_or_default();
        let actual_aggs: Vec<Expr> = exprs
            .into_iter()
            .filter(|e| matches!(e, Expr::AggregateExpr { .. }))
            .collect();

        working_plan = LogicalPlan::Aggregate {
            input: Box::new(working_plan),
            group_expr,
            aggr_expr: actual_aggs,
        };
    } else {
        // Simple Projection with Wildcard Expansion support
        let mut cols = Vec::new();
        for e in &exprs {
            if let Expr::Column(c) = e {
                if c == "*" {
                    // Expand wildcard
                    for field in &source_schema.fields {
                        cols.push(field.name.clone());
                    }
                } else {
                    cols.push(c.clone());
                }
            } else {
                return Err(DslError::Parse {
                    line: line_no,
                    msg: "Only columns or Aggregates supported".into(),
                });
            }
        }

        working_plan = LogicalPlan::Project {
            input: Box::new(working_plan),
            columns: cols,
        };
    }

    Ok(working_plan)
}

pub fn build_dataset_query_plan(
    db: &mut TensorDb,
    line: &str,
    line_no: usize,
) -> Result<(String, LogicalPlan), DslError> {
    let rest = line.trim_start_matches("DATASET").trim();

    // Split into target and FROM source...
    let parts: Vec<&str> = rest.splitn(2, " FROM ").collect();
    if parts.len() != 2 {
        return Err(DslError::Parse {
            line: line_no,
            msg: "Expected: DATASET target FROM source ...".into(),
        });
    }

    let target_name = parts[0].trim().to_string();
    let query_part = parts[1].trim();

    let keywords = [
        "FILTER", "SELECT", "ORDER BY", "LIMIT", "GROUP BY", "HAVING",
    ];
    let mut first_keyword_idx = None;

    for &kw in &keywords {
        if let Some(idx) = query_part.find(kw) {
            // Ensure matches whole word
            if idx > 0 && !query_part[idx - 1..].starts_with(' ') {
                continue; // part of another word
            }
            if first_keyword_idx.map_or(true, |curr| idx < curr) {
                first_keyword_idx = Some(idx);
            }
        }
    }

    let (source_name, mut clauses_str) = if let Some(idx) = first_keyword_idx {
        (query_part[..idx].trim(), &query_part[idx..])
    } else {
        (query_part.trim(), "")
    };

    // Get source dataset schema for validation
    let source_ds = db.get_dataset(source_name).map_err(|e| DslError::Engine {
        line: line_no,
        source: e,
    })?;
    let source_schema = source_ds.schema.clone();

    // Initial Plan: Scan
    let mut current_plan = LogicalPlan::Scan {
        dataset_name: source_name.to_string(),
        schema: source_schema.clone(),
    };

    // Process clauses
    let mut pending_group_by: Option<Vec<Expr>> = None;
    while !clauses_str.is_empty() {
        let clauses_trimmed = clauses_str.trim();

        if clauses_trimmed.starts_with("FILTER ") {
            let (cond_str, remaining) = split_clause(clauses_trimmed, "FILTER", &keywords);
            clauses_str = remaining;

            // Parse condition: col > val
            let (col, op, val) = parse_filter_condition(cond_str, line_no)?;

            current_plan = LogicalPlan::Filter {
                input: Box::new(current_plan),
                predicate: Expr::BinaryExpr {
                    left: Box::new(Expr::Column(col)),
                    op,
                    right: Box::new(Expr::Literal(val)),
                },
            };
        } else if clauses_trimmed.starts_with("GROUP BY ") {
            let (group_str, remaining) = split_clause(clauses_trimmed, "GROUP BY", &keywords);
            clauses_str = remaining;

            let cols: Vec<String> = group_str.split(',').map(|s| s.trim().to_string()).collect();
            let exprs: Vec<Expr> = cols.into_iter().map(Expr::Column).collect();
            pending_group_by = Some(exprs);
        } else if clauses_trimmed.starts_with("SELECT ") {
            let (cols_str, remaining) = split_clause(clauses_trimmed, "SELECT", &keywords);
            clauses_str = remaining;

            // New parse function for expressions
            let exprs = parse_select_items(cols_str, line_no)?;

            // Check if we need Aggregate or Project
            let has_aggr = exprs
                .iter()
                .any(|e| matches!(e, Expr::AggregateExpr { .. }));

            if pending_group_by.is_some() || has_aggr {
                // Must be Aggregate
                let group_expr = pending_group_by.take().unwrap_or_default();

                // Filter aggr_expr to strictly include AggregateExprs
                // Non-aggregates (Columns) are assumed to be Group Keys or ignored for now.
                // This ensures Schema (Keys + Aggs) matches Execution (Keys + Accs).
                let actual_aggs: Vec<Expr> = exprs
                    .into_iter()
                    .filter(|e| matches!(e, Expr::AggregateExpr { .. }))
                    .collect();

                // If it's a global aggregation (no group by), group_expr is empty.
                // We construct Aggregate plan.
                current_plan = LogicalPlan::Aggregate {
                    input: Box::new(current_plan),
                    group_expr,
                    aggr_expr: actual_aggs,
                };
            } else {
                // Simple Projection (backward compat)
                // Convert Expr::Column back to String
                let cols: Vec<String> = exprs
                    .iter()
                    .map(|e| {
                        if let Expr::Column(c) = e {
                            Ok(c.clone())
                        } else {
                            // Projecting literals or unsupported exprs in Project?
                            // Current LogicalPlan::Project only supports Columns.
                            // If we have literal, we can't map to Project yet.
                            // But parse_select_items only parses Col or AggFunc(Col).
                            // So it should be fine.
                            Err(DslError::Parse {
                                line: line_no,
                                msg: "Only columns supported in simple SELECT (Project)".into(),
                            })
                        }
                    })
                    .collect::<Result<_, _>>()?;

                current_plan = LogicalPlan::Project {
                    input: Box::new(current_plan),
                    columns: cols,
                };
            }
        } else if clauses_trimmed.starts_with("HAVING ") {
            // HAVING comes after aggregation
            let (cond_str, remaining) = split_clause(clauses_trimmed, "HAVING", &keywords);
            clauses_str = remaining;

            // Parse condition like filter
            // But strictly it should match an output of Aggregation.
            // For simplicity, reuse parse_filter_condition and wrap in Filter
            // Because HAVING is just a Filter on the output of Aggregate.
            let (col, op, val) = parse_filter_condition(cond_str, line_no)?;

            current_plan = LogicalPlan::Filter {
                input: Box::new(current_plan),
                predicate: Expr::BinaryExpr {
                    left: Box::new(Expr::Column(col)),
                    op,
                    right: Box::new(Expr::Literal(val)),
                },
            };
        } else if clauses_trimmed.starts_with("ORDER BY ") {
            let (order_str, remaining) = split_clause(clauses_trimmed, "ORDER BY", &keywords);
            clauses_str = remaining;

            let parts: Vec<&str> = order_str.split_whitespace().collect();
            if parts.is_empty() {
                return Err(DslError::Parse {
                    line: line_no,
                    msg: "Empty ORDER BY clause".into(),
                });
            }
            let col_name = parts[0].to_string();
            let ascending = if parts.len() > 1 && parts[1] == "DESC" {
                false
            } else {
                true
            };

            current_plan = LogicalPlan::Sort {
                input: Box::new(current_plan),
                column: col_name,
                ascending,
            };
        } else if clauses_trimmed.starts_with("LIMIT ") {
            let (limit_str, remaining) = split_clause(clauses_trimmed, "LIMIT", &keywords);
            clauses_str = remaining;

            let n: usize = limit_str.trim().parse().map_err(|_| DslError::Parse {
                line: line_no,
                msg: format!("Invalid LIMIT: {}", limit_str),
            })?;

            current_plan = LogicalPlan::Limit {
                input: Box::new(current_plan),
                n,
            };
        } else {
            return Err(DslError::Parse {
                line: line_no,
                msg: format!("Unexpected clause: {}", clauses_str),
            });
        }
    }

    Ok((target_name, current_plan))
}

fn split_clause<'a>(s: &'a str, current_kw: &str, all_kws: &[&str]) -> (&'a str, &'a str) {
    let content_start = current_kw.len();
    let remaining_s = &s[content_start..];

    // Find next keyword
    let mut next_kw_idx = None;
    for &kw in all_kws {
        if let Some(idx) = remaining_s.find(kw) {
            // ensure word boundary roughly (space before)
            if idx > 0 && remaining_s.as_bytes()[idx - 1] == b' ' {
                if next_kw_idx.map_or(true, |curr| idx < curr) {
                    next_kw_idx = Some(idx);
                }
            }
        }
    }

    if let Some(idx) = next_kw_idx {
        (&remaining_s[..idx].trim(), &remaining_s[idx..])
    } else {
        (remaining_s.trim(), "")
    }
}

fn parse_filter_condition(s: &str, line_no: usize) -> Result<(String, String, Value), DslError> {
    // col > val
    // Split by operators: >=, <=, >, <, =, !=
    // Order matters (longest first)
    let ops = [">=", "<=", "!=", "=", ">", "<"];

    for op in ops {
        if let Some(idx) = s.find(op) {
            let col = s[..idx].trim().to_string();
            let val_str = s[idx + op.len()..].trim();
            // Parse value (try float, int, string - naive inference or use context?)
            // parse_single_value assumes generic.
            let val = parse_single_value(val_str, line_no)?;
            return Ok((col, op.to_string(), val));
        }
    }

    Err(DslError::Parse {
        line: line_no,
        msg: format!("Invalid filter condition: {}", s),
    })
}

// ... existing code ...

/// Parse column definitions from: (col1: TYPE1, col2: TYPE2, ...)
fn parse_column_definitions(columns_str: &str, line_no: usize) -> Result<Vec<Field>, DslError> {
    // Remove only outer parentheses
    let columns_str = columns_str.trim();
    let inner = if columns_str.starts_with('(') && columns_str.ends_with(')') {
        &columns_str[1..columns_str.len() - 1]
    } else {
        columns_str
    };
    let inner = inner.trim();

    if inner.is_empty() {
        return Err(DslError::Parse {
            line: line_no,
            msg: "Empty column definition".into(),
        });
    }

    // Split into comma arguments
    // Ensure we stripped outer parens if they exist
    let columns_str = columns_str.trim();
    let inner = if columns_str.starts_with('(') && columns_str.ends_with(')') {
        &columns_str[1..columns_str.len() - 1]
    } else {
        columns_str
    };

    println!("DEBUG: columns_str='{}'", columns_str);
    println!("DEBUG: inner='{}'", inner);

    let mut fields = Vec::new();

    // Split by comma, respecting parentheses for types like Matrix(R, C)
    let parts = split_args(inner);
    for col_def in parts {
        let col_def = col_def.trim();

        // Split by colon: name: TYPE
        let parts: Vec<&str> = col_def.splitn(2, ':').collect();
        if parts.len() != 2 {
            return Err(DslError::Parse {
                line: line_no,
                msg: format!("Invalid column definition: {}", col_def),
            });
        }

        let col_name = parts[0].trim();
        let type_str = parts[1].trim();

        let value_type = parse_value_type(type_str, line_no)?;
        fields.push(Field::new(col_name, value_type));
    }

    Ok(fields)
}

/// Parse a value type from string
fn split_args(s: &str) -> Vec<String> {
    let mut args = Vec::new();
    let mut current = String::new();
    let mut depth = 0;

    for ch in s.chars() {
        match ch {
            '(' | '[' => {
                depth += 1;
                current.push(ch);
            }
            ')' | ']' => {
                depth -= 1;
                current.push(ch);
            }
            ',' if depth == 0 => {
                args.push(current.trim().to_string());
                current.clear();
            }
            _ => current.push(ch),
        }
    }
    if !current.trim().is_empty() {
        args.push(current.trim().to_string());
    }
    args
}

fn parse_value_type(type_str: &str, line_no: usize) -> Result<ValueType, DslError> {
    let upper = type_str.to_uppercase();
    if upper == "INT" {
        Ok(ValueType::Int)
    } else if upper == "FLOAT" {
        Ok(ValueType::Float)
    } else if upper == "STRING" {
        Ok(ValueType::String)
    } else if upper == "BOOL" {
        Ok(ValueType::Bool)
    } else if upper.starts_with("VECTOR") {
        // Expected format: VECTOR(N)
        let start = upper.find('(');
        let end = upper.find(')');
        if let (Some(s), Some(e)) = (start, end) {
            let dim_str = &upper[s + 1..e];
            let dim: usize = dim_str.parse().map_err(|_| DslError::Parse {
                line: line_no,
                msg: format!("Invalid dimension in Vector definition: {}", dim_str),
            })?;
            Ok(ValueType::Vector(dim))
        } else {
            Err(DslError::Parse {
                line: line_no,
                msg: format!(
                    "Invalid Vector definition: {}. Expected VECTOR(N)",
                    type_str
                ),
            })
        }
    } else if upper.starts_with("MATRIX") {
        // Expected format: MATRIX(R, C)
        let start = upper.find('(');
        let end = upper.find(')');
        if let (Some(s), Some(e)) = (start, end) {
            let dims_str = &upper[s + 1..e];
            let parts: Vec<&str> = dims_str.split(',').collect();
            if parts.len() != 2 {
                return Err(DslError::Parse {
                    line: line_no,
                    msg: format!(
                        "Invalid Matrix definition: {}. Expected MATRIX(R, C)",
                        type_str
                    ),
                });
            }
            let r: usize = parts[0].trim().parse().map_err(|_| DslError::Parse {
                line: line_no,
                msg: "Invalid rows".into(),
            })?;
            let c: usize = parts[1].trim().parse().map_err(|_| DslError::Parse {
                line: line_no,
                msg: "Invalid cols".into(),
            })?;
            Ok(ValueType::Matrix(r, c))
        } else {
            Err(DslError::Parse {
                line: line_no,
                msg: format!(
                    "Invalid Matrix definition: {}. Expected MATRIX(R, C)",
                    type_str
                ),
            })
        }
    } else {
        Err(DslError::Parse {
            line: line_no,
            msg: format!("Unknown type: {}", type_str),
        })
    }
}

pub fn parse_single_value(s: &str, line_no: usize) -> Result<Value, DslError> {
    let s = s.trim();

    // String (quoted)
    if s.starts_with('"') && s.ends_with('"') {
        let content = &s[1..s.len() - 1];
        return Ok(Value::String(content.to_string()));
    }

    // Boolean
    if s == "true" {
        return Ok(Value::Bool(true));
    }
    if s == "false" {
        return Ok(Value::Bool(false));
    }

    // Float (has decimal point)
    if s.contains('.') && !s.starts_with('[') {
        return s
            .parse::<f32>()
            .map(Value::Float)
            .map_err(|_| DslError::Parse {
                line: line_no,
                msg: format!("Invalid float: {}", s),
            });
    }

    // Vector [val1, val2, ...] OR Matrix [[...], [...]]
    if s.starts_with('[') && s.ends_with(']') {
        let content = &s[1..s.len() - 1];
        let parts = split_args(content);

        // Detect Matrix: if first element is array?
        if !parts.is_empty() && parts[0].starts_with('[') {
            // Matrix
            let mut matrix = Vec::new();
            for p in parts {
                if let Value::Vector(v) = parse_single_value(&p, line_no)? {
                    matrix.push(v);
                } else {
                    return Err(DslError::Parse {
                        line: line_no,
                        msg: format!("Matrix elements must verify to vectors. Got: {}", p),
                    });
                }
            }
            return Ok(Value::Matrix(matrix));
        }

        let mut floats = Vec::with_capacity(parts.len());
        for p in parts {
            if p.is_empty() {
                continue;
            }
            let f = p.parse::<f32>().map_err(|_| DslError::Parse {
                line: line_no,
                msg: format!("Invalid vector element: {}", p),
            })?;
            floats.push(f);
        }
        return Ok(Value::Vector(floats));
    }

    // Int
    s.parse::<i64>()
        .map(Value::Int)
        .map_err(|_| DslError::Parse {
            line: line_no,
            msg: format!("Invalid value: {}", s),
        })
}

/// INSERT INTO dataset_name VALUES (val1, val2, ...)
pub fn handle_insert(db: &mut TensorDb, line: &str, line_no: usize) -> Result<DslOutput, DslError> {
    let rest = line.trim_start_matches("INSERT INTO").trim();

    // Split into dataset_name and values part
    let parts: Vec<&str> = rest.splitn(2, "VALUES").collect();
    if parts.len() != 2 {
        return Err(DslError::Parse {
            line: line_no,
            msg: "Expected: INSERT INTO dataset_name VALUES (val1, val2, ...)".into(),
        });
    }

    let dataset_name = parts[0].trim();
    let values_str = parts[1].trim();

    // Get dataset to know schema
    let dataset = db.get_dataset(dataset_name).map_err(|e| DslError::Engine {
        line: line_no,
        source: e,
    })?;
    let schema = dataset.schema.clone();

    // Parse values
    let values = parse_tuple_values(values_str, &schema, line_no)?;
    let tuple = Tuple::new(schema.clone(), values).map_err(|e| DslError::Parse {
        line: line_no,
        msg: e,
    })?;

    db.insert_row(dataset_name, tuple)
        .map_err(|e| DslError::Engine {
            line: line_no,
            source: e,
        })?;

    Ok(DslOutput::None)
}

/// Parse tuple values from: (val1, val2, ...)
fn parse_tuple_values(
    values_str: &str,
    schema: &Schema,
    line_no: usize,
) -> Result<Vec<Value>, DslError> {
    // Remove parentheses
    let inner = values_str
        .trim()
        .trim_start_matches('(')
        .trim_end_matches(')')
        .trim();

    if inner.is_empty() {
        return Err(DslError::Parse {
            line: line_no,
            msg: "Empty values".into(),
        });
    }

    let mut values = Vec::new();
    let mut current = String::new();
    let mut in_string = false;
    let mut depth = 0;

    // Parse values, handling strings and nested structures
    for ch in inner.chars() {
        match ch {
            '"' => {
                in_string = !in_string;
                current.push(ch);
            }
            '[' | '(' if !in_string => {
                depth += 1;
                current.push(ch);
            }
            ']' | ')' if !in_string => {
                depth -= 1;
                current.push(ch);
            }
            ',' if !in_string && depth == 0 => {
                values.push(parse_single_value(&current.trim(), line_no)?);
                current.clear();
            }
            _ => {
                current.push(ch);
            }
        }
    }

    // Don't forget the last value
    if !current.trim().is_empty() {
        values.push(parse_single_value(&current.trim(), line_no)?);
    }

    // Validate count matches schema
    if values.len() != schema.len() {
        return Err(DslError::Parse {
            line: line_no,
            msg: format!("Expected {} values, got {}", schema.len(), values.len()),
        });
    }

    Ok(values)
}

/// Handle DATASET <name> ADD COLUMN <col>: <type> [DEFAULT <val>]
/// or
/// Handle DATASET <name> ADD COLUMN <col> = <expression> (computed column)
fn handle_add_column(db: &mut TensorDb, line: &str, line_no: usize) -> Result<DslOutput, DslError> {
    let rest = line.trim_start_matches("DATASET").trim();

    // Split into dataset name and ADD COLUMN part
    let parts: Vec<&str> = rest.splitn(2, " ADD COLUMN ").collect();
    if parts.len() != 2 {
        return Err(DslError::Parse {
            line: line_no,
            msg: "Expected: DATASET <name> ADD COLUMN <col>: <type> [DEFAULT <val>] or DATASET <name> ADD COLUMN <col> = <expression>".into(),
        });
    }

    let dataset_name = parts[0].trim();
    let column_spec = parts[1].trim();

    // Check if it's a computed column (has =) or regular column (has :)
    if column_spec.contains('=') && !column_spec.contains(':') {
        // Computed column: <col> = <expression> [LAZY]
        let eq_idx = column_spec.find('=').ok_or_else(|| DslError::Parse {
            line: line_no,
            msg: "Invalid computed column syntax".into(),
        })?;
        
        // Check for LAZY keyword
        let is_lazy = column_spec.to_uppercase().contains("LAZY");
        let expression_part = if is_lazy {
            // Remove LAZY keyword from expression part
            let upper = column_spec.to_uppercase();
            let lazy_pos = upper.find("LAZY").unwrap();
            column_spec[eq_idx + 1..lazy_pos].trim()
        } else {
            column_spec[eq_idx + 1..].trim()
        };
        
        let column_name = column_spec[..eq_idx].trim().to_string();
        
        if column_name.is_empty() {
            return Err(DslError::Parse {
                line: line_no,
                msg: "Column name cannot be empty".into(),
            });
        }
        
        // Parse the expression
        let expr = parse_expression(expression_part, line_no)?;
        
        // Get dataset
        let dataset = db.get_dataset(dataset_name).map_err(|e| DslError::Engine {
            line: line_no,
            source: e,
        })?;
        
        if is_lazy {
            // For lazy columns, we only need to infer the type from one row
            let value_type = if dataset.rows.is_empty() {
                return Err(DslError::Parse {
                    line: line_no,
                    msg: "Cannot infer type from empty dataset for lazy column".into(),
                });
            } else {
                use crate::query::physical::evaluate_expression;
                let val = evaluate_expression(&expr, &dataset.rows[0]);
                val.value_type()
            };
            
            // Add lazy column (no pre-computed values needed)
            db.alter_dataset_add_computed_column(
                dataset_name,
                column_name.clone(),
                value_type,
                vec![], // Empty for lazy columns
                expr,
                true, // lazy = true
            )
            .map_err(|e| DslError::Engine {
                line: line_no,
                source: e,
            })?;
            
            Ok(DslOutput::Message(format!(
                "Added lazy computed column '{}' to dataset '{}'",
                column_name, dataset_name
            )))
        } else {
            // Materialized: evaluate expression for each row
            use crate::query::physical::evaluate_expression;
            let mut computed_values = Vec::new();
            let mut inferred_type: Option<crate::core::value::ValueType> = None;
            
            for row in &dataset.rows {
                let val = evaluate_expression(&expr, row);
                if inferred_type.is_none() {
                    inferred_type = Some(val.value_type());
                }
                computed_values.push(val);
            }
            
            let value_type = inferred_type.ok_or_else(|| DslError::Parse {
                line: line_no,
                msg: "Cannot infer type from empty dataset".into(),
            })?;
            
            // Add column with computed values
            db.alter_dataset_add_computed_column(
                dataset_name,
                column_name.clone(),
                value_type,
                computed_values,
                expr,
                false, // lazy = false
            )
            .map_err(|e| DslError::Engine {
                line: line_no,
                source: e,
            })?;
            
            Ok(DslOutput::Message(format!(
                "Added computed column '{}' to dataset '{}'",
                column_name, dataset_name
            )))
        }
    } else {
        // Regular column: <col>: <type> [DEFAULT <val>]
        // Parse column specification: <col>: <type> [DEFAULT <val>]
        // Split by DEFAULT first
        let (col_type_part, default_val) = if let Some(idx) = column_spec.find(" DEFAULT ") {
            let col_type = &column_spec[..idx];
            let default_str = &column_spec[idx + 9..].trim();
            (col_type, Some(parse_single_value(default_str, line_no)?))
        } else {
            (column_spec, None)
        };

        // Parse <col>: <type>
        let col_parts: Vec<&str> = col_type_part.splitn(2, ':').collect();
        if col_parts.len() != 2 {
            return Err(DslError::Parse {
                line: line_no,
                msg: "Expected column definition: <name>: <type>".into(),
            });
        }

        let column_name = col_parts[0].trim().to_string();
        let type_str = col_parts[1].trim();

        // Check if nullable (ends with ?)
        let (type_str_clean, nullable) = if type_str.ends_with('?') {
            (&type_str[..type_str.len() - 1], true)
        } else {
            (type_str, false)
        };

        // Parse type
        let value_type = parse_value_type(type_str_clean, line_no)?;

        // Determine default value
        let default_value = default_val.unwrap_or_else(|| {
            if nullable {
                Value::Null
            } else {
                // Use type-appropriate default
                match value_type {
                    ValueType::Int => Value::Int(0),
                    ValueType::Float => Value::Float(0.0),
                    ValueType::String => Value::String(String::new()),
                    ValueType::Bool => Value::Bool(false),
                    ValueType::Vector(dim) => Value::Vector(vec![0.0; dim]),
                    ValueType::Matrix(r, c) => Value::Matrix(vec![vec![0.0; c]; r]),
                    ValueType::Null => Value::Null,
                }
            }
        });

        // Execute the alteration
        db.alter_dataset_add_column(
            dataset_name,
            column_name.clone(),
            value_type,
            default_value,
            nullable,
        )
        .map_err(|e| DslError::Engine {
            line: line_no,
            source: e,
        })?;

        Ok(DslOutput::Message(format!(
            "Added column '{}' to dataset '{}'",
            column_name, dataset_name
        )))
    }
}

fn parse_select_items(s: &str, line_no: usize) -> Result<Vec<Expr>, DslError> {
    let s = s.trim();
    if s.is_empty() {
        return Err(DslError::Parse {
            line: line_no,
            msg: "Empty SELECT clause".into(),
        });
    }

    let parts = split_args(s);
    let mut exprs = Vec::new();

    use crate::query::logical::{AggregateFunction, Expr};

    for part in parts {
        let part = part.trim();
        if part == "*" {
            exprs.push(Expr::Column("*".to_string()));
            continue;
        }
        // Check for function call: FUNC(col)
        // Check if it looks like Func Call (starts with Name + '(' and ends with ')')
        // Be careful not to match (a+b) as function call.
        if let Some(idx) = part.find('(') {
            let possible_func = part[..idx].trim().to_uppercase();
            // Validate if it is a known function
            let func = match possible_func.as_str() {
                "SUM" => Some(AggregateFunction::Sum),
                "AVG" => Some(AggregateFunction::Avg),
                "COUNT" => Some(AggregateFunction::Count),
                "MIN" => Some(AggregateFunction::Min),
                "MAX" => Some(AggregateFunction::Max),
                _ => None,
            };

            if let Some(f) = func {
                if part.ends_with(')') {
                    let content = &part[idx + 1..part.len() - 1].trim();
                    // Inner expr
                    let inner = if *content == "*" {
                        Expr::Literal(Value::Int(1))
                    } else {
                        parse_expression(content, line_no)?
                    };

                    exprs.push(Expr::AggregateExpr {
                        func: f,
                        expr: Box::new(inner),
                    });
                    continue;
                }
            }
        }

        // If not aggregation function, parse as expression
        exprs.push(parse_expression(part, line_no)?);
    }
    Ok(exprs)
}

fn parse_expression(s: &str, line_no: usize) -> Result<Expr, DslError> {
    parse_expr_add_sub(s, line_no)
}

fn parse_expr_add_sub(s: &str, line_no: usize) -> Result<Expr, DslError> {
    let chars: Vec<char> = s.chars().collect();
    let mut i = chars.len();
    let mut depth = 0;
    let mut last_op_idx = None;
    let mut last_op = ' ';

    while i > 0 {
        i -= 1;
        let c = chars[i];
        if c == ')' {
            depth += 1;
        } else if c == '(' {
            depth -= 1;
        } else if depth == 0 && (c == '+' || c == '-') {
            last_op_idx = Some(i);
            last_op = c;
            break;
        }
    }

    if let Some(idx) = last_op_idx {
        let left_str = s[..idx].trim();
        let right_str = s[idx + 1..].trim();

        // Check if left_str is empty? (Unary ops not supported yet like -5)
        // If left is empty, it's unary?
        if left_str.is_empty() {
            return Err(DslError::Parse {
                line: line_no,
                msg: "Unary operators not supported yet".into(),
            });
        }

        let left = parse_expr_add_sub(left_str, line_no)?;
        let right = parse_term_mul_div(right_str, line_no)?;

        return Ok(Expr::BinaryExpr {
            left: Box::new(left),
            op: last_op.to_string(),
            right: Box::new(right),
        });
    }

    parse_term_mul_div(s, line_no)
}

fn parse_term_mul_div(s: &str, line_no: usize) -> Result<Expr, DslError> {
    let chars: Vec<char> = s.chars().collect();
    let mut i = chars.len();
    let mut depth = 0;
    let mut last_op_idx = None;
    let mut last_op = ' ';

    while i > 0 {
        i -= 1;
        let c = chars[i];
        if c == ')' {
            depth += 1;
        } else if c == '(' {
            depth -= 1;
        } else if depth == 0 && (c == '*' || c == '/') {
            last_op_idx = Some(i);
            last_op = c;
            break;
        }
    }

    if let Some(idx) = last_op_idx {
        let left_str = s[..idx].trim();
        let right_str = s[idx + 1..].trim();

        let left = parse_term_mul_div(left_str, line_no)?;
        let right = parse_factor(right_str, line_no)?;

        return Ok(Expr::BinaryExpr {
            left: Box::new(left),
            op: last_op.to_string(),
            right: Box::new(right),
        });
    }

    parse_factor(s, line_no)
}

/// Handle MATERIALIZE command
/// MATERIALIZE <dataset>.<column> or MATERIALIZE <dataset>
pub fn handle_materialize(db: &mut TensorDb, line: &str, line_no: usize) -> Result<DslOutput, DslError> {
    let rest = line.trim_start_matches("MATERIALIZE").trim();
    
    // Check if it's dataset.column or just dataset
    if rest.contains('.') {
        // MATERIALIZE dataset.column (for now, materialize all lazy columns)
        let dot_idx = rest.find('.').unwrap();
        let dataset_name = rest[..dot_idx].trim();
        let _column_name = rest[dot_idx + 1..].trim();
        
        // For now, materialize all lazy columns (we can optimize later to materialize just one)
        db.materialize_lazy_columns(dataset_name)
            .map_err(|e| DslError::Engine {
                line: line_no,
                source: e,
            })?;
        
        Ok(DslOutput::Message(format!(
            "Materialized lazy columns in dataset '{}'",
            dataset_name
        )))
    } else {
        // MATERIALIZE dataset
        let dataset_name = rest.trim();
        db.materialize_lazy_columns(dataset_name)
            .map_err(|e| DslError::Engine {
                line: line_no,
                source: e,
            })?;
        
        Ok(DslOutput::Message(format!(
            "Materialized lazy columns in dataset '{}'",
            dataset_name
        )))
    }
}

fn parse_factor(s: &str, line_no: usize) -> Result<Expr, DslError> {
    let s = s.trim();
    if s.starts_with('(') && s.ends_with(')') {
        return parse_expression(&s[1..s.len() - 1], line_no);
    }

    if let Ok(val) = parse_single_value(s, line_no) {
        Ok(Expr::Literal(val))
    } else {
        // Assume column.
        Ok(Expr::Column(s.to_string()))
    }
}
