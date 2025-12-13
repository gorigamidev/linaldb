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

    let keywords = ["FILTER", "SELECT", "ORDER BY", "LIMIT"];
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
        } else if clauses_trimmed.starts_with("SELECT ") {
            let (cols_str, remaining) = split_clause(clauses_trimmed, "SELECT", &keywords);
            clauses_str = remaining;

            let cols: Vec<String> = cols_str.split(',').map(|s| s.trim().to_string()).collect();
            current_plan = LogicalPlan::Project {
                input: Box::new(current_plan),
                columns: cols,
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

    let mut fields = Vec::new();

    // Split by comma
    for col_def in inner.split(',') {
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
    } else {
        Err(DslError::Parse {
            line: line_no,
            msg: format!("Unknown type: {}", type_str),
        })
    }
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

/// Parse a single value
/// Re-used from existing implementation
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

    // Vector [val1, val2, ...]
    if s.starts_with('[') && s.ends_with(']') {
        let content = &s[1..s.len() - 1];
        let parts: Vec<&str> = content.split(',').map(|s| s.trim()).collect();
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

/// Handle DATASET <name> ADD COLUMN <col>: <type> [DEFAULT <val>]
fn handle_add_column(db: &mut TensorDb, line: &str, line_no: usize) -> Result<DslOutput, DslError> {
    let rest = line.trim_start_matches("DATASET").trim();

    // Split into dataset name and ADD COLUMN part
    let parts: Vec<&str> = rest.splitn(2, " ADD COLUMN ").collect();
    if parts.len() != 2 {
        return Err(DslError::Parse {
            line: line_no,
            msg: "Expected: DATASET <name> ADD COLUMN <col>: <type> [DEFAULT <val>]".into(),
        });
    }

    let dataset_name = parts[0].trim();
    let column_spec = parts[1].trim();

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
