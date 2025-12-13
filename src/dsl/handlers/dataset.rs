use crate::core::tuple::{Field, Schema, Tuple};
use crate::core::value::{Value, ValueType};
use crate::engine::TensorDb;
use std::cmp::Ordering;
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

/// DATASET target FROM source [FILTER col > val] [SELECT col1, col2] [ORDER BY col [DESC]] [LIMIT n]
fn handle_dataset_query(
    db: &mut TensorDb,
    line: &str,
    line_no: usize,
) -> Result<DslOutput, DslError> {
    let rest = line.trim_start_matches("DATASET").trim();

    // Split into target and FROM source...
    let parts: Vec<&str> = rest.splitn(2, " FROM ").collect();
    if parts.len() != 2 {
        return Err(DslError::Parse {
            line: line_no,
            msg: "Expected: DATASET target FROM source ...".into(),
        });
    }

    let target_name = parts[0].trim();
    let query_part = parts[1].trim();

    // Parse source name (first word of query_part) and the rest
    // But query_part might just be source_name if no other clauses.
    // We need to find the start of the next clause.
    // Clauses: FILTER, SELECT, ORDER BY, LIMIT
    // We can assume they appear in that order for simplicity, or we can parse iteratively.
    // Iterative parsing:
    // source_name is until the first whitespace that is followed by a keyword?
    // Or just look for keywords.

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

    // Get source dataset (snapshot)
    let source_ds = db.get_dataset(source_name).map_err(|e| DslError::Engine {
        line: line_no,
        source: e,
    })?;
    let mut working_ds = source_ds.clone();

    // Process clauses
    // Strategy: consume clauses_str until empty
    while !clauses_str.is_empty() {
        let clauses_trimmed = clauses_str.trim();

        if clauses_trimmed.starts_with("FILTER ") {
            let (cond_str, remaining) = split_clause(clauses_trimmed, "FILTER", &keywords);
            clauses_str = remaining;

            // Parse condition: col > val
            let (col, op, val) = parse_filter_condition(cond_str, line_no)?;

            // Check column exists
            let col_idx =
                working_ds
                    .schema
                    .get_field_index(&col)
                    .ok_or_else(|| DslError::Parse {
                        line: line_no,
                        msg: format!("Column '{}' not found in dataset", col),
                    })?;

            working_ds = working_ds.filter(|row| {
                let row_val = &row.values[col_idx];
                evaluate_condition(row_val, &op, &val)
            });
        } else if clauses_trimmed.starts_with("SELECT ") {
            let (cols_str, remaining) = split_clause(clauses_trimmed, "SELECT", &keywords);
            clauses_str = remaining;

            let cols: Vec<&str> = cols_str.split(',').map(|s| s.trim()).collect();
            working_ds = working_ds.select(&cols).map_err(|e| DslError::Parse {
                line: line_no,
                msg: e,
            })?;
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
            let col_name = parts[0];
            let ascending = if parts.len() > 1 && parts[1] == "DESC" {
                false
            } else {
                true
            };

            working_ds = working_ds
                .sort_by(col_name, ascending)
                .map_err(|e| DslError::Parse {
                    line: line_no,
                    msg: e,
                })?;
        } else if clauses_trimmed.starts_with("LIMIT ") {
            let (limit_str, remaining) = split_clause(clauses_trimmed, "LIMIT", &keywords);
            clauses_str = remaining;

            let n: usize = limit_str.trim().parse().map_err(|_| DslError::Parse {
                line: line_no,
                msg: format!("Invalid LIMIT: {}", limit_str),
            })?;

            working_ds = working_ds.take(n);
        } else {
            return Err(DslError::Parse {
                line: line_no,
                msg: format!("Unexpected clause: {}", clauses_str),
            });
        }
    }

    // Save result
    db.create_dataset(target_name.to_string(), working_ds.schema.clone())
        .map_err(|e| DslError::Engine {
            line: line_no,
            source: e,
        })?;

    // We created an empty dataset with the schema, now we need to insert the rows.
    // The `create_dataset` method creates a NEW empty dataset.
    // `working_ds` has the rows we want, but it's a separate object (with same ID as source initially, then modified).
    // The `filter`, `select` etc return a NEW dataset object but reusing schema/metadata.
    // If we want to store it in DB, we should overwrite the dataset in the store or insert it.
    // `create_dataset` in `db.rs` creates a NEW one and puts it in `dataset_store`.
    // But `working_ds` IS a `Dataset`. We should insert THIS dataset directly into the store.
    // However, `TensorDb` only exposes `create_dataset` (which makes a new one).
    // Checking `TensorDb` implementation... `create_dataset` calls `dataset_store.insert`.
    // `dataset_store.insert(dataset, name)`.
    // `working_ds` has the ID of the source. We should probably give it a new ID?
    // Or `dataset_store` assigns one?
    // `create_dataset` in `db.rs`: `let id = self.dataset_store.gen_id(); let dataset = Dataset::new(id, schema, ...); self.dataset_store.insert...`.

    // I need a way to insert an existing `Dataset` object into `TensorDb`.
    // `db.rs` doesn't seem to have `insert_dataset_object`.
    // I should add it? Or recreate it.
    // Recreating: `create_dataset` -> empty. Then `insert_row` for every row? Expensive but works with current API.
    // Better: Add helper to `TensorDb` or modify `create_dataset`.
    // Or allow `DatasetStore` to take the object.

    // For now, I'll iterate and insert rows. It's safe given I can't modify `engine/db.rs` easily without another tool call and I am in `dsl/handlers/dataset.rs`.
    // But wait, `working_ds` has `rows` field which is `pub`.
    // Using `db.get_dataset_mut(target_name)` gives me access to the new empty dataset.
    // I can just replace its rows!

    let target_ds = db
        .get_dataset_mut(target_name)
        .map_err(|e| DslError::Engine {
            line: line_no,
            source: e,
        })?;
    target_ds.rows = working_ds.rows;
    // And metadata/stats?
    target_ds.metadata = working_ds.metadata; // If compatible? stats are updated.

    Ok(DslOutput::None)
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

fn evaluate_condition(val: &Value, op: &str, target: &Value) -> bool {
    let ord = val.compare(target);
    match op {
        "=" => ord == Some(Ordering::Equal),
        "!=" => ord.is_some() && ord != Some(Ordering::Equal),
        ">" => ord == Some(Ordering::Greater),
        "<" => ord == Some(Ordering::Less),
        ">=" => ord == Some(Ordering::Greater) || ord == Some(Ordering::Equal),
        "<=" => ord == Some(Ordering::Less) || ord == Some(Ordering::Equal),
        _ => false,
    }
}

// ... existing code ...

/// Parse column definitions from: (col1: TYPE1, col2: TYPE2, ...)
fn parse_column_definitions(columns_str: &str, line_no: usize) -> Result<Vec<Field>, DslError> {
    // Remove parentheses
    let inner = columns_str
        .trim()
        .trim_start_matches('(')
        .trim_end_matches(')')
        .trim();

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
    match type_str.to_uppercase().as_str() {
        "INT" => Ok(ValueType::Int),
        "FLOAT" => Ok(ValueType::Float),
        "STRING" => Ok(ValueType::String),
        "BOOL" => Ok(ValueType::Bool),
        _ => Err(DslError::Parse {
            line: line_no,
            msg: format!("Unknown type: {}", type_str),
        }),
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
fn parse_single_value(s: &str, line_no: usize) -> Result<Value, DslError> {
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
    if s.contains('.') {
        return s
            .parse::<f32>()
            .map(Value::Float)
            .map_err(|_| DslError::Parse {
                line: line_no,
                msg: format!("Invalid float: {}", s),
            });
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
