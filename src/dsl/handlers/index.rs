use crate::dsl::error::DslError;
use crate::dsl::DslOutput;
use crate::engine::TensorDb;

/// Handle CREATE INDEX commands
/// Syntax:
/// CREATE INDEX idx_name ON dataset(column)
/// CREATE VECTOR INDEX idx_name ON dataset(column)
pub fn handle_create_index(
    db: &mut TensorDb,
    input: &str,
    line_no: usize,
) -> Result<DslOutput, DslError> {
    // Expected formats:
    // CREATE INDEX idx_name ON dataset(column)
    // CREATE VECTOR INDEX idx_name ON dataset(column)

    let parts: Vec<&str> = input.split_whitespace().collect();

    // Check if VECTOR is present
    let is_vector = parts.get(1).map(|s| *s == "VECTOR").unwrap_or(false);

    let idx_name_pos = if is_vector { 3 } else { 2 };
    let on_keyword_pos = if is_vector { 4 } else { 3 };
    let target_pos = if is_vector { 5 } else { 4 };

    if parts.len() <= target_pos || parts[on_keyword_pos] != "ON" {
        return Err(DslError::Parse {
            line: line_no,
            msg: "Invalid syntax. Expected: CREATE [VECTOR] INDEX name ON dataset(column)".into(),
        });
    }

    let idx_name = parts[idx_name_pos];
    let target_input = parts[target_pos]; // dataset(column) or dataset.column?
                                          // The prompt suggested dataset(column), let's support that or dataset.column

    // Parse dataset and column
    let (dataset_name, column_name) = if let Some(start) = target_input.find('(') {
        if let Some(end) = target_input.find(')') {
            let ds = &target_input[..start];
            let col = &target_input[start + 1..end];
            (ds, col)
        } else {
            return Err(DslError::Parse {
                line: line_no,
                msg: "Missing closing parenthesis in dataset(column)".into(),
            });
        }
    } else if let Some(dot) = target_input.find('.') {
        let ds = &target_input[..dot];
        let col = &target_input[dot + 1..];
        (ds, col)
    } else {
        return Err(DslError::Parse {
            line: line_no,
            msg: "Invalid target format. Use dataset(column)".into(),
        });
    };

    if is_vector {
        db.create_vector_index(dataset_name, column_name)
            .map_err(|e| DslError::Engine {
                line: line_no,
                source: e,
            })?;
        Ok(DslOutput::Message(format!(
            "Created VECTOR index '{}' on {}({})",
            idx_name, dataset_name, column_name
        )))
    } else {
        db.create_index(dataset_name, column_name)
            .map_err(|e| DslError::Engine {
                line: line_no,
                source: e,
            })?;
        Ok(DslOutput::Message(format!(
            "Created HASH index '{}' on {}({})",
            idx_name, dataset_name, column_name
        )))
    }
}
