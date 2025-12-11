use crate::core::tensor::Shape;
use crate::dsl::{DslError, DslOutput};
use crate::engine::{TensorDb, TensorKind};
use crate::utils::parsing::{parse_f32_list, parse_usize_list};

/// DEFINE a AS TENSOR [3] VALUES [1, 0, 0]
/// DEFINE a AS STRICT TENSOR [3] VALUES [1, 0, 0]
pub fn handle_define(db: &mut TensorDb, line: &str, line_no: usize) -> Result<DslOutput, DslError> {
    // Quitamos el prefijo DEFINE
    let rest = line.trim_start_matches("DEFINE").trim();

    // name AS ...
    let parts: Vec<&str> = rest.splitn(2, " AS ").collect();
    if parts.len() != 2 {
        return Err(DslError::Parse {
            line: line_no,
            msg: "Expected: DEFINE <name> AS [STRICT] TENSOR [dims] VALUES [values]".into(),
        });
    }

    let name = parts[0].trim();
    if name.is_empty() {
        return Err(DslError::Parse {
            line: line_no,
            msg: "Missing tensor name after DEFINE".into(),
        });
    }

    let rhs = parts[1].trim();

    // Detectar STRICT o no
    let (kind, tail) = if rhs.starts_with("STRICT TENSOR ") {
        (
            TensorKind::Strict,
            rhs.trim_start_matches("STRICT TENSOR ").trim(),
        )
    } else if rhs.starts_with("TENSOR ") {
        (TensorKind::Normal, rhs.trim_start_matches("TENSOR ").trim())
    } else {
        return Err(DslError::Parse {
            line: line_no,
            msg: "Expected: AS TENSOR ... or AS STRICT TENSOR ...".into(),
        });
    };

    // tail: [dims] VALUES [values]
    let parts2: Vec<&str> = tail.splitn(2, " VALUES ").collect();
    if parts2.len() != 2 {
        return Err(DslError::Parse {
            line: line_no,
            msg: "Expected: ... [dims] VALUES [values]".into(),
        });
    }

    let shape_str = parts2[0].trim();
    let values_str = parts2[1].trim();

    // Use map_err to convert String error to DslError
    let dims = parse_usize_list(shape_str).map_err(|msg| DslError::Parse { line: line_no, msg })?;
    let shape = Shape::new(dims);

    let values =
        parse_f32_list(values_str).map_err(|msg| DslError::Parse { line: line_no, msg })?;

    db.insert_named_with_kind(name, shape, values, kind)
        .map_err(|e| DslError::Engine {
            line: line_no,
            source: e,
        })
        .map(|_| DslOutput::Message(format!("Defined tensor: {}", name)))
}

/// VECTOR v = [1, 2, 3]
pub fn handle_vector(db: &mut TensorDb, line: &str, line_no: usize) -> Result<DslOutput, DslError> {
    let rest = line.trim_start_matches("VECTOR").trim();
    let parts: Vec<&str> = rest.splitn(2, '=').collect();
    if parts.len() != 2 {
        return Err(DslError::Parse {
            line: line_no,
            msg: "Expected: VECTOR <name> = [<values>]".into(),
        });
    }

    let name = parts[0].trim();
    if name.is_empty() {
        return Err(DslError::Parse {
            line: line_no,
            msg: "Missing vector name".into(),
        });
    }

    let values_str = parts[1].trim();
    let values =
        parse_f32_list(values_str).map_err(|msg| DslError::Parse { line: line_no, msg })?;
    let shape = Shape::new(vec![values.len()]);

    db.insert_named_with_kind(name, shape, values, TensorKind::Normal)
        .map_err(|e| DslError::Engine {
            line: line_no,
            source: e,
        })
        .map(|_| DslOutput::Message(format!("Defined vector: {}", name)))
}

/// MATRIX m = [[1, 2], [3, 4]]
pub fn handle_matrix(db: &mut TensorDb, line: &str, line_no: usize) -> Result<DslOutput, DslError> {
    let rest = line.trim_start_matches("MATRIX").trim();
    let parts: Vec<&str> = rest.splitn(2, '=').collect();
    if parts.len() != 2 {
        return Err(DslError::Parse {
            line: line_no,
            msg: "Expected: MATRIX <name> = [[values], [values]]".into(),
        });
    }

    let name = parts[0].trim();
    if name.is_empty() {
        return Err(DslError::Parse {
            line: line_no,
            msg: "Missing matrix name".into(),
        });
    }

    let values_str = parts[1].trim();
    let (rows, cols, values) =
        parse_matrix_values(values_str).map_err(|msg| DslError::Parse { line: line_no, msg })?;

    let shape = Shape::new(vec![rows, cols]);
    db.insert_named_with_kind(name, shape, values, TensorKind::Normal)
        .map_err(|e| DslError::Engine {
            line: line_no,
            source: e,
        })
        .map(|_| DslOutput::Message(format!("Defined matrix: {}", name)))
}

fn parse_matrix_values(s: &str) -> Result<(usize, usize, Vec<f32>), String> {
    // Expected [[1, 2], [3, 4]]
    // Simple manual parser
    let s = s.trim();
    if !s.starts_with('[') || !s.ends_with(']') {
        return Err("Matrix must be enclosed in [...]".into());
    }

    let inner = &s[1..s.len() - 1];

    let mut rows_data = Vec::new();
    let mut current_row_str = String::new();
    let mut in_row = false;

    for ch in inner.chars() {
        if ch == '[' {
            if in_row {
                return Err("Nested arrays not supported beyond 2D".into());
            }
            in_row = true;
            current_row_str.clear();
        } else if ch == ']' {
            if !in_row {
                return Err("Unexpected ]".into());
            }
            in_row = false;
            // Parse row
            let row_vals = parse_f32_list(&format!("[{}]", current_row_str))?; // reuse parse_f32_list expecting brackets
            rows_data.push(row_vals);
        } else if in_row {
            current_row_str.push(ch);
        } else if ch == ',' || ch.is_whitespace() {
            // ignore delimiters between rows
        } else {
            return Err(format!("Unexpected character between rows: {}", ch));
        }
    }

    if rows_data.is_empty() {
        return Err("Empty matrix".into());
    }

    let rows = rows_data.len();
    let cols = rows_data[0].len();

    let mut all_values = Vec::with_capacity(rows * cols);

    for (i, row) in rows_data.iter().enumerate() {
        if row.len() != cols {
            return Err(format!(
                "Row {} has wrong length {}, expected {}",
                i,
                row.len(),
                cols
            ));
        }
        all_values.extend_from_slice(row);
    }

    Ok((rows, cols, all_values))
}
