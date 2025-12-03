// src/dsl.rs

use crate::engine::{TensorDb, EngineError, BinaryOp, UnaryOp, TensorKind};
use crate::tensor::Shape;

/// Errores del lenguaje de alto nivel (DSL)
#[derive(Debug)]
pub enum DslError {
    Parse { line: usize, msg: String },
    Engine { line: usize, source: EngineError },
}

impl std::fmt::Display for DslError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DslError::Parse { line, msg } => {
                write!(f, "[line {}] Parse error: {}", line, msg)
            }
            DslError::Engine { line, source } => {
                write!(f, "[line {}] Engine error: {}", line, source)
            }
        }
    }
}

impl std::error::Error for DslError {}

/// Ejecuta un script completo (varias líneas) sobre un TensorDb
pub fn execute_script(db: &mut TensorDb, script: &str) -> Result<(), DslError> {
    for (idx, raw_line) in script.lines().enumerate() {
        let line_no = idx + 1;
        let line = raw_line.trim();

        // Ignorar vacío y comentarios
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        if let Err(e) = execute_line(db, line, line_no) {
            eprintln!("WARNING: {}", e);
            // NO devolvemos error, seguimos con el script
        }
    }

    Ok(())
}

/// Ejecuta una sola línea de DSL
pub fn execute_line(db: &mut TensorDb, line: &str, line_no: usize) -> Result<(), DslError> {
    if line.starts_with("DEFINE ") {
        handle_define(db, line, line_no)
    } else if line.starts_with("LET ") {
        handle_let(db, line, line_no)
    } else if line.starts_with("SHOW ") {
        handle_show(db, line, line_no)
    } else {
        Err(DslError::Parse {
            line: line_no,
            msg: format!("Unknown command: {}", line),
        })
    }
}

/// DEFINE a AS TENSOR [3] VALUES [1, 0, 0]
/// DEFINE a AS STRICT TENSOR [3] VALUES [1, 0, 0]
fn handle_define(db: &mut TensorDb, line: &str, line_no: usize) -> Result<(), DslError> {
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
        (TensorKind::Strict, rhs.trim_start_matches("STRICT TENSOR ").trim())
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

    let dims = parse_usize_list(shape_str, line_no)?;
    let shape = Shape::new(dims);

    let values = parse_f32_list(values_str, line_no)?;

    db.insert_named_with_kind(name, shape, values, kind)
        .map_err(|e| DslError::Engine { line: line_no, source: e })
}

/// LET c = ADD a b
/// LET score = CORRELATE a WITH b
/// LET sim = SIMILARITY a WITH b
/// LET half = SCALE a BY 0.5
fn handle_let(db: &mut TensorDb, line: &str, line_no: usize) -> Result<(), DslError> {
    // Quitamos LET
    let rest = line.trim_start_matches("LET").trim();

    // output = ...
    let parts: Vec<&str> = rest.splitn(2, '=').collect();
    if parts.len() != 2 {
        return Err(DslError::Parse {
            line: line_no,
            msg: "Expected: LET <name> = ...".into(),
        });
    }

    let output_name = parts[0].trim();
    if output_name.is_empty() {
        return Err(DslError::Parse {
            line: line_no,
            msg: "Missing output name in LET".into(),
        });
    }

    let expr = parts[1].trim();
    let tokens: Vec<&str> = expr.split_whitespace().collect();
    if tokens.is_empty() {
        return Err(DslError::Parse {
            line: line_no,
            msg: "Missing expression in LET".into(),
        });
    }

        match tokens[0] {
        "ADD" => {
            if tokens.len() != 3 {
                return Err(DslError::Parse {
                    line: line_no,
                    msg: "Expected: LET x = ADD a b".into(),
                });
            }
            let left = tokens[1];
            let right = tokens[2];
            db.eval_binary(output_name, left, right, BinaryOp::Add)
                .map_err(|e| DslError::Engine { line: line_no, source: e })
        }
        "SUBTRACT" => {
            if tokens.len() != 3 {
                return Err(DslError::Parse {
                    line: line_no,
                    msg: "Expected: LET x = SUBTRACT a b".into(),
                });
            }
            let left = tokens[1];
            let right = tokens[2];
            db.eval_binary(output_name, left, right, BinaryOp::Subtract)
                .map_err(|e| DslError::Engine { line: line_no, source: e })
        }
        "MULTIPLY" => {
            if tokens.len() != 3 {
                return Err(DslError::Parse {
                    line: line_no,
                    msg: "Expected: LET x = MULTIPLY a b".into(),
                });
            }
            let left = tokens[1];
            let right = tokens[2];
            db.eval_binary(output_name, left, right, BinaryOp::Multiply)
                .map_err(|e| DslError::Engine { line: line_no, source: e })
        }
        "DIVIDE" => {
            if tokens.len() != 3 {
                return Err(DslError::Parse {
                    line: line_no,
                    msg: "Expected: LET x = DIVIDE a b".into(),
                });
            }
            let left = tokens[1];
            let right = tokens[2];
            db.eval_binary(output_name, left, right, BinaryOp::Divide)
                .map_err(|e| DslError::Engine { line: line_no, source: e })
        }
        "CORRELATE" => {
            // CORRELATE a WITH b
            if tokens.len() != 4 || tokens[2] != "WITH" {
                return Err(DslError::Parse {
                    line: line_no,
                    msg: "Expected: LET x = CORRELATE a WITH b".into(),
                });
            }
            let left = tokens[1];
            let right = tokens[3];
            db.eval_binary(output_name, left, right, BinaryOp::Correlate)
                .map_err(|e| DslError::Engine { line: line_no, source: e })
        }
        "SIMILARITY" => {
            // SIMILARITY a WITH b
            if tokens.len() != 4 || tokens[2] != "WITH" {
                return Err(DslError::Parse {
                    line: line_no,
                    msg: "Expected: LET x = SIMILARITY a WITH b".into(),
                });
            }
            let left = tokens[1];
            let right = tokens[3];
            db.eval_binary(output_name, left, right, BinaryOp::Similarity)
                .map_err(|e| DslError::Engine { line: line_no, source: e })
        }
        "DISTANCE" => {
            // DISTANCE a TO b
            if tokens.len() != 4 || tokens[2] != "TO" {
                return Err(DslError::Parse {
                    line: line_no,
                    msg: "Expected: LET x = DISTANCE a TO b".into(),
                });
            }
            let left = tokens[1];
            let right = tokens[3];
            db.eval_binary(output_name, left, right, BinaryOp::Distance)
                .map_err(|e| DslError::Engine { line: line_no, source: e })
        }
        "SCALE" => {
            // SCALE a BY 0.5
            if tokens.len() != 4 || tokens[2] != "BY" {
                return Err(DslError::Parse {
                    line: line_no,
                    msg: "Expected: LET x = SCALE a BY <number>".into(),
                });
            }
            let input_name = tokens[1];
            let factor: f32 = tokens[3].parse().map_err(|_| DslError::Parse {
                line: line_no,
                msg: format!("Invalid scale factor: {}", tokens[3]),
            })?;
            db.eval_unary(output_name, input_name, UnaryOp::Scale(factor))
                .map_err(|e| DslError::Engine { line: line_no, source: e })
        }
        "NORMALIZE" => {
            // NORMALIZE a
            if tokens.len() != 2 {
                return Err(DslError::Parse {
                    line: line_no,
                    msg: "Expected: LET x = NORMALIZE a".into(),
                });
            }
            let input_name = tokens[1];
            db.eval_unary(output_name, input_name, UnaryOp::Normalize)
                .map_err(|e| DslError::Engine { line: line_no, source: e })
        }
        other => Err(DslError::Parse {
            line: line_no,
            msg: format!("Unknown LET operation: {}", other),
        }),
    }
}

/// SHOW x
/// SHOW ALL
fn handle_show(db: &mut TensorDb, line: &str, line_no: usize) -> Result<(), DslError> {
    let rest = line.trim_start_matches("SHOW").trim();

    if rest == "ALL" {
        let mut names = db.list_names();
        names.sort();
        println!("--- ALL TENSORS ---");
        for name in names {
            if let Ok(t) = db.get(&name) {
                println!(
                    "{}: shape {:?}, len {}, data = {:?}",
                    name,
                    t.shape.dims,
                    t.data.len(),
                    t.data
                );
            }
        }
        println!("-------------------");
        Ok(())
    } else {
        let name = rest;
        if name.is_empty() {
            return Err(DslError::Parse {
                line: line_no,
                msg: "Expected: SHOW <name> or SHOW ALL".into(),
            });
        }
        let t = db
            .get(name)
            .map_err(|e| DslError::Engine { line: line_no, source: e })?;
        println!(
            "{}: shape {:?}, len {}, data = {:?}",
            name,
            t.shape.dims,
            t.data.len(),
            t.data
        );
        Ok(())
    }
}

/// Parse de algo como: [1, 3, 224, 224]
fn parse_usize_list(text: &str, line_no: usize) -> Result<Vec<usize>, DslError> {
    let inner = text.trim();
    if !inner.starts_with('[') || !inner.ends_with(']') {
        return Err(DslError::Parse {
            line: line_no,
            msg: format!("Expected [d1, d2, ...], got: {}", text),
        });
    }
    let inner = &inner[1..inner.len() - 1]; // sin [ ]
    if inner.trim().is_empty() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for part in inner.split(',') {
        let p = part.trim();
        let n: usize = p.parse().map_err(|_| DslError::Parse {
            line: line_no,
            msg: format!("Invalid dimension: {}", p),
        })?;
        out.push(n);
    }
    Ok(out)
}

/// Parse de algo como: [1, 0, 0] a Vec<f32>
fn parse_f32_list(text: &str, line_no: usize) -> Result<Vec<f32>, DslError> {
    let inner = text.trim();
    if !inner.starts_with('[') || !inner.ends_with(']') {
        return Err(DslError::Parse {
            line: line_no,
            msg: format!("Expected [v1, v2, ...], got: {}", text),
        });
    }
    let inner = &inner[1..inner.len() - 1]; // sin [ ]
    if inner.trim().is_empty() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for part in inner.split(',') {
        let p = part.trim();
        let n: f32 = p.parse().map_err(|_| DslError::Parse {
            line: line_no,
            msg: format!("Invalid float: {}", p),
        })?;
        out.push(n);
    }
    Ok(out)
}
