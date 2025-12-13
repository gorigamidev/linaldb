pub mod error;
pub mod handlers;
// pub mod parser; // Not used currently, logic is in handlers/parsing logic

pub use error::DslError;

use crate::core::dataset::Dataset;
use crate::core::tensor::Tensor;
use crate::engine::TensorDb;
use handlers::{handle_define, handle_let, handle_show};
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub enum DslOutput {
    None,
    Message(String),
    Table(Dataset),
    Tensor(Tensor),
}

use std::fmt;

impl fmt::Display for DslOutput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DslOutput::None => Ok(()),
            DslOutput::Message(s) => write!(f, "{}", s),
            DslOutput::Table(ds) => {
                writeln!(
                    f,
                    "Dataset: {} (rows: {}, columns: {})",
                    ds.metadata.name.as_deref().unwrap_or("?"),
                    ds.len(),
                    ds.schema.len()
                )?;
                for field in &ds.schema.fields {
                    writeln!(f, "  - {}: {}", field.name, field.value_type)?;
                }
                Ok(())
            }
            DslOutput::Tensor(t) => write!(f, "Tensor: {:?} values: {:?}", t.shape, t.data), // simplified
        }
    }
}

/// Ejecuta un script completo (varias líneas) sobre un TensorDb
pub fn execute_script(db: &mut TensorDb, script: &str) -> Result<(), DslError> {
    for (idx, raw_line) in script.lines().enumerate() {
        let line_no = idx + 1;
        let line = raw_line.trim();

        // Ignorar vacío y comentarios
        if line.is_empty() {
            continue;
        }
        if line.starts_with('#') || line.starts_with("//") {
            continue;
        }

        match execute_line(db, line, line_no) {
            Ok(output) => {
                // For script execution (CLI run), we effectively print the output if it's significant?
                // Or we just ignore unless it's a Message/Show?
                // Current behavior was: Handlers print.
                // New behavior: Handlers return DslOutput. We must print it.
                if !matches!(output, DslOutput::None) {
                    println!("{}", output);
                }
            }
            Err(e) => return Err(e),
        }
    }

    Ok(())
}

/// Ejecuta una sola línea de DSL
pub fn execute_line(db: &mut TensorDb, line: &str, line_no: usize) -> Result<DslOutput, DslError> {
    if line.starts_with("DEFINE ") {
        handle_define(db, line, line_no)
    } else if line.starts_with("VECTOR ") {
        handlers::tensor::handle_vector(db, line, line_no)
    } else if line.starts_with("MATRIX ") {
        handlers::tensor::handle_matrix(db, line, line_no)
    } else if line.starts_with("LET ") {
        handle_let(db, line, line_no)
    } else if line.starts_with("SHOW ") {
        handle_show(db, line, line_no)
    } else if line.starts_with("DATASET ") {
        handlers::dataset::handle_dataset(db, line, line_no)
    } else if line.starts_with("INSERT INTO ") {
        handlers::dataset::handle_insert(db, line, line_no)
    } else if line.starts_with("SEARCH ") {
        handlers::search::handle_search(db, line, line_no)
    } else if line.starts_with("CREATE ") {
        // Check for CREATE INDEX or CREATE VECTOR INDEX
        if line.contains("INDEX ") {
            handlers::index::handle_create_index(db, line, line_no)
        } else {
            // Fallback or Error?
            // Existing code used `handlers::ddl::handle_create`.
            // But we don't have DDL handler. Maybe it's `operations.rs` or unimplemented.
            // Checking `operations.rs` content might be useful, or just return error.
            // For now return error to be safe.
            Err(DslError::Parse {
                line: line_no,
                msg: format!("Unsupported CREATE command: {}", line),
            })
        }
    } else {
        // Comment or empty? handled in script, but for single line exec check too
        if line.is_empty() || line.starts_with('#') || line.starts_with("//") {
            return Ok(DslOutput::None);
        }
        Err(DslError::Parse {
            line: line_no,
            msg: format!("Unknown command: {}", line),
        })
    }
}
