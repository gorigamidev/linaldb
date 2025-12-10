pub mod error;
pub mod handlers;
// pub mod parser; // Not used currently, logic is in handlers/parsing logic

pub use error::DslError;

use crate::engine::TensorDb;
use handlers::{handle_dataset, handle_define, handle_insert, handle_let, handle_show};

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

        if let Err(e) = execute_line(db, line, line_no) {
            // Abort on error
            return Err(e);
        }
    }

    Ok(())
}

/// Ejecuta una sola línea de DSL
pub fn execute_line(db: &mut TensorDb, line: &str, line_no: usize) -> Result<(), DslError> {
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
        handle_dataset(db, line, line_no)
    } else if line.starts_with("INSERT INTO ") {
        handle_insert(db, line, line_no)
    } else {
        // Comment or empty? handled in script, but for single line exec check too
        if line.is_empty() || line.starts_with('#') || line.starts_with("//") {
            return Ok(());
        }
        Err(DslError::Parse {
            line: line_no,
            msg: format!("Unknown command: {}", line),
        })
    }
}
