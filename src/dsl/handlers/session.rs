use crate::dsl::{DslError, DslOutput};
use crate::engine::TensorDb;

/// Handle SESSION commands
/// Syntax: RESET SESSION
pub fn handle_session(
    db: &mut TensorDb,
    line: &str,
    line_no: usize,
) -> Result<DslOutput, DslError> {
    let rest = line.strip_prefix("RESET ").unwrap_or(line).trim();

    if rest == "SESSION" {
        db.reset_session();
        Ok(DslOutput::Message(
            "Session reset complete. All in-memory data has been cleared from the active database."
                .to_string(),
        ))
    } else {
        Err(DslError::Parse {
            line: line_no,
            msg: format!("Unknown SESSION command: {}", line),
        })
    }
}
