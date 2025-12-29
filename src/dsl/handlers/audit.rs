use crate::dsl::{DslError, DslOutput};
use crate::engine::TensorDb;

/// AUDIT DATASET <name>
pub fn handle_audit(db: &mut TensorDb, line: &str, line_no: usize) -> Result<DslOutput, DslError> {
    let rest = line.strip_prefix("AUDIT").unwrap().trim();

    if rest.starts_with("DATASET ") {
        let ds_name = rest.trim_start_matches("DATASET ").trim();
        let issues = db
            .verify_tensor_dataset(ds_name)
            .map_err(|e| DslError::Engine {
                line: line_no,
                source: e,
            })?;

        if issues.is_empty() {
            Ok(DslOutput::Message(format!(
                "Audit PASSED for dataset '{}'. All column references are valid.",
                ds_name
            )))
        } else {
            let msg = format!(
                "Audit FAILED for dataset '{}'. The following columns point to missing or invalid tensors: {:?}",
                ds_name, issues
            );
            Ok(DslOutput::Message(msg))
        }
    } else {
        Err(DslError::Parse {
            line: line_no,
            msg: "Expected syntax: AUDIT DATASET <name>".into(),
        })
    }
}
