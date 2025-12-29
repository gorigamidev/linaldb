use crate::dsl::{DslError, DslOutput};
use crate::engine::context::ExecutionContext;
use crate::engine::TensorDb;

/// Handle BIND <alias> TO <source>
pub fn handle_bind(db: &mut TensorDb, line: &str, line_no: usize) -> Result<DslOutput, DslError> {
    let rest = line.strip_prefix("BIND").unwrap().trim();

    let parts: Vec<&str> = rest.splitn(2, " TO ").collect();
    if parts.len() != 2 {
        return Err(DslError::Parse {
            line: line_no,
            msg: "Expected syntax: BIND <alias> TO <source>".into(),
        });
    }

    let alias = parts[0].trim();
    let source = parts[1].trim();

    db.active_instance_mut()
        .bind_resource(alias, source)
        .map_err(|e| DslError::Engine {
            line: line_no,
            source: e,
        })?;

    Ok(DslOutput::Message(format!(
        "Bound alias '{}' to resource '{}'",
        alias, source
    )))
}

/// Handle ATTACH <tensor> TO <dataset>.<column>
pub fn handle_attach(db: &mut TensorDb, line: &str, line_no: usize) -> Result<DslOutput, DslError> {
    let rest = line.strip_prefix("ATTACH").unwrap().trim();

    let parts: Vec<&str> = rest.splitn(2, " TO ").collect();
    if parts.len() != 2 {
        return Err(DslError::Parse {
            line: line_no,
            msg: "Expected syntax: ATTACH <tensor> TO <dataset>.<column>".into(),
        });
    }

    let tensor_var = parts[0].trim();
    let target = parts[1].trim();

    if !target.contains('.') {
        return Err(DslError::Parse {
            line: line_no,
            msg: "Target must be in the format <dataset>.<column>".into(),
        });
    }

    let dot_idx = target.find('.').unwrap();
    let ds_name = &target[..dot_idx].trim();
    let col_name = &target[dot_idx + 1..].trim();

    db.active_instance_mut()
        .add_column_to_tensor_dataset(ds_name, col_name, tensor_var)
        .map_err(|e| DslError::Engine {
            line: line_no,
            source: e,
        })?;

    Ok(DslOutput::Message(format!(
        "Attached tensor '{}' as column '{}' in dataset '{}'",
        tensor_var, col_name, ds_name
    )))
}

/// Handle DERIVE <name> FROM <expression>
/// Semantically similar to LET, but emphasizes provenance.
pub fn handle_derive(
    db: &mut TensorDb,
    line: &str,
    line_no: usize,
    ctx: Option<&mut ExecutionContext>,
) -> Result<DslOutput, DslError> {
    let rest = line.strip_prefix("DERIVE").unwrap().trim();

    let parts: Vec<&str> = rest.splitn(2, " FROM ").collect();
    if parts.len() != 2 {
        return Err(DslError::Parse {
            line: line_no,
            msg: "Expected syntax: DERIVE <name> FROM <expression>".into(),
        });
    }

    let target_name = parts[0].trim();
    let expr = parts[1].trim();

    // Re-use LET logic by transforming DERIVE into a LET-like structure internally
    // but keeping DERIVE in the message.
    let let_line = format!("LET {} = {}", target_name, expr);

    // We call handle_let directly
    let mut output = super::operations::handle_let(db, &let_line, line_no, ctx)?;

    // Customize output message
    if let DslOutput::None = output {
        output = DslOutput::Message(format!("Derived '{}' from expression", target_name));
    }

    Ok(output)
}
