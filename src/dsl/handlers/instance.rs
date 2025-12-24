use crate::dsl::{DslError, DslOutput};
use crate::engine::TensorDb;

/// Handle CREATE DATABASE command
pub fn handle_create_database(
    db: &mut TensorDb,
    line: &str,
    line_no: usize,
) -> Result<DslOutput, DslError> {
    let name = line.strip_prefix("CREATE DATABASE ").unwrap().trim();
    if name.is_empty() {
        return Err(DslError::Parse {
            line: line_no,
            msg: "Database name cannot be empty".to_string(),
        });
    }

    db.create_database(name.to_string())
        .map_err(|e| DslError::Engine {
            line: line_no,
            source: e,
        })?;

    Ok(DslOutput::Message(format!("Database '{}' created", name)))
}

/// Handle USE command
pub fn handle_use_database(
    db: &mut TensorDb,
    line: &str,
    line_no: usize,
) -> Result<DslOutput, DslError> {
    let name = line.strip_prefix("USE ").unwrap().trim();
    if name.is_empty() {
        return Err(DslError::Parse {
            line: line_no,
            msg: "Database name cannot be empty".to_string(),
        });
    }

    db.use_database(name).map_err(|e| DslError::Engine {
        line: line_no,
        source: e,
    })?;

    Ok(DslOutput::Message(format!(
        "Switched to database '{}'",
        name
    )))
}

/// Handle DROP DATABASE command
pub fn handle_drop_database(
    db: &mut TensorDb,
    line: &str,
    line_no: usize,
) -> Result<DslOutput, DslError> {
    let name = line.strip_prefix("DROP DATABASE ").unwrap().trim();
    if name.is_empty() {
        return Err(DslError::Parse {
            line: line_no,
            msg: "Database name cannot be empty".to_string(),
        });
    }

    db.drop_database(name).map_err(|e| DslError::Engine {
        line: line_no,
        source: e,
    })?;

    Ok(DslOutput::Message(format!("Database '{}' dropped", name)))
}

/// Handle SHOW DATABASES command
pub fn handle_show_databases(db: &TensorDb) -> Result<DslOutput, DslError> {
    let mut names = db.list_databases();
    names.sort();

    let msg = if names.is_empty() {
        "No databases found".to_string()
    } else {
        format!("Databases:\n  - {}", names.join("\n  - "))
    };

    Ok(DslOutput::Message(msg))
}
