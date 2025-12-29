use crate::dsl::{DslError, DslOutput};
use crate::engine::TensorDb;

/// Handle CREATE DATABASE command
pub fn handle_create_database(
    db: &mut TensorDb,
    line: &str,
    line_no: usize,
) -> Result<DslOutput, DslError> {
    let remainder = line.strip_prefix("CREATE DATABASE ").unwrap().trim();

    let (name, if_not_exists) = if remainder.starts_with("IF NOT EXISTS ") {
        (
            remainder.strip_prefix("IF NOT EXISTS ").unwrap().trim(),
            true,
        )
    } else {
        (remainder, false)
    };

    // The name is the first word before any space or parenthesis
    let name = name
        .split(|c: char| c.is_whitespace() || c == '(')
        .next()
        .unwrap_or("")
        .trim();

    if name.is_empty() {
        return Err(DslError::Parse {
            line: line_no,
            msg: "Database name cannot be empty".to_string(),
        });
    }

    if if_not_exists && db.list_databases().contains(&name.to_string()) {
        return Ok(DslOutput::Message(format!(
            "Database '{}' already exists (skipped)",
            name
        )));
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
    let remainder = line.strip_prefix("DROP DATABASE ").unwrap().trim();

    let (name, if_exists) = if remainder.starts_with("IF EXISTS ") {
        (remainder.strip_prefix("IF EXISTS ").unwrap().trim(), true)
    } else {
        (remainder, false)
    };

    if name.is_empty() {
        return Err(DslError::Parse {
            line: line_no,
            msg: "Database name cannot be empty".to_string(),
        });
    }

    if if_exists && !db.list_databases().contains(&name.to_string()) {
        return Ok(DslOutput::Message(format!(
            "Database '{}' not found (skipped)",
            name
        )));
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
