use crate::dsl::{DslError, DslOutput};
use crate::engine::TensorDb;

/// SHOW x
/// SHOW ALL
/// SHOW ALL DATASETS
pub fn handle_show(db: &mut TensorDb, line: &str, line_no: usize) -> Result<DslOutput, DslError> {
    let rest = line.trim_start_matches("SHOW").trim();

    if rest == "ALL" || rest == "ALL TENSORS" {
        let mut names = db.list_names();
        names.sort();
        let mut output = String::from("--- ALL TENSORS ---\n");
        for name in names {
            if let Ok(t) = db.get(&name) {
                output.push_str(&format!(
                    "{}: shape {:?}, len {}, data = {:?}\n",
                    name,
                    t.shape.dims,
                    t.data.len(),
                    t.data
                ));
            }
        }
        output.push_str("-------------------");
        Ok(DslOutput::Message(output))
    } else if rest == "ALL DATASETS" {
        let mut names = db.list_dataset_names();
        names.sort();
        let mut output = String::from("--- ALL DATASETS ---\n");
        for name in names {
            if let Ok(dataset) = db.get_dataset(&name) {
                output.push_str(&format!(
                    "Dataset: {} (rows: {}, columns: {})\n",
                    name,
                    dataset.len(),
                    dataset.schema.len()
                ));
                for field in &dataset.schema.fields {
                    output.push_str(&format!("  - {}: {}\n", field.name, field.value_type));
                }
            }
        }
        output.push_str("--------------------");
        Ok(DslOutput::Message(output))
    } else if rest == "DATABASES" || rest == "ALL DATABASES" {
        let mut names = db.list_databases();
        names.sort();
        let mut output = String::from("--- ALL DATABASES ---\n");
        for name in names {
            output.push_str(&format!("  - {}\n", name));
        }
        output.push_str("---------------------");
        Ok(DslOutput::Message(output))
    } else if rest.starts_with("INDEXES") {
        let dataset_filter = if rest == "INDEXES" || rest == "ALL INDEXES" {
            None
        } else {
            Some(rest.trim_start_matches("INDEXES ").trim())
        };

        let indices = db.list_indices();
        let mut output = if let Some(ds_name) = dataset_filter {
            format!("--- INDICES FOR {} ---\n", ds_name)
        } else {
            String::from("--- ALL INDICES ---\n")
        };

        output.push_str(&format!(
            "{:<20} {:<20} {:<10}\n",
            "Dataset", "Column", "Type"
        ));
        output.push_str(&format!("{:-<52}\n", ""));

        let mut count = 0;
        for (ds, col, type_str) in indices {
            if let Some(target) = dataset_filter {
                if ds != target {
                    continue;
                }
            }
            output.push_str(&format!("{:<20} {:<20} {:<10}\n", ds, col, type_str));
            count += 1;
        }
        output.push_str("-------------------");

        if count == 0 && dataset_filter.is_some() {
            // Check if dataset exists to give better error message?
            if db.get_dataset(dataset_filter.unwrap()).is_err() {
                return Err(DslError::Engine {
                    line: line_no,
                    source: crate::engine::EngineError::NameNotFound(
                        dataset_filter.unwrap().to_string(),
                    ),
                });
            }
        }

        Ok(DslOutput::Message(output))
    } else if rest.starts_with("LINEAGE ") {
        let name = rest.trim_start_matches("LINEAGE ").trim();
        let tree = db.get_lineage_tree(name).map_err(|e| DslError::Engine {
            line: line_no,
            source: e,
        })?;

        let mut output = format!("Lineage for tensor '{}':\n", name);
        output.push_str(&format_lineage_tree(&tree, 0));
        Ok(DslOutput::Message(output))
    } else if rest.starts_with("SHAPE ") {
        let name = rest.trim_start_matches("SHAPE ").trim();
        let t = db.get(name).map_err(|e| DslError::Engine {
            line: line_no,
            source: e,
        })?;
        Ok(DslOutput::Message(format!(
            "SHAPE {}: {:?}\n",
            name, t.shape.dims
        )))
    } else if rest.starts_with("SCHEMA ") {
        let name = rest.trim_start_matches("SCHEMA ").trim();
        let dataset = db.get_dataset(name).map_err(|e| DslError::Engine {
            line: line_no,
            source: e,
        })?;

        // Build schema output
        let mut output = format!("Schema for dataset '{}':\n", name);
        output.push_str(&format!(
            "{:<20} {:<10} {:<10}\n",
            "Field", "Type", "Nullable"
        ));
        output.push_str(&format!("{:-<42}\n", ""));

        for field in &dataset.schema.fields {
            output.push_str(&format!(
                "{:<20} {:<10} {:<10}\n",
                field.name,
                format!("{:?}", field.value_type),
                field.nullable
            ));
        }

        Ok(DslOutput::Message(output))
    } else {
        let name = rest;
        if name.is_empty() {
            return Err(DslError::Parse {
                line: line_no,
                msg: "Expected: SHOW <name> or SHOW ALL or SHOW ALL DATASETS".into(),
            });
        }

        // Check for string literal
        if name.starts_with('"') && name.ends_with('"') && name.len() >= 2 {
            let content = &name[1..name.len() - 1];
            return Ok(DslOutput::Message(content.to_string()));
        }

        // Check if it's a tensor
        if let Ok(t) = db.get(name) {
            return Ok(DslOutput::Tensor(t.clone()));
        }

        // Check if it's a dataset
        if let Ok(dataset) = db.get_dataset(name) {
            return Ok(DslOutput::Table(dataset.clone()));
        }

        // Check if it's a tensor dataset
        if let Some(ds) = db.get_tensor_dataset(name) {
            let health_info = db.verify_tensor_dataset(name).unwrap_or_default();
            return Ok(DslOutput::TensorTable(ds.clone(), health_info));
        }

        return Err(DslError::Engine {
            line: line_no,
            source: crate::engine::EngineError::NameNotFound(name.to_string()),
        });
    }
}

fn format_lineage_tree(node: &crate::engine::LineageNode, indent: usize) -> String {
    let mut out = String::new();
    let indent_str = "  ".repeat(indent);

    let name_part = if let Some(name) = &node.name {
        format!(" ({})", name)
    } else {
        String::new()
    };

    out.push_str(&format!(
        "{}{}{} [{}]\n",
        indent_str, node.operation, name_part, node.tensor_id.0
    ));

    for input in &node.inputs {
        out.push_str(&format_lineage_tree(input, indent + 1));
    }

    out
}
