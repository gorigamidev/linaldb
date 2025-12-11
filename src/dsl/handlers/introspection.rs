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
    } else {
        let name = rest;
        if name.is_empty() {
            return Err(DslError::Parse {
                line: line_no,
                msg: "Expected: SHOW <name> or SHOW ALL or SHOW ALL DATASETS".into(),
            });
        }
        // Check if it's a tensor
        if let Ok(t) = db.get(name) {
            return Ok(DslOutput::Tensor(t.clone()));
        }

        // Check if it's a dataset
        if let Ok(dataset) = db.get_dataset(name) {
            return Ok(DslOutput::Table(dataset.clone()));
        }

        return Err(DslError::Engine {
            line: line_no,
            source: crate::engine::EngineError::NameNotFound(name.to_string()),
        });
    }
}
