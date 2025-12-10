use crate::dsl::error::DslError;
use crate::engine::TensorDb;

/// SHOW x
/// SHOW ALL
/// SHOW ALL DATASETS
pub fn handle_show(db: &mut TensorDb, line: &str, line_no: usize) -> Result<(), DslError> {
    let rest = line.trim_start_matches("SHOW").trim();

    if rest == "ALL" || rest == "ALL TENSORS" {
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
    } else if rest == "ALL DATASETS" {
        let mut names = db.list_dataset_names();
        names.sort();
        println!("--- ALL DATASETS ---");
        for name in names {
            if let Ok(dataset) = db.get_dataset(&name) {
                println!(
                    "Dataset: {} (rows: {}, columns: {})",
                    name,
                    dataset.len(),
                    dataset.schema.len()
                );
                // Print schema details
                for field in &dataset.schema.fields {
                    println!("  - {}: {}", field.name, field.value_type);
                }
            }
        }
        println!("--------------------");
        Ok(())
    } else if rest.starts_with("SHAPE ") {
        let name = rest.trim_start_matches("SHAPE ").trim();
        let t = db.get(name).map_err(|e| DslError::Engine {
            line: line_no,
            source: e,
        })?;
        println!("SHAPE {}: {:?}", name, t.shape.dims);
        Ok(())
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
            println!(
                "{}: shape {:?}, len {}, data = {:?}",
                name,
                t.shape.dims,
                t.data.len(),
                t.data
            );
            return Ok(());
        }

        // Check if it's a dataset
        if let Ok(dataset) = db.get_dataset(name) {
            println!(
                "Dataset: {} (rows: {}, columns: {})",
                name,
                dataset.len(),
                dataset.schema.len()
            );
            // Print schema details
            for field in &dataset.schema.fields {
                println!("  - {}: {}", field.name, field.value_type);
            }
            // Print rows (first 10?)
            println!("  Data:");
            for (i, row) in dataset.rows.iter().take(10).enumerate() {
                // Formatting tuple is verbose, simple debug print
                println!("    {}: {:?}", i, row.values);
            }
            return Ok(());
        }

        return Err(DslError::Engine {
            line: line_no,
            source: crate::engine::EngineError::NameNotFound(name.to_string()),
        });
    }
}
