use crate::core::storage::{ParquetStorage, StorageEngine};
use crate::dsl::{DslError, DslOutput};
use crate::engine::TensorDb;

/// Handle SAVE command
/// Syntax: SAVE DATASET dataset_name TO "path"
///         SAVE TENSOR tensor_name TO "path"
pub fn handle_save(db: &mut TensorDb, line: &str, line_no: usize) -> Result<DslOutput, DslError> {
    let rest = line.strip_prefix("SAVE ").unwrap().trim();
    
    if rest.starts_with("DATASET ") {
        handle_save_dataset(db, rest, line_no)
    } else if rest.starts_with("TENSOR ") {
        handle_save_tensor(db, rest, line_no)
    } else {
        Err(DslError::Parse {
            line: line_no,
            msg: "Expected 'DATASET' or 'TENSOR' after 'SAVE'".to_string(),
        })
    }
}

fn handle_save_dataset(db: &mut TensorDb, rest: &str, line_no: usize) -> Result<DslOutput, DslError> {
    let rest = rest.strip_prefix("DATASET ").unwrap().trim();
    
    // Find TO keyword
    let parts: Vec<&str> = rest.splitn(2, " TO ").collect();
    if parts.len() != 2 {
        return Err(DslError::Parse {
            line: line_no,
            msg: "Expected 'TO' keyword in SAVE command".to_string(),
        });
    }
    
    let dataset_name = parts[0].trim();
    let path = parts[1].trim().trim_matches('"');
    
    // Get dataset from store using public method
    let dataset = db.get_dataset(dataset_name)
        .map_err(|e| DslError::Engine {
            line: line_no,
            source: e,
        })?;
    
    // Save using storage engine
    let storage = ParquetStorage::new(path);
    storage.save_dataset(dataset)
        .map_err(|e| DslError::Parse {
            line: line_no,
            msg: format!("Failed to save dataset: {}", e),
        })?;
    
    Ok(DslOutput::Message(format!(
        "Saved dataset '{}' to '{}'",
        dataset_name, path
    )))
}

fn handle_save_tensor(db: &mut TensorDb, rest: &str, line_no: usize) -> Result<DslOutput, DslError> {
    let rest = rest.strip_prefix("TENSOR ").unwrap().trim();
    
    // Find TO keyword
    let parts: Vec<&str> = rest.splitn(2, " TO ").collect();
    if parts.len() != 2 {
        return Err(DslError::Parse {
            line: line_no,
            msg: "Expected 'TO' keyword in SAVE TENSOR command".to_string(),
        });
    }
    
    let tensor_name = parts[0].trim();
    let path = parts[1].trim().trim_matches('"');
    
    // Get tensor from db
    let tensor = db.get(tensor_name)
        .map_err(|e| DslError::Engine {
            line: line_no,
            source: e,
        })?;
    
    // Save using storage engine
    let storage = ParquetStorage::new(path);
    storage.save_tensor(tensor_name, tensor)
        .map_err(|e| DslError::Parse {
            line: line_no,
            msg: format!("Failed to save tensor: {}", e),
        })?;
    
    Ok(DslOutput::Message(format!(
        "Saved tensor '{}' to '{}'",
        tensor_name, path
    )))
}

/// Handle LOAD command
/// Syntax: LOAD DATASET dataset_name FROM "path"
///         LOAD TENSOR tensor_name FROM "path"
pub fn handle_load(db: &mut TensorDb, line: &str, line_no: usize) -> Result<DslOutput, DslError> {
    let rest = line.strip_prefix("LOAD ").unwrap().trim();
    
    if rest.starts_with("DATASET ") {
        handle_load_dataset(db, rest, line_no)
    } else if rest.starts_with("TENSOR ") {
        handle_load_tensor(db, rest, line_no)
    } else {
        Err(DslError::Parse {
            line: line_no,
            msg: "Expected 'DATASET' or 'TENSOR' after 'LOAD'".to_string(),
        })
    }
}

fn handle_load_dataset(db: &mut TensorDb, rest: &str, line_no: usize) -> Result<DslOutput, DslError> {
    let rest = rest.strip_prefix("DATASET ").unwrap().trim();
    
    // Find FROM keyword
    let parts: Vec<&str> = rest.splitn(2, " FROM ").collect();
    if parts.len() != 2 {
        return Err(DslError::Parse {
            line: line_no,
            msg: "Expected 'FROM' keyword in LOAD DATASET command".to_string(),
        });
    }
    
    let dataset_name = parts[0].trim();
    let path = parts[1].trim().trim_matches('"');
    
    // Load from storage
    let storage = ParquetStorage::new(path);
    let dataset = storage.load_dataset(dataset_name)
        .map_err(|e| DslError::Parse {
            line: line_no,
            msg: format!("Failed to load dataset: {}", e),
        })?;
    
    // Insert into DB
    // We explicitly insert the dataset. create_dataset usually takes name+schema.
    // But we have a full dataset. We need a way to insert a full dataset or insert it via crate::core::store
    // TensorDb has dataset_store field but it's private from here (handlers).
    // TensorDb has `create_dataset` (makes empty), `insert_row` (adds one by one).
    // We should probably add a `restore_dataset` method to TensorDb or use `dataset_store` if we expose it?
    // Let's check TensorDb methods exposed.
    // Step 398 shows:
    // dataset_store is private.
    // create_dataset(name, schema) -> Result<DatasetId>
    // We can iterate and insert rows, but that's slow for bulk load.
    // Ideally we add `import_dataset` to TensorDb or similar.
    
    // For now, let's assume we add `import_dataset` to TensorDb or similar.
    // Or we iterate. Iterating is fine for MVP.
    
    let schema = dataset.schema.clone();
    // Create new dataset in DB (this registers it)
    match db.create_dataset(dataset_name.to_string(), schema) {
        Ok(_) => {},
        Err(crate::engine::EngineError::DatasetError(
            crate::core::store::DatasetStoreError::NameAlreadyExists(_)
        )) => {
            // Option: Overwrite? Or Error?
            // "LOAD" usually implies bringing it in. If it exists, maybe we should error or drop first.
            return Err(DslError::Engine {
                line: line_no,
                source: crate::engine::EngineError::DatasetError(
                    crate::core::store::DatasetStoreError::NameAlreadyExists(dataset_name.to_string())
                ),
            });
        }
        Err(e) => return Err(DslError::Engine { line: line_no, source: e }),
    }
    
    let row_count = dataset.len();
    
    // Insert rows
    for row in dataset.rows {
        db.insert_row(dataset_name, row)
            .map_err(|e| DslError::Engine { line: line_no, source: e })?;
    }
    
    Ok(DslOutput::Message(format!(
        "Loaded dataset '{}' from '{}' ({} rows)",
        dataset_name, path, row_count
    )))
}

fn handle_load_tensor(db: &mut TensorDb, rest: &str, line_no: usize) -> Result<DslOutput, DslError> {
    let rest = rest.strip_prefix("TENSOR ").unwrap().trim();
    
    // Find FROM keyword
    let parts: Vec<&str> = rest.splitn(2, " FROM ").collect();
    if parts.len() != 2 {
        return Err(DslError::Parse {
            line: line_no,
            msg: "Expected 'FROM' keyword in LOAD TENSOR command".to_string(),
        });
    }
    
    let tensor_name = parts[0].trim();
    let path = parts[1].trim().trim_matches('"');
    
    // Load using storage engine
    let storage = ParquetStorage::new(path);
    let tensor = storage.load_tensor(tensor_name)
        .map_err(|e| DslError::Parse {
            line: line_no,
            msg: format!("Failed to load tensor: {}", e),
        })?;
    
    // Insert into db
    // We create a new tensor with a new ID in the current DB, but reuse shape and data
    db.insert_named(tensor_name, tensor.shape.clone(), tensor.data.clone())
        .map_err(|e| DslError::Engine {
            line: line_no,
            source: e,
        })?;
    
    Ok(DslOutput::Message(format!(
        "Loaded tensor '{}' from '{}'",
        tensor_name, path
    )))
}

/// Handle LIST DATASETS command
/// Syntax: LIST DATASETS FROM "path"
///         LIST TENSORS FROM "path"
pub fn handle_list_datasets(db: &mut TensorDb, line: &str, line_no: usize) -> Result<DslOutput, DslError> {
    let rest = line.strip_prefix("LIST ").unwrap().trim();
    
    if rest.starts_with("DATASETS") {
        handle_list_datasets_impl(db, rest, line_no)
    } else if rest.starts_with("TENSORS") {
        handle_list_tensors_impl(db, rest, line_no)
    } else {
        Err(DslError::Parse {
            line: line_no,
            msg: "Expected 'DATASETS' or 'TENSORS' after 'LIST'".to_string(),
        })
    }
}

fn handle_list_datasets_impl(_db: &mut TensorDb, rest: &str, line_no: usize) -> Result<DslOutput, DslError> {
    let rest = rest.strip_prefix("DATASETS").unwrap().trim();
    
    if !rest.starts_with("FROM ") {
        return Err(DslError::Parse {
            line: line_no,
            msg: "Expected 'FROM' keyword in LIST DATASETS command".to_string(),
        });
    }
    
    let path = rest.strip_prefix("FROM ").unwrap().trim().trim_matches('"');
    
    let storage = ParquetStorage::new(path);
    let datasets = storage.list_datasets()
        .map_err(|e| DslError::Parse {
            line: line_no,
            msg: format!("Failed to list datasets: {}", e),
        })?;
    
    let message = if datasets.is_empty() {
        format!("No datasets found in '{}'", path)
    } else {
        format!("Datasets in '{}':\n  - {}", path, datasets.join("\n  - "))
    };
    
    Ok(DslOutput::Message(message))
}

fn handle_list_tensors_impl(_db: &mut TensorDb, rest: &str, line_no: usize) -> Result<DslOutput, DslError> {
    let rest = rest.strip_prefix("TENSORS").unwrap().trim();
    
    if !rest.starts_with("FROM ") {
        return Err(DslError::Parse {
            line: line_no,
            msg: "Expected 'FROM' keyword in LIST TENSORS command".to_string(),
        });
    }
    
    let path = rest.strip_prefix("FROM ").unwrap().trim().trim_matches('"');
    
    let storage = ParquetStorage::new(path);
    let tensors = storage.list_tensors()
        .map_err(|e| DslError::Parse {
            line: line_no,
            msg: format!("Failed to list tensors: {}", e),
        })?;
    
    let message = if tensors.is_empty() {
        format!("No tensors found in '{}'", path)
    } else {
        format!("Tensors in '{}':\n  - {}", path, tensors.join("\n  - "))
    };
    
    Ok(DslOutput::Message(message))
}
