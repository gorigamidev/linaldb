use crate::core::connectors::{
    csv_connector::CsvConnector, hdf5_connector::Hdf5Connector, numpy_connector::NumpyConnector,
    zarr_connector::ZarrConnector, ConnectorRegistry,
};
use crate::core::dataset::{Dataset, DatasetMetadata, DatasetOrigin, ResourceReference};
use crate::core::storage::{record_batch_to_tensors, CsvStorage, ParquetStorage, StorageEngine};
use crate::dsl::{DslError, DslOutput};
use crate::engine::TensorDb;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};

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

/// Helper to resolve relative paths to data_dir/db_name/path
fn resolve_persistence_path(db: &TensorDb, path: &str) -> String {
    let path_buf = PathBuf::from(path);
    if path_buf.is_absolute() {
        return path.to_string();
    }

    // If it's a simple filename or a relative path, put it in data_dir/db_name/
    let mut resolved = db.config.storage.data_dir.clone();
    resolved.push(&db.active_instance().name);

    if !path.is_empty() {
        resolved.push(path);
    }

    // Ensure parent exists
    if let Some(parent) = resolved.parent() {
        let _ = fs::create_dir_all(parent);
    }

    resolved.to_string_lossy().into_owned()
}

fn handle_save_dataset(
    db: &mut TensorDb,
    rest: &str,
    line_no: usize,
) -> Result<DslOutput, DslError> {
    let rest = rest.strip_prefix("DATASET ").unwrap().trim();

    // Check for " TO " keyword
    let (dataset_name, disk_name, path) = if let Some(idx) = rest.find(" TO ") {
        let name = rest[..idx].trim();
        let p_str = rest[idx + 4..].trim().trim_matches('"');
        let p_path = Path::new(p_str);

        if p_path.extension().is_some() {
            // File-like path: "path/to/file.parquet"
            let disk_name = p_path
                .file_stem()
                .unwrap_or_default()
                .to_str()
                .unwrap_or(name);
            let parent_str = p_path
                .parent()
                .map(|p| p.to_str().unwrap_or(""))
                .unwrap_or("");
            (name, disk_name, resolve_persistence_path(db, parent_str))
        } else {
            // Directory-like path: "path/to/dir"
            (name, name, resolve_persistence_path(db, p_str))
        }
    } else {
        (rest, rest, resolve_persistence_path(db, ""))
    };

    // Get dataset from store using public method
    let mut dataset = match db.get_dataset(dataset_name) {
        Ok(ds) => ds.clone(),
        Err(_) => db
            .materialize_tensor_dataset(dataset_name)
            .map_err(|e| DslError::Engine {
                line: line_no,
                source: e,
            })?,
    };

    // Temporarily rename dataset to disk name for saving if they differ
    if disk_name != dataset_name {
        dataset.metadata.name = Some(disk_name.to_string());
    }

    // Save using storage engine
    let storage = ParquetStorage::new(&path);
    storage
        .save_dataset(&dataset)
        .map_err(|e| DslError::Parse {
            line: line_no,
            msg: format!("Failed to save dataset: {}", e),
        })?;

    // Create or update metadata
    // Use disk_name for metadata file to match parquet file
    let mut metadata = if storage.metadata_exists(disk_name) {
        // Load existing metadata and increment version
        let mut meta = storage
            .load_dataset_metadata(disk_name)
            .unwrap_or_else(|_| {
                DatasetMetadata::new(disk_name.to_string(), DatasetOrigin::Created)
            });
        meta.increment_version();
        meta
    } else {
        // Create new metadata
        DatasetMetadata::new(disk_name.to_string(), DatasetOrigin::Created)
    };

    // Compute content hash (simple hash of dataset name + row count for now)
    let content_hash = format!("{}:{}", dataset_name, dataset.rows.len());
    metadata.update_hash(content_hash);

    // Record schema in history
    metadata.record_schema(dataset.schema.as_ref().clone().into());

    // Save metadata
    storage
        .save_dataset_metadata(&metadata)
        .map_err(|e| DslError::Parse {
            line: line_no,
            msg: format!("Failed to save metadata: {}", e),
        })?;

    Ok(DslOutput::Message(format!(
        "Saved dataset '{}' (v{}) to '{}'",
        dataset_name, metadata.version, path
    )))
}

fn handle_save_tensor(
    db: &mut TensorDb,
    rest: &str,
    line_no: usize,
) -> Result<DslOutput, DslError> {
    let rest = rest.strip_prefix("TENSOR ").unwrap().trim();

    // Check for " TO " keyword
    let (tensor_name, path) = if let Some(idx) = rest.find(" TO ") {
        let name = rest[..idx].trim();
        let p = rest[idx + 4..].trim().trim_matches('"');
        (name, resolve_persistence_path(db, p))
    } else {
        let p = resolve_persistence_path(db, "");
        (rest, p)
    };

    // Get tensor from db
    let tensor = db
        .active_instance()
        .get(tensor_name)
        .map_err(|e| DslError::Engine {
            line: line_no,
            source: e,
        })?;

    // Save using storage engine
    let storage = ParquetStorage::new(&path);
    storage
        .save_tensor(tensor_name, tensor)
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

fn handle_load_dataset(
    db: &mut TensorDb,
    rest: &str,
    line_no: usize,
) -> Result<DslOutput, DslError> {
    let rest = rest.strip_prefix("DATASET ").unwrap().trim();

    // Check for " FROM " keyword
    let (dataset_name, disk_name, path) = if rest.contains(" FROM ") {
        let parts: Vec<&str> = rest.splitn(2, " FROM ").collect();
        let name = parts[0].trim();
        let p_str = parts[1].trim().trim_matches('"');
        let p_path = Path::new(p_str);

        if p_path.extension().is_some() {
            // File-like path
            let disk_name = p_path
                .file_stem()
                .unwrap_or_default()
                .to_str()
                .unwrap_or(p_str);
            let parent_str = p_path
                .parent()
                .map(|p| p.to_str().unwrap_or(""))
                .unwrap_or("");
            (name, disk_name, resolve_persistence_path(db, parent_str))
        } else {
            // Directory-like path
            (name, name, resolve_persistence_path(db, p_str))
        }
    } else {
        let p = resolve_persistence_path(db, "");
        (rest, rest, p)
    };

    // 1. Try loading as a reference-based dataset
    let storage = ParquetStorage::new(&path);
    if let Ok(mut dataset) = storage.load_reference_dataset(disk_name) {
        // Try to load metadata if it exists
        let metadata_info = if storage.metadata_exists(disk_name) {
            if let Ok(meta) = storage.load_dataset_metadata(disk_name) {
                let info = format!(
                    " (v{}, {})",
                    meta.version,
                    match meta.origin {
                        crate::core::dataset::DatasetOrigin::Created => "Created",
                        crate::core::dataset::DatasetOrigin::Imported { .. } => "Imported",
                        crate::core::dataset::DatasetOrigin::Derived { .. } => "Derived",
                        crate::core::dataset::DatasetOrigin::Bound { .. } => "Bound",
                        crate::core::dataset::DatasetOrigin::Attached { .. } => "Attached",
                    }
                );
                dataset.metadata = Some(meta);
                info
            } else {
                String::new()
            }
        } else {
            String::new()
        };

        // Rename if target name is different
        if dataset_name != disk_name {
            dataset.name = dataset_name.to_string();
            if let Some(meta) = &mut dataset.metadata {
                meta.name = dataset_name.to_string();
            }
        }

        db.active_instance_mut().register_tensor_dataset(dataset);

        return Ok(DslOutput::Message(format!(
            "Loaded reference dataset '{}'{} from '{}'",
            dataset_name, metadata_info, path
        )));
    }

    // 2. Fallback to legacy dataset loading
    let mut dataset = storage
        .load_dataset(disk_name)
        .map_err(|e| DslError::Parse {
            line: line_no,
            msg: format!(
                "Failed to load dataset '{}' from '{}': {}",
                disk_name, path, e
            ),
        })?;

    // Rename if target name is different
    if dataset_name != disk_name {
        dataset.metadata.name = Some(dataset_name.to_string());
    }

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
        Ok(_) => {}
        Err(crate::engine::EngineError::DatasetError(
            crate::core::store::DatasetStoreError::NameAlreadyExists(_),
        )) => {
            // Option: Overwrite? Or Error?
            // "LOAD" usually implies bringing it in. If it exists, maybe we should error or drop first.
            return Err(DslError::Engine {
                line: line_no,
                source: crate::engine::EngineError::DatasetError(
                    crate::core::store::DatasetStoreError::NameAlreadyExists(
                        dataset_name.to_string(),
                    ),
                ),
            });
        }
        Err(e) => {
            return Err(DslError::Engine {
                line: line_no,
                source: e,
            })
        }
    }

    let row_count = dataset.len();

    // Insert rows
    for row in dataset.rows {
        db.insert_row(dataset_name, row)
            .map_err(|e| DslError::Engine {
                line: line_no,
                source: e,
            })?;
    }

    Ok(DslOutput::Message(format!(
        "Loaded dataset '{}' from '{}' ({} rows)",
        dataset_name, path, row_count
    )))
}

fn handle_load_tensor(
    db: &mut TensorDb,
    rest: &str,
    line_no: usize,
) -> Result<DslOutput, DslError> {
    let rest = rest.strip_prefix("TENSOR ").unwrap().trim();

    // Check for " FROM " keyword
    let (tensor_name, path) = if let Some(idx) = rest.find(" FROM ") {
        let name = rest[..idx].trim();
        let p = rest[idx + 6..].trim().trim_matches('"');
        (name, resolve_persistence_path(db, p))
    } else {
        let p = resolve_persistence_path(db, "");
        (rest, p)
    };

    // Load using storage engine
    let storage = ParquetStorage::new(&path);
    let tensor = storage
        .load_tensor(tensor_name)
        .map_err(|e| DslError::Parse {
            line: line_no,
            msg: format!("Failed to load tensor: {}", e),
        })?;

    // Insert into db
    // We preserve the loaded tensor (including its metadata/lineage)
    db.active_instance_mut()
        .insert_tensor_object(tensor_name, tensor)
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
pub fn handle_list_datasets(
    db: &TensorDb,
    line: &str,
    line_no: usize,
) -> Result<DslOutput, DslError> {
    let rest = line.strip_prefix("LIST ").unwrap().trim();

    if rest.starts_with("DATASETS") {
        handle_list_datasets_impl(db, rest, line_no)
    } else if rest.starts_with("TENSORS") {
        handle_list_tensors_impl(db, rest, line_no)
    } else if rest.starts_with("DATASET VERSIONS ") {
        handle_list_versions_impl(db, rest, line_no)
    } else {
        Err(DslError::Parse {
            line: line_no,
            msg: "Expected 'DATASETS', 'TENSORS', or 'DATASET VERSIONS' after 'LIST'".to_string(),
        })
    }
}

fn handle_list_versions_impl(
    db: &TensorDb,
    rest: &str,
    line_no: usize,
) -> Result<DslOutput, DslError> {
    // Syntax: DATASET VERSIONS <name>
    let dataset_name = rest.strip_prefix("DATASET VERSIONS ").unwrap().trim();

    let path = format!(
        "{}/{}",
        db.config.storage.data_dir.to_string_lossy(),
        db.active_instance().name
    );
    let storage = ParquetStorage::new(&path);

    if !storage.metadata_exists(dataset_name) {
        return Ok(DslOutput::Message(format!(
            "No metadata found for dataset '{}'",
            dataset_name
        )));
    }

    let metadata = storage
        .load_dataset_metadata(dataset_name)
        .map_err(|e| DslError::Parse {
            line: line_no,
            msg: format!("Failed to load metadata: {}", e),
        })?;

    let mut output = format!("=== Version History for Dataset: {} ===\n", dataset_name);
    output.push_str(&format!("Current Version: {}\n", metadata.version));
    output.push_str(&format!(
        "Current Schema Version: {}\n",
        metadata.schema_version
    ));
    output.push_str("\nSchema History:\n");

    if metadata.schema_history.is_empty() {
        output.push_str("  (Initial schema only)\n");
    } else {
        for v in &metadata.schema_history {
            output.push_str(&format!(
                "  - v{}: {} columns, migration: {:?}\n",
                v.version,
                v.schema.columns.len(),
                v.migration
            ));
        }
    }
    output.push_str("================================");

    Ok(DslOutput::Message(output))
}

fn handle_list_datasets_impl(
    db: &TensorDb,
    rest: &str,
    line_no: usize,
) -> Result<DslOutput, DslError> {
    let rest = rest.strip_prefix("DATASETS").unwrap().trim();

    let path = if rest.starts_with("FROM ") {
        let p = rest.strip_prefix("FROM ").unwrap().trim().trim_matches('"');
        resolve_persistence_path(db, p)
    } else {
        resolve_persistence_path(db, "")
    };

    let storage = ParquetStorage::new(&path);
    let datasets = storage.list_datasets().map_err(|e| DslError::Parse {
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

fn handle_list_tensors_impl(
    db: &TensorDb,
    rest: &str,
    line_no: usize,
) -> Result<DslOutput, DslError> {
    let rest = rest.strip_prefix("TENSORS").unwrap().trim();

    let path = if rest.starts_with("FROM ") {
        let p = rest.strip_prefix("FROM ").unwrap().trim().trim_matches('"');
        resolve_persistence_path(db, p)
    } else {
        resolve_persistence_path(db, "")
    };

    let storage = ParquetStorage::new(&path);
    let tensors = storage.list_tensors().map_err(|e| DslError::Parse {
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

/// Helper to get a connector registry with default connectors
pub fn get_connector_registry() -> ConnectorRegistry {
    let mut registry = ConnectorRegistry::new();
    registry.register(Box::new(CsvConnector::new()));
    registry.register(Box::new(NumpyConnector));
    registry.register(Box::new(Hdf5Connector));
    registry.register(Box::new(ZarrConnector));
    registry
}

pub fn handle_use_dataset(
    db: &mut TensorDb,
    line: &str,
    line_no: usize,
) -> Result<DslOutput, DslError> {
    let rest = line.strip_prefix("USE DATASET FROM ").unwrap().trim();

    // Parse: "path" [AS dataset_name]
    let (path_str, name_override) = if let Some(as_idx) = rest.find(" AS ") {
        let path = rest[..as_idx].trim().trim_matches('"');
        let name = rest[as_idx + 4..].trim();
        (path, Some(name))
    } else {
        (rest.trim_matches('"'), None)
    };

    let registry = get_connector_registry();
    let connector = registry
        .find_connector(path_str)
        .ok_or_else(|| DslError::Parse {
            line: line_no,
            msg: format!("No connector found for path: {}", path_str),
        })?;

    let (batch, _lineage) = connector
        .read_dataset(path_str)
        .map_err(|e| DslError::Parse {
            line: line_no,
            msg: format!("Connector failed: {}", e),
        })?;

    let tensors = record_batch_to_tensors(&batch).map_err(|e| DslError::Parse {
        line: line_no,
        msg: format!("Failed to convert to tensors: {}", e),
    })?;

    let ds_name = name_override.unwrap_or_else(|| {
        Path::new(path_str)
            .file_stem()
            .and_then(OsStr::to_str)
            .unwrap_or("ephemeral_ds")
    });

    let mut ds = Dataset::new(ds_name);
    for (col_name, tensor) in tensors {
        let tensor_id = tensor.id;
        let tensor_shape = tensor.shape.clone();

        // In ephemeral mode, we might want to prefix these tensors to avoid collision
        // but for now we follow the "load into store" pattern.
        db.active_instance_mut()
            .insert_tensor_object(format!("{}_{}", ds_name, col_name), tensor)
            .map_err(|e| DslError::Engine {
                line: line_no,
                source: e,
            })?;

        let value_type = match tensor_shape.rank() {
            1 => crate::core::value::ValueType::Vector(tensor_shape.dims[0]),
            2 => crate::core::value::ValueType::Matrix(tensor_shape.dims[0], tensor_shape.dims[1]),
            0 => crate::core::value::ValueType::Float,
            _ => crate::core::value::ValueType::Vector(tensor_shape.num_elements()),
        };

        let schema =
            crate::core::dataset::ColumnSchema::new(col_name.clone(), value_type, tensor_shape);
        ds.add_column(col_name, ResourceReference::tensor(tensor_id), schema);
    }

    db.active_instance_mut().register_tensor_dataset(ds);

    Ok(DslOutput::Table(
        db.active_instance()
            .materialize_tensor_dataset(ds_name)
            .map_err(|e| DslError::Engine {
                line: line_no,
                source: e,
            })?,
    ))
}

pub fn handle_import_dataset(
    db: &mut TensorDb,
    line: &str,
    line_no: usize,
) -> Result<DslOutput, DslError> {
    let rest = line.strip_prefix("IMPORT DATASET FROM ").unwrap().trim();

    // Parse: "path" [AS dataset_name]
    let (path_str, name_override) = if let Some(as_idx) = rest.find(" AS ") {
        let path = rest[..as_idx].trim().trim_matches('"');
        let name = rest[as_idx + 4..].trim();
        (path, Some(name))
    } else {
        (rest.trim_matches('"'), None)
    };

    let registry = get_connector_registry();
    let connector = registry
        .find_connector(path_str)
        .ok_or_else(|| DslError::Parse {
            line: line_no,
            msg: format!("No connector found for path: {}", path_str),
        })?;

    let (batch, lineage) = connector
        .read_dataset(path_str)
        .map_err(|e| DslError::Parse {
            line: line_no,
            msg: format!("Connector failed: {}", e),
        })?;

    let ds_name = name_override.unwrap_or_else(|| {
        Path::new(path_str)
            .file_stem()
            .and_then(OsStr::to_str)
            .unwrap_or("imported_ds")
    });

    // Persistent storage
    let storage_path = resolve_persistence_path(db, "");
    let storage = ParquetStorage::new(&storage_path);

    let metadata = DatasetMetadata::new(
        ds_name.to_string(),
        DatasetOrigin::Imported {
            source: path_str.to_string(),
        },
    );

    storage
        .save_dataset_package(ds_name, &batch, &metadata, &lineage)
        .map_err(|e| DslError::Parse {
            line: line_no,
            msg: format!("Failed to save dataset package: {}", e),
        })?;

    Ok(DslOutput::Message(format!(
        "Imported dataset '{}' and persisted to {}",
        ds_name, storage_path
    )))
}

/// Handle IMPORT CSV command (Legacy)
pub fn handle_import_csv(
    db: &mut TensorDb,
    line: &str,
    line_no: usize,
) -> Result<DslOutput, DslError> {
    let rest = line.strip_prefix("IMPORT ").unwrap().trim();
    let rest = rest.strip_prefix("CSV ").unwrap().trim();

    // Parse: FROM "path" [AS dataset_name]
    let (path, dataset_name_override) = if rest.starts_with("FROM ") {
        let rest = rest.strip_prefix("FROM ").unwrap().trim();
        if let Some(as_idx) = rest.find(" AS ") {
            let path = rest[..as_idx].trim().trim_matches('"');
            let name = rest[as_idx + 4..].trim();
            (path, Some(name))
        } else {
            (rest.trim_matches('"'), None)
        }
    } else {
        return Err(DslError::Parse {
            line: line_no,
            msg: "Expected 'FROM \"path\"' in IMPORT CSV command".to_string(),
        });
    };

    let resolved_path = resolve_persistence_path(db, path);
    let csv_storage = CsvStorage::new(&resolved_path);

    let dataset = csv_storage
        .import_dataset(&resolved_path)
        .map_err(|e| DslError::Parse {
            line: line_no,
            msg: format!("Failed to import CSV: {}", e),
        })?;

    let final_name =
        dataset_name_override.unwrap_or(dataset.metadata.name.as_deref().unwrap_or("imported_csv"));

    // Register dataset in DB
    let schema = dataset.schema.clone();
    db.create_dataset(final_name.to_string(), schema)
        .map_err(|e| DslError::Engine {
            line: line_no,
            source: e,
        })?;

    let row_count = dataset.len();
    for row in dataset.rows {
        db.insert_row(final_name, row)
            .map_err(|e| DslError::Engine {
                line: line_no,
                source: e,
            })?;
    }

    Ok(DslOutput::Message(format!(
        "Imported {} rows from '{}' into dataset '{}'",
        row_count, path, final_name
    )))
}

/// Handle IMPORT command
/// Syntax: IMPORT CSV FROM "path" [AS dataset_name]
///         IMPORT DATASET FROM "path" [AS dataset_name]
pub fn handle_import(db: &mut TensorDb, line: &str, line_no: usize) -> Result<DslOutput, DslError> {
    let rest = line.strip_prefix("IMPORT ").unwrap().trim();

    if rest.starts_with("CSV ") {
        handle_import_csv(db, line, line_no)
    } else if rest.starts_with("DATASET FROM ") {
        handle_import_dataset(db, line, line_no)
    } else {
        Err(DslError::Parse {
            line: line_no,
            msg: "Expected 'CSV' or 'DATASET FROM' after 'IMPORT'".to_string(),
        })
    }
}

/// Handle EXPORT CSV command
/// Syntax: EXPORT CSV dataset_name TO "path"
pub fn handle_export(db: &mut TensorDb, line: &str, line_no: usize) -> Result<DslOutput, DslError> {
    let rest = line.strip_prefix("EXPORT ").unwrap().trim();

    if !rest.starts_with("CSV ") {
        return Err(DslError::Parse {
            line: line_no,
            msg: "Expected 'CSV' after 'EXPORT'".to_string(),
        });
    }

    let rest = rest.strip_prefix("CSV ").unwrap().trim();

    // Parse: dataset_name TO "path"
    let (dataset_name, path) = if let Some(idx) = rest.find(" TO ") {
        let name = rest[..idx].trim();
        let path = rest[idx + 4..].trim().trim_matches('"');
        (name, path)
    } else {
        return Err(DslError::Parse {
            line: line_no,
            msg: "Expected 'TO \"path\"' in EXPORT CSV command".to_string(),
        });
    };

    let dataset = db.get_dataset(dataset_name).map_err(|e| DslError::Engine {
        line: line_no,
        source: e,
    })?;

    let resolved_path = resolve_persistence_path(db, path);
    let csv_storage = CsvStorage::new(&resolved_path);

    csv_storage
        .export_dataset(dataset, &resolved_path)
        .map_err(|e| DslError::Parse {
            line: line_no,
            msg: format!("Failed to export CSV: {}", e),
        })?;

    Ok(DslOutput::Message(format!(
        "Exported dataset '{}' to '{}'",
        dataset_name, path
    )))
}
