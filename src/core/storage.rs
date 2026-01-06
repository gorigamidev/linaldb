use crate::core::dataset_legacy::{Dataset, DatasetMetadata};
use crate::core::tensor::Tensor;
use crate::core::tuple::{Schema, Tuple};
use crate::core::value::{Value, ValueType};
use arrow::array::{
    Array, ArrayRef, BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array, StringArray,
};
use arrow::datatypes::{DataType, Field as ArrowField, Schema as ArrowSchema};
use arrow::record_batch::RecordBatch;
use arrow_csv as csv;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::arrow_writer::ArrowWriter;
use parquet::file::properties::WriterProperties;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum StorageError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Parquet error: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),

    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),

    #[error("Dataset not found: {0}")]
    DatasetNotFound(String),

    #[error("Tensor not found: {0}")]
    TensorNotFound(String),
}

/// Storage engine trait for persisting datasets and tensors
pub trait StorageEngine {
    /// Save a dataset to storage
    fn save_dataset(&self, dataset: &Dataset) -> Result<(), StorageError>;

    /// Load a dataset from storage
    fn load_dataset(&self, name: &str) -> Result<Dataset, StorageError>;

    /// Save a reference-based dataset
    fn save_reference_dataset(
        &self,
        dataset: &crate::core::dataset::Dataset,
    ) -> Result<(), StorageError>;

    /// Load a reference-based dataset
    fn load_reference_dataset(
        &self,
        name: &str,
    ) -> Result<crate::core::dataset::Dataset, StorageError>;

    /// Check if a dataset exists
    fn dataset_exists(&self, name: &str) -> bool;

    /// Delete a dataset
    fn delete_dataset(&self, name: &str) -> Result<(), StorageError>;

    /// List all datasets
    fn list_datasets(&self) -> Result<Vec<String>, StorageError>;

    /// Save a tensor to storage
    fn save_tensor(&self, name: &str, tensor: &Tensor) -> Result<(), StorageError>;

    /// Load a tensor from storage
    fn load_tensor(&self, name: &str) -> Result<Tensor, StorageError>;

    /// Check if a tensor exists
    fn tensor_exists(&self, name: &str) -> bool;

    /// Delete a tensor
    fn delete_tensor(&self, name: &str) -> Result<(), StorageError>;

    /// List all tensors
    fn list_tensors(&self) -> Result<Vec<String>, StorageError>;
}

/// Parquet-based storage implementation
pub struct ParquetStorage {
    base_path: String,
}

impl ParquetStorage {
    pub fn new(base_path: impl Into<String>) -> Self {
        Self {
            base_path: base_path.into(),
        }
    }

    fn dataset_path(&self, name: &str) -> String {
        format!("{}/datasets/{}.parquet", self.base_path, name)
    }

    fn metadata_path(&self, name: &str) -> String {
        format!("{}/datasets/{}.meta.json", self.base_path, name)
    }

    fn persistent_metadata_path(&self, name: &str) -> String {
        format!("{}/datasets/{}.metadata.json", self.base_path, name)
    }

    fn tensor_path(&self, name: &str) -> String {
        format!("{}/tensors/{}.json", self.base_path, name)
    }

    fn ensure_directories(&self) -> Result<(), StorageError> {
        let datasets_dir = format!("{}/datasets", self.base_path);
        let tensors_dir = format!("{}/tensors", self.base_path);
        fs::create_dir_all(datasets_dir)?;
        fs::create_dir_all(tensors_dir)?;
        Ok(())
    }

    /// Save dataset metadata to JSON file
    pub fn save_dataset_metadata(
        &self,
        metadata: &crate::core::dataset::DatasetMetadata,
    ) -> Result<(), StorageError> {
        self.ensure_directories()?;
        let path = self.persistent_metadata_path(&metadata.name);
        let json = serde_json::to_string_pretty(metadata).map_err(|e| {
            StorageError::Serialization(format!("Failed to serialize metadata: {}", e))
        })?;
        fs::write(path, json)?;
        Ok(())
    }

    /// Load dataset metadata from JSON file
    pub fn load_dataset_metadata(
        &self,
        name: &str,
    ) -> Result<crate::core::dataset::DatasetMetadata, StorageError> {
        let path = self.persistent_metadata_path(name);
        if !Path::new(&path).exists() {
            return Err(StorageError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Metadata file not found: {}", path),
            )));
        }
        let json = fs::read_to_string(path)?;
        let metadata = serde_json::from_str(&json).map_err(|e| {
            StorageError::Serialization(format!("Failed to deserialize metadata: {}", e))
        })?;
        Ok(metadata)
    }

    /// Check if metadata exists for a dataset
    pub fn metadata_exists(&self, name: &str) -> bool {
        Path::new(&self.persistent_metadata_path(name)).exists()
    }
}

/// Convert Arrow RecordBatch to LINAL Rows
pub fn record_batch_to_rows(
    batch: &RecordBatch,
    schema: &Arc<Schema>,
) -> Result<Vec<Tuple>, StorageError> {
    let num_rows = batch.num_rows();
    let mut tuples = Vec::with_capacity(num_rows);

    let mut columns_data: Vec<Vec<Value>> = Vec::new();

    for field in &schema.fields {
        let arrow_col = batch.column_by_name(&field.name).ok_or_else(|| {
            StorageError::Serialization(format!("Column {} missing in Arrow batch", field.name))
        })?;

        let values = arrow_array_to_values(arrow_col, &field.value_type, num_rows)?;
        columns_data.push(values);
    }

    for i in 0..num_rows {
        let mut row_values = Vec::with_capacity(schema.fields.len());
        for col_idx in 0..schema.fields.len() {
            row_values.push(columns_data[col_idx][i].clone());
        }
        tuples.push(
            Tuple::new(schema.clone(), row_values).map_err(|e| StorageError::Serialization(e))?,
        );
    }

    Ok(tuples)
}

fn arrow_array_to_values(
    array: &ArrayRef,
    target_type: &ValueType,
    num_rows: usize,
) -> Result<Vec<Value>, StorageError> {
    match target_type {
        ValueType::Int => {
            if let Some(int_array) = array.as_any().downcast_ref::<Int64Array>() {
                Ok((0..num_rows)
                    .map(|i| {
                        if int_array.is_null(i) {
                            Value::Null
                        } else {
                            Value::Int(int_array.value(i))
                        }
                    })
                    .collect())
            } else if let Some(int_array) = array.as_any().downcast_ref::<Int32Array>() {
                Ok((0..num_rows)
                    .map(|i| {
                        if int_array.is_null(i) {
                            Value::Null
                        } else {
                            Value::Int(int_array.value(i) as i64)
                        }
                    })
                    .collect())
            } else {
                Err(StorageError::Serialization(
                    "Expected Int64Array or Int32Array for Int type".to_string(),
                ))
            }
        }
        ValueType::Float => {
            if let Some(float_array) = array.as_any().downcast_ref::<Float32Array>() {
                Ok((0..num_rows)
                    .map(|i| {
                        if float_array.is_null(i) {
                            Value::Null
                        } else {
                            Value::Float(float_array.value(i))
                        }
                    })
                    .collect())
            } else if let Some(double_array) = array.as_any().downcast_ref::<Float64Array>() {
                Ok((0..num_rows)
                    .map(|i| {
                        if double_array.is_null(i) {
                            Value::Null
                        } else {
                            Value::Float(double_array.value(i) as f32)
                        }
                    })
                    .collect())
            } else {
                Err(StorageError::Serialization(
                    "Expected Float32Array or Float64Array for Float type".to_string(),
                ))
            }
        }
        ValueType::String => {
            let string_array = array
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| StorageError::Serialization("Expected StringArray".to_string()))?;
            Ok((0..num_rows)
                .map(|i| {
                    if string_array.is_null(i) {
                        Value::Null
                    } else {
                        Value::String(string_array.value(i).to_string())
                    }
                })
                .collect())
        }
        ValueType::Bool => {
            let bool_array = array
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| StorageError::Serialization("Expected BooleanArray".to_string()))?;
            Ok((0..num_rows)
                .map(|i| {
                    if bool_array.is_null(i) {
                        Value::Null
                    } else {
                        Value::Bool(bool_array.value(i))
                    }
                })
                .collect())
        }
        ValueType::Vector(_) | ValueType::Matrix(_, _) => {
            let string_array = array
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| {
                    StorageError::Serialization("Expected StringArray for complex type".to_string())
                })?;
            Ok((0..num_rows)
                .map(|i| {
                    if string_array.is_null(i) {
                        Value::Null
                    } else {
                        let json_str = string_array.value(i);
                        serde_json::from_str(json_str).unwrap_or(Value::Null)
                    }
                })
                .collect())
        }
        ValueType::Null => Ok(vec![Value::Null; num_rows]),
    }
}

/// Convert Dataset to Arrow RecordBatch
pub fn dataset_to_record_batch(dataset: &Dataset) -> Result<RecordBatch, StorageError> {
    // Build Arrow schema from dataset schema
    let arrow_fields: Vec<ArrowField> = dataset
        .schema
        .fields
        .iter()
        .map(|f| {
            let data_type = match &f.value_type {
                ValueType::Int => DataType::Int64,
                ValueType::Float => DataType::Float32,
                ValueType::String => DataType::Utf8,
                ValueType::Bool => DataType::Boolean,
                _ => DataType::Utf8, // Fallback for complex types (serialize as JSON string)
            };
            ArrowField::new(&f.name, data_type, f.nullable)
        })
        .collect();

    let arrow_schema = Arc::new(ArrowSchema::new(arrow_fields));

    // Convert rows to Arrow arrays
    let mut arrays: Vec<ArrayRef> = Vec::new();

    for field in &dataset.schema.fields {
        let column_data: Vec<&Value> = dataset
            .rows
            .iter()
            .map(|row| {
                row.values
                    .iter()
                    .zip(&row.schema.fields)
                    .find(|(_, f)| f.name == field.name)
                    .map(|(v, _)| v)
                    .unwrap_or(&Value::Null)
            })
            .collect();

        let array: ArrayRef = match &field.value_type {
            ValueType::Int => {
                let values: Vec<Option<i64>> = column_data
                    .iter()
                    .map(|v| match v {
                        Value::Int(i) => Some(*i),
                        Value::Null => None,
                        _ => None,
                    })
                    .collect();
                Arc::new(Int64Array::from(values))
            }
            ValueType::Float => {
                let values: Vec<Option<f32>> = column_data
                    .iter()
                    .map(|v| match v {
                        Value::Float(f) => Some(*f),
                        Value::Int(i) => Some(*i as f32),
                        Value::Null => None,
                        _ => None,
                    })
                    .collect();
                Arc::new(Float32Array::from(values))
            }
            ValueType::String => {
                let values: Vec<Option<&str>> = column_data
                    .iter()
                    .map(|v| match v {
                        Value::String(s) => Some(s.as_str()),
                        Value::Null => None,
                        _ => None,
                    })
                    .collect();
                Arc::new(StringArray::from(values))
            }
            ValueType::Bool => {
                let values: Vec<Option<bool>> = column_data
                    .iter()
                    .map(|v| match v {
                        Value::Bool(b) => Some(*b),
                        Value::Null => None,
                        _ => None,
                    })
                    .collect();
                Arc::new(BooleanArray::from(values))
            }
            _ => {
                // For complex types (Vector, Matrix), serialize as JSON strings
                let values: Vec<Option<String>> = column_data
                    .iter()
                    .map(|v| match v {
                        Value::Null => None,
                        v => Some(serde_json::to_string(v).unwrap_or_else(|_| "null".to_string())),
                    })
                    .collect();
                Arc::new(StringArray::from(values))
            }
        };

        arrays.push(array);
    }

    RecordBatch::try_new(arrow_schema, arrays).map_err(|e| StorageError::Arrow(e))
}

impl StorageEngine for ParquetStorage {
    fn save_dataset(&self, dataset: &Dataset) -> Result<(), StorageError> {
        self.ensure_directories()?;

        let dataset_name =
            dataset.metadata.name.as_ref().ok_or_else(|| {
                StorageError::Serialization("Dataset must have a name".to_string())
            })?;

        // Convert to RecordBatch
        let record_batch = dataset_to_record_batch(dataset)?;

        // Write to Parquet file
        let data_path = self.dataset_path(dataset_name);
        let file = fs::File::create(&data_path)?;
        let props = WriterProperties::builder().build();
        let mut writer = ArrowWriter::try_new(file, record_batch.schema(), Some(props))?;
        writer.write(&record_batch)?;
        writer.close()?;

        // Save metadata as JSON
        let meta_path = self.metadata_path(dataset_name);
        let metadata_json = serde_json::to_string_pretty(&dataset.metadata)
            .map_err(|e| StorageError::Serialization(e.to_string()))?;
        fs::write(&meta_path, metadata_json)?;

        Ok(())
    }

    fn load_dataset(&self, name: &str) -> Result<Dataset, StorageError> {
        let meta_path = self.metadata_path(name);
        if !Path::new(&meta_path).exists() {
            return Err(StorageError::DatasetNotFound(name.to_string()));
        }

        let metadata_json = fs::read_to_string(&meta_path)?;
        let metadata: DatasetMetadata = serde_json::from_str(&metadata_json)
            .map_err(|e| StorageError::Serialization(format!("Metadata error: {}", e)))?;

        let data_path = self.dataset_path(name);
        if !Path::new(&data_path).exists() {
            return Err(StorageError::DatasetNotFound(format!(
                "Data file missing for {}",
                name
            )));
        }

        let file = fs::File::open(&data_path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(|e| StorageError::Serialization(e.to_string()))?;
        let record_batch_reader = builder
            .with_batch_size(2048)
            .build()
            .map_err(|e| StorageError::Serialization(e.to_string()))?;

        let mut rows = Vec::new();
        let schema = Arc::new(metadata.schema.clone());

        for batch in record_batch_reader {
            let batch = batch?;
            let batch_rows = record_batch_to_rows(&batch, &schema)?;
            rows.extend(batch_rows);
        }

        let mut dataset = Dataset::new(
            crate::core::dataset_legacy::DatasetId(0),
            schema,
            Some(name.to_string()),
        );
        dataset.rows = rows;
        dataset.metadata = metadata;

        Ok(dataset)
    }

    fn save_reference_dataset(
        &self,
        dataset: &crate::core::dataset::Dataset,
    ) -> Result<(), StorageError> {
        self.ensure_directories()?;
        let path = format!("{}/datasets/{}.ref.json", self.base_path, dataset.name);
        let json = serde_json::to_string_pretty(dataset)
            .map_err(|e| StorageError::Serialization(e.to_string()))?;
        fs::write(path, json)?;
        Ok(())
    }

    fn load_reference_dataset(
        &self,
        name: &str,
    ) -> Result<crate::core::dataset::Dataset, StorageError> {
        let path = format!("{}/datasets/{}.ref.json", self.base_path, name);
        if !Path::new(&path).exists() {
            return Err(StorageError::DatasetNotFound(name.to_string()));
        }
        let json = fs::read_to_string(path)?;
        let dataset =
            serde_json::from_str(&json).map_err(|e| StorageError::Serialization(e.to_string()))?;
        Ok(dataset)
    }

    fn dataset_exists(&self, name: &str) -> bool {
        Path::new(&self.dataset_path(name)).exists()
    }

    fn delete_dataset(&self, name: &str) -> Result<(), StorageError> {
        let data_path = self.dataset_path(name);
        let meta_path = self.metadata_path(name);
        if Path::new(&data_path).exists() {
            fs::remove_file(&data_path)?;
        }
        if Path::new(&meta_path).exists() {
            fs::remove_file(&meta_path)?;
        }
        Ok(())
    }

    fn list_datasets(&self) -> Result<Vec<String>, StorageError> {
        self.ensure_directories()?;
        let datasets_dir = format!("{}/datasets", self.base_path);
        let mut datasets = Vec::new();
        for entry in fs::read_dir(&datasets_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("parquet") {
                if let Some(name) = path.file_stem().and_then(|s| s.to_str()) {
                    datasets.push(name.to_string());
                }
            }
        }
        Ok(datasets)
    }

    fn save_tensor(&self, name: &str, tensor: &Tensor) -> Result<(), StorageError> {
        self.ensure_directories()?;
        let tensor_path = self.tensor_path(name);
        let tensor_json = serde_json::to_string_pretty(tensor)
            .map_err(|e| StorageError::Serialization(e.to_string()))?;
        fs::write(&tensor_path, tensor_json)?;
        Ok(())
    }

    fn load_tensor(&self, name: &str) -> Result<Tensor, StorageError> {
        let tensor_path = self.tensor_path(name);
        if !Path::new(&tensor_path).exists() {
            return Err(StorageError::TensorNotFound(name.to_string()));
        }
        let tensor_json = fs::read_to_string(&tensor_path)?;
        let tensor: Tensor = serde_json::from_str(&tensor_json)
            .map_err(|e| StorageError::Serialization(e.to_string()))?;
        Ok(tensor)
    }

    fn tensor_exists(&self, name: &str) -> bool {
        Path::new(&self.tensor_path(name)).exists()
    }

    fn delete_tensor(&self, name: &str) -> Result<(), StorageError> {
        let tensor_path = self.tensor_path(name);
        if Path::new(&tensor_path).exists() {
            fs::remove_file(&tensor_path)?;
        }
        Ok(())
    }

    fn list_tensors(&self) -> Result<Vec<String>, StorageError> {
        self.ensure_directories()?;
        let tensors_dir = format!("{}/tensors", self.base_path);
        let mut tensors = Vec::new();
        for entry in fs::read_dir(&tensors_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                if let Some(name) = path.file_stem().and_then(|s| s.to_str()) {
                    tensors.push(name.to_string());
                }
            }
        }
        Ok(tensors)
    }
}

/// CSV-based storage implementation
pub struct CsvStorage {
    _path: String,
}

impl CsvStorage {
    pub fn new(path: impl Into<String>) -> Self {
        Self { _path: path.into() }
    }

    pub fn export_dataset(&self, dataset: &Dataset, path: &str) -> Result<(), StorageError> {
        let batch = dataset_to_record_batch(dataset)?;
        let file = fs::File::create(path)?;
        let mut writer = arrow::csv::Writer::new(file);
        writer.write(&batch)?;
        Ok(())
    }

    pub fn import_dataset(&self, path: &str) -> Result<Dataset, StorageError> {
        let file = fs::File::open(path)?;
        // Infer schema from CSV using Format
        let format = csv::reader::Format::default().with_header(true);
        let (arrow_schema, _) = format
            .infer_schema(file, Some(100))
            .map_err(|e| StorageError::Arrow(e))?;

        let arrow_schema_arc: Arc<ArrowSchema> = Arc::new(arrow_schema.clone());

        let builder = csv::ReaderBuilder::new(arrow_schema_arc).with_header(true);

        let file = fs::File::open(path)?;
        let mut csv_reader = builder.build(file)?;

        // Use filename as dataset name
        let name = Path::new(path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("imported_data")
            .to_string();

        let mut record_batches: Vec<RecordBatch> = Vec::new();
        while let Some(batch) = csv_reader.next() {
            record_batches.push(batch?);
        }

        if record_batches.is_empty() {
            return Err(StorageError::Serialization("CSV file is empty".to_string()));
        }

        // Combine arrow schema to LINAL schema
        let fields: Vec<crate::core::tuple::Field> = arrow_schema
            .fields()
            .iter()
            .map(|f: &Arc<ArrowField>| {
                let vt = match f.data_type() {
                    DataType::Int64 => ValueType::Int,
                    DataType::Float32 | DataType::Float64 => ValueType::Float,
                    DataType::Boolean => ValueType::Bool,
                    _ => ValueType::String,
                };
                crate::core::tuple::Field::new(f.name(), vt)
            })
            .collect();

        let schema = Arc::new(crate::core::tuple::Schema::new(fields));
        let mut rows = Vec::new();

        for batch in record_batches {
            rows.extend(record_batch_to_rows(&batch, &schema)?);
        }

        let mut dataset = Dataset::new(
            crate::core::dataset_legacy::DatasetId(0),
            schema,
            Some(name),
        );
        dataset.rows = rows;
        Ok(dataset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parquet_storage_basic() {
        let temp_dir = "/tmp/linal_test_storage";
        let storage = ParquetStorage::new(temp_dir);
        let _ = fs::remove_dir_all(temp_dir);
        storage.ensure_directories().unwrap();
        assert!(Path::new(&format!("{}/datasets", temp_dir)).exists());
        assert!(Path::new(&format!("{}/tensors", temp_dir)).exists());
        let _ = fs::remove_dir_all(temp_dir);
    }
}
