// src/dataset_store.rs

use crate::dataset::{Dataset, DatasetId};
use std::collections::HashMap;

/// Error types for dataset store operations
#[derive(Debug)]
pub enum DatasetStoreError {
    DatasetNotFound(DatasetId),
    NameAlreadyExists(String),
    InvalidDataset(String),
}

impl std::fmt::Display for DatasetStoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DatasetStoreError::DatasetNotFound(id) => write!(f, "Dataset not found: {:?}", id),
            DatasetStoreError::NameAlreadyExists(name) => {
                write!(f, "Dataset name already exists: {}", name)
            }
            DatasetStoreError::InvalidDataset(msg) => write!(f, "Invalid dataset: {}", msg),
        }
    }
}

impl std::error::Error for DatasetStoreError {}

/// In-memory storage for datasets
#[derive(Debug)]
pub struct DatasetStore {
    next_id: u64,
    datasets: HashMap<DatasetId, Dataset>,
    names: HashMap<String, DatasetId>,
}

impl DatasetStore {
    pub fn new() -> Self {
        Self {
            next_id: 0,
            datasets: HashMap::new(),
            names: HashMap::new(),
        }
    }

    /// Generate a new dataset ID
    pub fn gen_id(&mut self) -> DatasetId {
        let id = DatasetId(self.next_id);
        self.next_id += 1;
        id
    }

    /// Insert a dataset with an optional name
    pub fn insert(
        &mut self,
        mut dataset: Dataset,
        name: Option<String>,
    ) -> Result<DatasetId, DatasetStoreError> {
        let id = dataset.id;

        // Check if name already exists
        if let Some(ref name_str) = name {
            if self.names.contains_key(name_str) {
                return Err(DatasetStoreError::NameAlreadyExists(name_str.clone()));
            }
        }

        // Update dataset metadata name
        dataset.metadata.name = name.clone();

        // Insert dataset
        self.datasets.insert(id, dataset);

        // Register name if provided
        if let Some(name_str) = name {
            self.names.insert(name_str, id);
        }

        Ok(id)
    }

    /// Get a dataset by ID
    pub fn get(&self, id: DatasetId) -> Result<&Dataset, DatasetStoreError> {
        self.datasets
            .get(&id)
            .ok_or(DatasetStoreError::DatasetNotFound(id))
    }

    /// Get a mutable reference to a dataset by ID
    pub fn get_mut(&mut self, id: DatasetId) -> Result<&mut Dataset, DatasetStoreError> {
        self.datasets
            .get_mut(&id)
            .ok_or(DatasetStoreError::DatasetNotFound(id))
    }

    /// Get a dataset by name
    pub fn get_by_name(&self, name: &str) -> Result<&Dataset, DatasetStoreError> {
        let id = self.names.get(name).ok_or_else(|| {
            DatasetStoreError::InvalidDataset(format!("Dataset '{}' not found", name))
        })?;
        self.get(*id)
    }

    /// Get a mutable reference to a dataset by name
    pub fn get_mut_by_name(&mut self, name: &str) -> Result<&mut Dataset, DatasetStoreError> {
        let id = self.names.get(name).copied().ok_or_else(|| {
            DatasetStoreError::InvalidDataset(format!("Dataset '{}' not found", name))
        })?;
        self.get_mut(id)
    }

    /// Remove a dataset by ID
    pub fn remove(&mut self, id: DatasetId) -> Result<Dataset, DatasetStoreError> {
        let dataset = self
            .datasets
            .remove(&id)
            .ok_or(DatasetStoreError::DatasetNotFound(id))?;

        // Remove name mapping if it exists
        if let Some(ref name) = dataset.metadata.name {
            self.names.remove(name);
        }

        Ok(dataset)
    }

    /// Remove a dataset by name
    pub fn remove_by_name(&mut self, name: &str) -> Result<Dataset, DatasetStoreError> {
        let id = self.names.remove(name).ok_or_else(|| {
            DatasetStoreError::InvalidDataset(format!("Dataset '{}' not found", name))
        })?;

        self.datasets
            .remove(&id)
            .ok_or(DatasetStoreError::DatasetNotFound(id))
    }

    /// List all dataset IDs
    pub fn list_ids(&self) -> Vec<DatasetId> {
        self.datasets.keys().copied().collect()
    }

    /// List all dataset names
    pub fn list_names(&self) -> Vec<String> {
        self.names.keys().cloned().collect()
    }

    /// Get the number of datasets
    pub fn len(&self) -> usize {
        self.datasets.len()
    }

    pub fn is_empty(&self) -> bool {
        self.datasets.is_empty()
    }
}

impl Default for DatasetStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tuple::{Field, Schema};
    use crate::value::ValueType;
    use std::sync::Arc;

    fn create_test_dataset(id: DatasetId) -> Dataset {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", ValueType::Int),
            Field::new("name", ValueType::String),
        ]));

        Dataset::new(id, schema, None)
    }

    #[test]
    fn test_dataset_store_insert_and_get() {
        let mut store = DatasetStore::new();
        let id = store.gen_id();
        let dataset = create_test_dataset(id);

        store.insert(dataset, Some("test".to_string())).unwrap();

        assert_eq!(store.len(), 1);
        assert!(store.get(id).is_ok());
        assert!(store.get_by_name("test").is_ok());
    }

    #[test]
    fn test_dataset_store_duplicate_name() {
        let mut store = DatasetStore::new();

        let id1 = store.gen_id();
        let dataset1 = create_test_dataset(id1);
        store.insert(dataset1, Some("test".to_string())).unwrap();

        let id2 = store.gen_id();
        let dataset2 = create_test_dataset(id2);
        let result = store.insert(dataset2, Some("test".to_string()));

        assert!(result.is_err());
    }

    #[test]
    fn test_dataset_store_remove() {
        let mut store = DatasetStore::new();
        let id = store.gen_id();
        let dataset = create_test_dataset(id);

        store.insert(dataset, Some("test".to_string())).unwrap();
        assert_eq!(store.len(), 1);

        let removed = store.remove_by_name("test").unwrap();
        assert_eq!(removed.id, id);
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn test_dataset_store_list() {
        let mut store = DatasetStore::new();

        let id1 = store.gen_id();
        let dataset1 = create_test_dataset(id1);
        store
            .insert(dataset1, Some("dataset1".to_string()))
            .unwrap();

        let id2 = store.gen_id();
        let dataset2 = create_test_dataset(id2);
        store
            .insert(dataset2, Some("dataset2".to_string()))
            .unwrap();

        assert_eq!(store.len(), 2);
        assert_eq!(store.list_names().len(), 2);
        assert!(store.list_names().contains(&"dataset1".to_string()));
        assert!(store.list_names().contains(&"dataset2".to_string()));
    }
}
