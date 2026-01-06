// src/store.rs

use crate::core::tensor::{Shape, Tensor, TensorId};

#[derive(Debug)]
pub enum StoreError {
    ShapeMismatch(String),
    TensorNotFound(TensorId),
    InvalidTensor(String),
}

impl std::fmt::Display for StoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StoreError::ShapeMismatch(msg) => write!(f, "Shape mismatch: {}", msg),
            StoreError::TensorNotFound(id) => write!(f, "Tensor not found: {:?}", id),
            StoreError::InvalidTensor(msg) => write!(f, "Invalid tensor: {}", msg),
        }
    }
}

impl std::error::Error for StoreError {}

/// Motor en memoria: guarda tensores en una lista.
#[derive(Debug)]
pub struct InMemoryTensorStore {
    tensors: Vec<Tensor>,
}

impl InMemoryTensorStore {
    pub fn new() -> Self {
        Self {
            tensors: Vec::new(),
        }
    }

    /// Genera un nuevo ID interno
    pub fn gen_id(&mut self) -> TensorId {
        TensorId::new()
    }

    /// Inserta un tensor a partir de shape + data
    pub fn insert_tensor(&mut self, shape: Shape, data: Vec<f32>) -> Result<TensorId, StoreError> {
        let id = self.gen_id();
        let metadata = crate::core::tensor::TensorMetadata::new(id, None);
        let tensor = Tensor::new(id, shape, data, metadata).map_err(StoreError::InvalidTensor)?;
        self.tensors.push(tensor);
        Ok(id)
    }

    /// Inserta un Tensor ya construido
    pub fn insert_existing_tensor(&mut self, tensor: Tensor) -> Result<TensorId, StoreError> {
        if tensor.data.len() != tensor.shape.num_elements() {
            return Err(StoreError::InvalidTensor(format!(
                "Tensor data length {} does not match shape {:?}",
                tensor.data.len(),
                tensor.shape.dims
            )));
        }

        let id = tensor.id;
        self.tensors.push(tensor);
        Ok(id)
    }

    pub fn get(&self, id: TensorId) -> Result<&Tensor, StoreError> {
        self.tensors
            .iter()
            .find(|t| t.id == id)
            .ok_or(StoreError::TensorNotFound(id))
    }

    /// Removes a tensor by ID. Returns true if it was found and removed.
    pub fn remove(&mut self, id: TensorId) -> bool {
        let len_before = self.tensors.len();
        self.tensors.retain(|t| t.id != id);
        self.tensors.len() < len_before
    }

    /// Clears the store
    pub fn clear(&mut self) {
        self.tensors.clear();
    }
}
