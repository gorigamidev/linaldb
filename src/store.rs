// src/store.rs

use crate::tensor::{Tensor, TensorId, Shape};

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
    next_id: u64,
    tensors: Vec<Tensor>,
}

impl InMemoryTensorStore {
    pub fn new() -> Self {
        Self {
            next_id: 0,
            tensors: Vec::new(),
        }
    }

    /// Genera un nuevo ID interno
    pub fn gen_id_internal(&mut self) -> TensorId {
        let id = TensorId(self.next_id);
        self.next_id += 1;
        id
    }

    /// Inserta un tensor a partir de shape + data
    pub fn insert_tensor(&mut self, shape: Shape, data: Vec<f32>) -> Result<TensorId, StoreError> {
        let id = self.gen_id_internal();
        let tensor = Tensor::new(id, shape, data).map_err(StoreError::InvalidTensor)?;
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

    /// Obtiene referencia a un tensor por ID
    pub fn get(&self, id: TensorId) -> Result<&Tensor, StoreError> {
        self.tensors
            .iter()
            .find(|t| t.id == id)
            .ok_or(StoreError::TensorNotFound(id))
    }
}
