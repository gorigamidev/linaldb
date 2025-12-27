// src/tensor.rs
use serde::{Deserialize, Serialize};

/// Identificador de tensor (newtype para no confundir con otros u64)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TensorId(pub u64);

/// Representa la forma (shape) de un tensor.
/// []        -> escalar (rank 0)
/// [3]       -> vector 3D (rank 1)
/// [2, 3]    -> matriz 2x3 (rank 2)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Shape {
    pub dims: Vec<usize>,
}

impl Shape {
    /// Crea un nuevo shape a partir de una lista de dimensiones
    pub fn new<D: Into<Vec<usize>>>(dims: D) -> Self {
        Self { dims: dims.into() }
    }

    /// Número de dimensiones (rank)
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Número total de elementos
    pub fn num_elements(&self) -> usize {
        self.dims.iter().product()
    }
}

/// Tensor denso de f32 con layout row-major
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tensor {
    pub id: TensorId,
    pub shape: Shape,
    /// Primary storage using Arc for zero-copy sharing
    pub data: std::sync::Arc<Vec<f32>>,
}

impl Tensor {
    /// Creates a new tensor, wrapping data in an Arc
    pub fn new(id: TensorId, shape: Shape, data: Vec<f32>) -> Result<Self, String> {
        let expected = shape.num_elements();
        if data.len() != expected {
            return Err(format!(
                "Data length {} does not match shape {:?} (expected {})",
                data.len(),
                shape.dims,
                expected
            ));
        }

        Ok(Self {
            id,
            shape,
            data: std::sync::Arc::new(data),
        })
    }

    /// Creates a tensor from shared data (zero-copy)
    pub fn from_shared(
        id: TensorId,
        shape: Shape,
        data: std::sync::Arc<Vec<f32>>,
    ) -> Result<Self, String> {
        let expected = shape.num_elements();
        if data.len() != expected {
            return Err(format!(
                "Data length {} does not match shape {:?} (expected {})",
                data.len(),
                shape.dims,
                expected
            ));
        }

        Ok(Self { id, shape, data })
    }

    /// Get reference to data slice
    pub fn data_ref(&self) -> &[f32] {
        &self.data
    }

    /// Get shared reference to data (zero-copy)
    pub fn share(&self) -> std::sync::Arc<Vec<f32>> {
        self.data.clone()
    }

    /// Get mutable reference to data (copy-on-write)
    pub fn data_mut(&mut self) -> &mut Vec<f32> {
        std::sync::Arc::make_mut(&mut self.data)
    }

    /// Rank del tensor (0 = escalar, 1 = vector, 2 = matriz...)
    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    /// Número de elementos
    pub fn len(&self) -> usize {
        self.data_ref().len()
    }

    pub fn is_empty(&self) -> bool {
        self.data_ref().is_empty()
    }
}
