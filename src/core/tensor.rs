// src/tensor.rs

/// Identificador de tensor (newtype para no confundir con otros u64)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorId(pub u64);

/// Representa la forma (shape) de un tensor.
/// []        -> escalar (rank 0)
/// [3]       -> vector 3D (rank 1)
/// [2, 3]    -> matriz 2x3 (rank 2)
#[derive(Debug, Clone, PartialEq, Eq)]
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
#[derive(Debug, Clone)]
pub struct Tensor {
    pub id: TensorId,
    pub shape: Shape,
    pub data: Vec<f32>,
}

impl Tensor {
    /// Crea un tensor verificando que data.len() coincide con shape.num_elements()
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
        Ok(Self { id, shape, data })
    }

    /// Rank del tensor (0 = escalar, 1 = vector, 2 = matriz...)
    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    /// Número de elementos
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}
