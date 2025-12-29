use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::sync::Arc;
use uuid::Uuid;

/// Unique identifier for a tensor
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TensorId(pub Uuid);

impl TensorId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

/// Unique identifier for an execution/query
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ExecutionId(pub Uuid);

impl ExecutionId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

/// Information about how a tensor was derived
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lineage {
    pub execution_id: ExecutionId,
    pub operation: String,
    pub inputs: Vec<TensorId>,
}

/// Metadata about a tensor
#[derive(Debug, Clone)]
pub struct TensorMetadata {
    pub id: TensorId,
    pub created_at: DateTime<Utc>,
    pub creator: Option<String>,
    pub lineage: Option<Lineage>,
    pub(crate) data_hash: std::sync::OnceLock<String>,
}

impl Serialize for TensorMetadata {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let has_hash = self.data_hash.get().is_some();
        let num_fields =
            3 + (if has_hash { 1 } else { 0 }) + (if self.lineage.is_some() { 1 } else { 0 });

        let mut state = serializer.serialize_struct("TensorMetadata", num_fields)?;
        state.serialize_field("id", &self.id)?;
        state.serialize_field("created_at", &self.created_at)?;
        state.serialize_field("creator", &self.creator)?;

        if let Some(lineage) = &self.lineage {
            state.serialize_field("lineage", lineage)?;
        }

        if let Some(hash) = self.data_hash.get() {
            state.serialize_field("data_hash", hash)?;
        }

        state.end()
    }
}

impl<'de> Deserialize<'de> for TensorMetadata {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct MetadataFields {
            id: TensorId,
            created_at: DateTime<Utc>,
            creator: Option<String>,
            lineage: Option<Lineage>,
            data_hash: Option<String>,
        }

        let fields = MetadataFields::deserialize(deserializer)?;
        let lock = std::sync::OnceLock::new();
        if let Some(hash) = fields.data_hash {
            let _ = lock.set(hash);
        }

        Ok(Self {
            id: fields.id,
            created_at: fields.created_at,
            creator: fields.creator,
            lineage: fields.lineage,
            data_hash: lock,
        })
    }
}

impl TensorMetadata {
    pub fn new(id: TensorId, creator: Option<String>) -> Self {
        Self {
            id,
            created_at: Utc::now(),
            creator,
            lineage: None,
            data_hash: std::sync::OnceLock::new(),
        }
    }

    pub fn with_lineage(mut self, lineage: Lineage) -> Self {
        self.lineage = Some(lineage);
        self
    }

    /// Internal method to compute hash if not already set.
    /// Usually called by Tensor::data_hash()
    pub(crate) fn compute_hash(&self, data: &[f32]) -> &str {
        self.data_hash.get_or_init(|| {
            let mut hasher = Sha256::new();
            // Convert f32 slice to byte slice for hashing
            // Safety: f32 has no padding bits and we are just reading bits for hashing
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    data.as_ptr() as *const u8,
                    data.len() * std::mem::size_of::<f32>(),
                )
            };
            hasher.update(bytes);
            format!("{:x}", hasher.finalize())
        })
    }
}

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
    pub data: Arc<Vec<f32>>,
    pub metadata: Arc<TensorMetadata>,
}

impl Tensor {
    /// Creates a new tensor, wrapping data in an Arc
    pub fn new(
        id: TensorId,
        shape: Shape,
        data: Vec<f32>,
        metadata: TensorMetadata,
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

        Ok(Self {
            id,
            shape,
            data: Arc::new(data),
            metadata: Arc::new(metadata),
        })
    }

    /// Creates a tensor from shared data (zero-copy)
    pub fn from_shared(
        id: TensorId,
        shape: Shape,
        data: Arc<Vec<f32>>,
        metadata: Arc<TensorMetadata>,
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

        Ok(Self {
            id,
            shape,
            data,
            metadata,
        })
    }

    /// Get reference to data slice
    pub fn data_ref(&self) -> &[f32] {
        &self.data
    }

    /// Get shared reference to data (zero-copy)
    pub fn share(&self) -> Arc<Vec<f32>> {
        self.data.clone()
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

    /// Get or compute the cryptographic hash of the tensor data.
    /// This is a lazy operation.
    pub fn data_hash(&self) -> &str {
        self.metadata.compute_hash(&self.data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_metadata_serialization() {
        let id = TensorId::new();
        let metadata = TensorMetadata::new(id, Some("test_creator".to_string()));
        let serialized = serde_json::to_string(&metadata).unwrap();

        // Initially, data_hash should not be present in serialized JSON
        assert!(!serialized.contains("data_hash"));

        // Compute hash
        let data = vec![1.0, 2.0, 3.0];
        metadata.compute_hash(&data);

        let serialized_with_hash = serde_json::to_string(&metadata).unwrap();
        assert!(serialized_with_hash.contains("data_hash"));
    }

    #[test]
    fn test_lazy_hashing() {
        let id = TensorId::new();
        let data = vec![1.0, 2.0, 3.0];
        let metadata = TensorMetadata::new(id, None);
        let tensor = Tensor::new(id, Shape::new(vec![3]), data, metadata).unwrap();

        // Hash should not be set yet
        assert!(tensor.metadata.data_hash.get().is_none());

        // Request hash
        let hash1 = tensor.data_hash();
        assert!(!hash1.is_empty());

        // Hash should be cached now
        assert!(tensor.metadata.data_hash.get().is_some());

        // Request again, should be the same instance (cached)
        let hash2 = tensor.data_hash();
        assert_eq!(hash1, hash2);
    }
}
