use super::{Index, IndexType};
use crate::core::tensor::Tensor;
use crate::core::value::Value;

/// An index for vector similarity search
/// MVP: Brute-force linear scan
#[derive(Debug)]
pub struct VectorIndex {
    /// List of (row_id, embedding_tensor)
    vectors: Vec<(usize, Tensor)>,
}

impl VectorIndex {
    pub fn new() -> Self {
        Self {
            vectors: Vec::new(),
        }
    }

    /// Calculate cosine similarity between two tensors
    fn cosine_similarity(t1: &Tensor, t2: &Tensor) -> Result<f32, String> {
        if t1.shape != t2.shape {
            return Err(format!("Shape mismatch: {:?} vs {:?}", t1.shape, t2.shape));
        }

        if t1.data.len() != t2.data.len() {
            return Err("Data length mismatch".to_string());
        }

        let dot_product: f32 = t1.data.iter().zip(&t2.data).map(|(a, b)| a * b).sum();
        let norm_t1: f32 = t1.data.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_t2: f32 = t2.data.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_t1 == 0.0 || norm_t2 == 0.0 {
            return Ok(0.0); // Handle zero vectors
        }

        Ok(dot_product / (norm_t1 * norm_t2))
    }
}

impl Index for VectorIndex {
    fn add(&mut self, row_id: usize, value: &Value) -> Result<(), String> {
        match value {
            Value::Vector(data) => {
                // Convert Vec<f32> to Tensor (MVP: Shape is inferred as [len])
                use crate::core::tensor::{Shape, Tensor, TensorId};
                let tensor = Tensor::new(TensorId(0), Shape::new(vec![data.len()]), data.clone())
                    .map_err(|e| e.to_string())?;

                self.vectors.push((row_id, tensor));
                Ok(())
            }
            Value::Bool(_) => Err("Cannot index Boolean as Vector".to_string()),
            Value::Int(_) => Err("Cannot index Int as Vector".to_string()),
            Value::String(_) => Err("Cannot index String as Vector".to_string()),
            Value::Null => Ok(()),
            Value::Float(_) => Err("Cannot index Float as Vector".to_string()),
            Value::Matrix(_) => Err("Cannot index Matrix as Vector".to_string()),
        }
    }

    fn lookup(&self, _value: &Value) -> Result<Vec<usize>, String> {
        Err("VectorIndex does not support exact value lookup".to_string())
    }

    fn search(&self, query: &Tensor, k: usize) -> Result<Vec<(usize, f32)>, String> {
        let mut scores = Vec::with_capacity(self.vectors.len());

        for (row_id, vec) in &self.vectors {
            let score = Self::cosine_similarity(query, vec)?;
            scores.push((*row_id, score));
        }

        // Sort by score descending
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top k
        Ok(scores.into_iter().take(k).collect())
    }

    fn index_type(&self) -> IndexType {
        IndexType::Vector
    }

    fn box_clone(&self) -> Box<dyn Index> {
        Box::new(Self {
            vectors: self.vectors.clone(),
        })
    }
}
