use std::collections::HashMap;
use std::sync::Arc;

use crate::core::dataset::{Dataset, DatasetId};
use crate::core::store::{DatasetStore, InMemoryTensorStore};
use crate::core::tensor::{Shape, Tensor, TensorId};
use crate::core::tuple::{Schema, Tuple};

use super::kernels::{
    add, add_relaxed, cosine_similarity_1d, distance_1d, divide, divide_relaxed, dot_1d, flatten,
    matmul, multiply, multiply_relaxed, normalize_1d, reshape, scalar_mul, sub, sub_relaxed,
    transpose,
};

use super::error::EngineError;
use super::operations::{BinaryOp, TensorKind, UnaryOp};

struct NameEntry {
    id: TensorId,
    kind: TensorKind,
}

/// DB de alto nivel que trabaja con nombres y delega en el store
pub struct TensorDb {
    store: InMemoryTensorStore,
    names: HashMap<String, NameEntry>,
    dataset_store: DatasetStore,
}

impl TensorDb {
    pub fn new() -> Self {
        Self {
            store: InMemoryTensorStore::new(),
            names: HashMap::new(),
            dataset_store: DatasetStore::new(),
        }
    }

    /// Inserta un tensor y lo asocia a un nombre (modo NORMAL por defecto)
    pub fn insert_named(
        &mut self,
        name: impl Into<String>,
        shape: Shape,
        data: Vec<f32>,
    ) -> Result<(), EngineError> {
        self.insert_named_with_kind(name, shape, data, TensorKind::Normal)
    }

    /// Inserta un tensor con un "kind" explícito (NORMAL o STRICT)
    pub fn insert_named_with_kind(
        &mut self,
        name: impl Into<String>,
        shape: Shape,
        data: Vec<f32>,
        kind: TensorKind,
    ) -> Result<(), EngineError> {
        let id = self.store.insert_tensor(shape, data)?;
        self.names.insert(name.into(), NameEntry { id, kind });
        Ok(())
    }

    /// Obtiene un tensor por nombre
    pub fn get(&self, name: &str) -> Result<&Tensor, EngineError> {
        let entry = self
            .names
            .get(name)
            .ok_or_else(|| EngineError::NameNotFound(name.to_string()))?;
        Ok(self.store.get(entry.id)?)
    }

    /// Obtiene (tensor, kind) por nombre (para decisiones de ejecución)
    pub(crate) fn get_with_kind(&self, name: &str) -> Result<(&Tensor, TensorKind), EngineError> {
        let entry = self
            .names
            .get(name)
            .ok_or_else(|| EngineError::NameNotFound(name.to_string()))?;
        let t = self.store.get(entry.id)?;
        Ok((t, entry.kind))
    }

    /// Evalúa operación unaria: SCALE, etc.
    pub fn eval_unary(
        &mut self,
        output_name: impl Into<String>,
        input_name: &str,
        op: UnaryOp,
    ) -> Result<(), EngineError> {
        let (in_tensor_ref, in_kind) = self.get_with_kind(input_name)?;
        let in_tensor = in_tensor_ref.clone();
        let new_id = self.store.gen_id_internal();

        let result = match op {
            UnaryOp::Scale(s) => {
                scalar_mul(&in_tensor, s, new_id).map_err(EngineError::InvalidOp)?
            }
            UnaryOp::Normalize => {
                normalize_1d(&in_tensor, new_id).map_err(EngineError::InvalidOp)?
            }
            UnaryOp::Transpose => transpose(&in_tensor, new_id).map_err(EngineError::InvalidOp)?,
            UnaryOp::Flatten => flatten(&in_tensor, new_id).map_err(EngineError::InvalidOp)?,
        };

        let out_id = self.store.insert_existing_tensor(result)?;
        self.names.insert(
            output_name.into(),
            NameEntry {
                id: out_id,
                kind: in_kind, // hereda el modo del input
            },
        );
        Ok(())
    }

    /// Evalúa operación binaria: ADD, SUBTRACT, CORRELATE, SIMILARITY, DISTANCE
    pub fn eval_binary(
        &mut self,
        output_name: impl Into<String>,
        left_name: &str,
        right_name: &str,
        op: BinaryOp,
    ) -> Result<(), EngineError> {
        let (a_ref, kind_a) = self.get_with_kind(left_name)?;
        let (b_ref, kind_b) = self.get_with_kind(right_name)?;
        let a = a_ref.clone();
        let b = b_ref.clone();
        let new_id = self.store.gen_id_internal();

        // Si alguno es STRICT, el resultado también es STRICT.
        let out_kind = match (kind_a, kind_b) {
            (TensorKind::Strict, _) | (_, TensorKind::Strict) => TensorKind::Strict,
            _ => TensorKind::Normal,
        };

        let result_tensor = match op {
            BinaryOp::Add => match out_kind {
                TensorKind::Strict => add(&a, &b, new_id).map_err(EngineError::InvalidOp)?,
                TensorKind::Normal => {
                    add_relaxed(&a, &b, new_id).map_err(EngineError::InvalidOp)?
                }
            },
            BinaryOp::Subtract => match out_kind {
                TensorKind::Strict => sub(&a, &b, new_id).map_err(EngineError::InvalidOp)?,
                TensorKind::Normal => {
                    sub_relaxed(&a, &b, new_id).map_err(EngineError::InvalidOp)?
                }
            },
            BinaryOp::Multiply => match out_kind {
                TensorKind::Strict => multiply(&a, &b, new_id).map_err(EngineError::InvalidOp)?,
                TensorKind::Normal => {
                    multiply_relaxed(&a, &b, new_id).map_err(EngineError::InvalidOp)?
                }
            },
            BinaryOp::Divide => match out_kind {
                TensorKind::Strict => divide(&a, &b, new_id).map_err(EngineError::InvalidOp)?,
                TensorKind::Normal => {
                    divide_relaxed(&a, &b, new_id).map_err(EngineError::InvalidOp)?
                }
            },
            BinaryOp::Correlate => {
                // CORRELATE = dot → escalar (sigue siendo estricto en shape)
                let value = dot_1d(&a, &b).map_err(EngineError::InvalidOp)?;
                let shape = Shape::new(Vec::<usize>::new());
                let data = vec![value];
                Tensor::new(new_id, shape, data).map_err(EngineError::InvalidOp)?
            }
            BinaryOp::Similarity => {
                let value = cosine_similarity_1d(&a, &b).map_err(EngineError::InvalidOp)?;
                let shape = Shape::new(Vec::<usize>::new());
                let data = vec![value];
                Tensor::new(new_id, shape, data).map_err(EngineError::InvalidOp)?
            }
            BinaryOp::Distance => {
                let value = distance_1d(&a, &b).map_err(EngineError::InvalidOp)?;
                let shape = Shape::new(Vec::<usize>::new());
                let data = vec![value];
                Tensor::new(new_id, shape, data).map_err(EngineError::InvalidOp)?
            }
        };

        let out_id = self.store.insert_existing_tensor(result_tensor)?;
        self.names.insert(
            output_name.into(),
            NameEntry {
                id: out_id,
                kind: out_kind,
            },
        );
        Ok(())
    }

    /// Para debug: todos los nombres registrados
    pub fn list_names(&self) -> Vec<String> {
        self.names.keys().cloned().collect()
    }

    /// Matrix multiplication: C = MATMUL A B
    pub fn eval_matmul(
        &mut self,
        output_name: impl Into<String>,
        left_name: &str,
        right_name: &str,
    ) -> Result<(), EngineError> {
        let (a_ref, kind_a) = self.get_with_kind(left_name)?;
        let (b_ref, kind_b) = self.get_with_kind(right_name)?;
        let a = a_ref.clone();
        let b = b_ref.clone();
        let new_id = self.store.gen_id_internal();

        let result = matmul(&a, &b, new_id).map_err(EngineError::InvalidOp)?;

        let out_kind = match (kind_a, kind_b) {
            (TensorKind::Strict, _) | (_, TensorKind::Strict) => TensorKind::Strict,
            _ => TensorKind::Normal,
        };

        let out_id = self.store.insert_existing_tensor(result)?;
        self.names.insert(
            output_name.into(),
            NameEntry {
                id: out_id,
                kind: out_kind,
            },
        );
        Ok(())
    }

    /// Reshape tensor: B = RESHAPE A TO [new_shape]
    pub fn eval_reshape(
        &mut self,
        output_name: impl Into<String>,
        input_name: &str,
        new_shape: Shape,
    ) -> Result<(), EngineError> {
        let (in_tensor_ref, in_kind) = self.get_with_kind(input_name)?;
        let in_tensor = in_tensor_ref.clone();
        let new_id = self.store.gen_id_internal();

        let result = reshape(&in_tensor, new_shape, new_id).map_err(EngineError::InvalidOp)?;

        let out_id = self.store.insert_existing_tensor(result)?;
        self.names.insert(
            output_name.into(),
            NameEntry {
                id: out_id,
                kind: in_kind,
            },
        );
        Ok(())
    }

    /// Stack tensors: C = STACK A B
    pub fn eval_stack(
        &mut self,
        output_name: impl Into<String>,
        input_names: Vec<&str>,
        axis: usize,
    ) -> Result<(), EngineError> {
        // Collect tensors
        let mut tensors = Vec::with_capacity(input_names.len());
        let mut kind = TensorKind::Normal;

        // We need to fetch all first.
        // references issue: we can't hold refs while mutating self?
        // self.get_with_kind borrows self immutable.
        // We clone them immediately.

        for name in input_names {
            let (t, k) = self.get_with_kind(name)?;
            if matches!(k, TensorKind::Strict) {
                kind = TensorKind::Strict;
            }
            tensors.push(t.clone()); // Cloning data. Optimization: use Arc<Tensor> later.
        }

        let tensor_refs: Vec<&Tensor> = tensors.iter().collect();
        let new_id = self.store.gen_id_internal();

        let result =
            super::kernels::stack(&tensor_refs, axis, new_id).map_err(EngineError::InvalidOp)?;

        let out_id = self.store.insert_existing_tensor(result)?;
        self.names
            .insert(output_name.into(), NameEntry { id: out_id, kind });
        Ok(())
    }

    // ===== Dataset Management Methods =====

    /// Create a new dataset with schema
    pub fn create_dataset(
        &mut self,
        name: String,
        schema: Arc<Schema>,
    ) -> Result<DatasetId, EngineError> {
        let id = self.dataset_store.gen_id();
        let dataset = Dataset::new(id, schema, Some(name.clone()));
        self.dataset_store
            .insert(dataset, Some(name))
            .map_err(EngineError::from)
    }

    /// Get dataset by name
    pub fn get_dataset(&self, name: &str) -> Result<&Dataset, EngineError> {
        self.dataset_store
            .get_by_name(name)
            .map_err(|_| EngineError::DatasetNotFound(name.to_string()))
    }

    /// Get mutable dataset by name
    pub fn get_dataset_mut(&mut self, name: &str) -> Result<&mut Dataset, EngineError> {
        self.dataset_store
            .get_mut_by_name(name)
            .map_err(|_| EngineError::DatasetNotFound(name.to_string()))
    }

    /// Insert row into dataset
    pub fn insert_row(&mut self, dataset_name: &str, tuple: Tuple) -> Result<(), EngineError> {
        let dataset = self.get_dataset_mut(dataset_name)?;
        dataset
            .add_row(tuple)
            .map_err(|e| EngineError::InvalidOp(e))
    }

    /// List all dataset names
    pub fn list_dataset_names(&self) -> Vec<String> {
        self.dataset_store.list_names()
    }

    /// Add a column to an existing dataset
    pub fn alter_dataset_add_column(
        &mut self,
        dataset_name: &str,
        column_name: String,
        value_type: crate::core::value::ValueType,
        default_value: crate::core::value::Value,
        nullable: bool,
    ) -> Result<(), EngineError> {
        let dataset = self.get_dataset_mut(dataset_name)?;
        dataset
            .add_column(column_name, value_type, default_value, nullable)
            .map_err(|e| EngineError::InvalidOp(e))
    }

    /// Index into a tensor: output = tensor[indices]
    pub fn eval_index(
        &mut self,
        output_name: impl Into<String>,
        tensor_name: &str,
        indices: Vec<usize>,
    ) -> Result<(), EngineError> {
        let (tensor_ref, kind) = self.get_with_kind(tensor_name)?;
        let tensor = tensor_ref.clone();
        let new_id = self.store.gen_id_internal();

        let result = super::kernels::index_to_scalar(&tensor, &indices, new_id)
            .map_err(EngineError::InvalidOp)?;

        let out_id = self.store.insert_existing_tensor(result)?;
        self.names
            .insert(output_name.into(), NameEntry { id: out_id, kind });
        Ok(())
    }

    /// Slice a tensor: output = tensor[slice_specs]
    pub fn eval_slice(
        &mut self,
        output_name: impl Into<String>,
        tensor_name: &str,
        specs: Vec<super::kernels::SliceSpec>,
    ) -> Result<(), EngineError> {
        let (tensor_ref, kind) = self.get_with_kind(tensor_name)?;
        let tensor = tensor_ref.clone();
        let new_id = self.store.gen_id_internal();

        let result =
            super::kernels::slice_multi(&tensor, &specs, new_id).map_err(EngineError::InvalidOp)?;

        let out_id = self.store.insert_existing_tensor(result)?;
        self.names
            .insert(output_name.into(), NameEntry { id: out_id, kind });
        Ok(())
    }

    /// Access a tuple field: output = tuple.field
    /// Returns the field value as a scalar tensor
    pub fn eval_field_access(
        &mut self,
        output_name: impl Into<String>,
        tuple_name: &str,
        field_name: &str,
    ) -> Result<(), EngineError> {
        // For now, we'll store tuples as datasets with a single row
        // This is a simplification - in the future we might have dedicated tuple storage
        let dataset = self.get_dataset(tuple_name)?;

        if dataset.rows.is_empty() {
            return Err(EngineError::InvalidOp(format!(
                "Cannot access field of empty dataset '{}'",
                tuple_name
            )));
        }

        // Get the first row (treating dataset as tuple)
        let row = &dataset.rows[0];
        let value = row
            .get(field_name)
            .ok_or_else(|| EngineError::InvalidOp(format!("Field '{}' not found", field_name)))?
            .clone(); // Clone to avoid borrow issues

        // Convert value to scalar tensor
        let new_id = self.store.gen_id_internal();
        let shape = crate::core::tensor::Shape::new(vec![]);

        let tensor_data = match value {
            crate::core::value::Value::Float(f) => vec![f],
            crate::core::value::Value::Int(i) => vec![i as f32],
            crate::core::value::Value::Bool(b) => vec![if b { 1.0 } else { 0.0 }],
            _ => {
                return Err(EngineError::InvalidOp(format!(
                    "Cannot convert field '{}' to tensor",
                    field_name
                )))
            }
        };

        let tensor = crate::core::tensor::Tensor::new(new_id, shape, tensor_data)
            .map_err(|e| EngineError::InvalidOp(e))?;

        let out_id = self.store.insert_existing_tensor(tensor)?;
        self.names.insert(
            output_name.into(),
            NameEntry {
                id: out_id,
                kind: TensorKind::Normal,
            },
        );
        Ok(())
    }

    /// Extract a column from a dataset: output = dataset.column
    /// Returns the column as a vector tensor
    pub fn eval_column_access(
        &mut self,
        output_name: impl Into<String>,
        dataset_name: &str,
        column_name: &str,
    ) -> Result<(), EngineError> {
        // Clone the dataset to avoid borrow issues
        let dataset = self.get_dataset(dataset_name)?.clone();
        let column_values = dataset
            .get_column(column_name)
            .map_err(|e| EngineError::InvalidOp(e))?;

        // Convert column values to tensor
        let new_id = self.store.gen_id_internal();
        let shape = crate::core::tensor::Shape::new(vec![column_values.len()]);

        let tensor_data: Result<Vec<f32>, String> = column_values
            .iter()
            .map(|v| match v {
                crate::core::value::Value::Float(f) => Ok(*f),
                crate::core::value::Value::Int(i) => Ok(*i as f32),
                crate::core::value::Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
                _ => Err(format!("Cannot convert value to tensor: {:?}", v)),
            })
            .collect();

        let tensor_data = tensor_data.map_err(|e| EngineError::InvalidOp(e))?;
        let tensor = crate::core::tensor::Tensor::new(new_id, shape, tensor_data)
            .map_err(|e| EngineError::InvalidOp(e))?;

        let out_id = self.store.insert_existing_tensor(tensor)?;
        self.names.insert(
            output_name.into(),
            NameEntry {
                id: out_id,
                kind: TensorKind::Normal,
            },
        );
        Ok(())
    }

    /// Create a standard hash index on a dataset column
    pub fn create_index(
        &mut self,
        dataset_name: &str,
        column_name: &str,
    ) -> Result<(), EngineError> {
        let dataset = self.get_dataset_mut(dataset_name)?;
        let index = Box::new(crate::core::index::hash::HashIndex::new());
        dataset
            .create_index(column_name.to_string(), index)
            .map_err(|e| EngineError::InvalidOp(e))
    }

    /// Create a vector index on a dataset column
    pub fn create_vector_index(
        &mut self,
        dataset_name: &str,
        column_name: &str,
    ) -> Result<(), EngineError> {
        let dataset = self.get_dataset_mut(dataset_name)?;
        let index = Box::new(crate::core::index::vector::VectorIndex::new());
        dataset
            .create_index(column_name.to_string(), index)
            .map_err(|e| EngineError::InvalidOp(e))
    }

    /// Get all indices info
    pub fn list_indices(&self) -> Vec<(String, String, String)> {
        let mut result = Vec::new();
        for name in self.dataset_store.list_names() {
            if let Ok(ds) = self.get_dataset(&name) {
                for (col, idx) in &ds.indices {
                    let type_str = match idx.index_type() {
                        crate::core::index::IndexType::Hash => "HASH",
                        crate::core::index::IndexType::Vector => "VECTOR",
                    };
                    result.push((name.clone(), col.clone(), type_str.to_string()));
                }
            }
        }
        result
    }
}
