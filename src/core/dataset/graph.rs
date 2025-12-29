use super::reference::ResourceReference;
use super::registry::DatasetRegistry;
use crate::core::tensor::TensorId;
use std::collections::HashSet;

/// Manages a collection of datasets and resolves their references.
pub struct DatasetGraph<'a> {
    registry: &'a DatasetRegistry,
}

impl<'a> DatasetGraph<'a> {
    pub fn new(registry: &'a DatasetRegistry) -> Self {
        Self { registry }
    }

    /// Resolves a ResourceReference to a final TensorId.
    /// Handles transitive references (Column -> Column -> Tensor).
    pub fn resolve_to_tensor(&self, reference: &ResourceReference) -> Result<TensorId, String> {
        self.resolve_recursive(reference, &mut HashSet::new())
    }

    fn resolve_recursive(
        &self,
        reference: &ResourceReference,
        visited: &mut HashSet<(String, String)>,
    ) -> Result<TensorId, String> {
        match reference {
            ResourceReference::Tensor { id } => Ok(*id),
            ResourceReference::Column { dataset, column } => {
                let key = (dataset.clone(), column.clone());
                if !visited.insert(key) {
                    return Err(format!(
                        "Circular dependency detected at {}.{}",
                        dataset, column
                    ));
                }

                let ds = self
                    .registry
                    .get(dataset)
                    .ok_or_else(|| format!("Dataset '{}' not found during resolution", dataset))?;

                let next_ref = ds.get_reference(column).ok_or_else(|| {
                    format!(
                        "Column '{}' not found in dataset '{}' during resolution",
                        column, dataset
                    )
                })?;

                self.resolve_recursive(next_ref, visited)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::dataset::schema::ColumnSchema;
    use crate::core::dataset::Dataset;
    use crate::core::tensor::Shape;
    use crate::core::value::ValueType;

    #[test]
    fn test_transitive_resolution() {
        let mut registry = DatasetRegistry::new();
        let tensor_id = TensorId::new();

        // Dataset A: c1 -> Tensor
        let mut ds_a = Dataset::new("A");
        ds_a.add_column(
            "c1".to_string(),
            ResourceReference::tensor(tensor_id),
            ColumnSchema::new("c1".to_string(), ValueType::Float, Shape::new(vec![])),
        );
        registry.register(ds_a).unwrap();

        // Dataset B: c2 -> A.c1
        let mut ds_b = Dataset::new("B");
        ds_b.add_column(
            "c2".to_string(),
            ResourceReference::column("A", "c1"),
            ColumnSchema::new("c2".to_string(), ValueType::Float, Shape::new(vec![])),
        );
        registry.register(ds_b).unwrap();

        let graph = DatasetGraph::new(&registry);
        let resolved = graph
            .resolve_to_tensor(&ResourceReference::column("B", "c2"))
            .unwrap();
        assert_eq!(resolved, tensor_id);
    }

    #[test]
    fn test_circular_dependency() {
        let mut registry = DatasetRegistry::new();

        // Dataset A: c1 -> B.c2
        let mut ds_a = Dataset::new("A");
        ds_a.add_column(
            "c1".to_string(),
            ResourceReference::column("B", "c2"),
            ColumnSchema::new("c1".to_string(), ValueType::Float, Shape::new(vec![])),
        );
        registry.register(ds_a).unwrap();

        // Dataset B: c2 -> A.c1
        let mut ds_b = Dataset::new("B");
        ds_b.add_column(
            "c2".to_string(),
            ResourceReference::column("A", "c1"),
            ColumnSchema::new("c2".to_string(), ValueType::Float, Shape::new(vec![])),
        );
        registry.register(ds_b).unwrap();

        let graph = DatasetGraph::new(&registry);
        let result = graph.resolve_to_tensor(&ResourceReference::column("A", "c1"));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Circular dependency"));
    }
}
