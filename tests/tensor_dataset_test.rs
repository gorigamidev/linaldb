use linal::core::dataset::registry::DatasetRegistry;
use linal::core::dataset::{ColumnSchema, Dataset};
use linal::core::tensor::{Shape, TensorId};
use linal::core::value::ValueType;

#[test]
fn test_new_tensor_dataset() {
    let mut ds = Dataset::new("my_ds");

    ds.add_column(
        "id".to_string(),
        TensorId(1),
        ColumnSchema {
            name: "id".to_string(),
            value_type: ValueType::Int,
            shape: Shape::new(vec![10]),
        },
    );

    ds.add_column(
        "emb".to_string(),
        TensorId(2),
        ColumnSchema {
            name: "emb".to_string(),
            value_type: ValueType::Vector(128),
            shape: Shape::new(vec![10, 128]),
        },
    );

    assert_eq!(ds.name, "my_ds");
    assert_eq!(ds.get_tensor_id("id"), Some(TensorId(1)));
    assert_eq!(ds.get_tensor_id("emb"), Some(TensorId(2)));
}

#[test]
fn test_dataset_registry() {
    let mut registry = DatasetRegistry::new();

    let ds = Dataset::new("test");

    registry.register(ds).unwrap();

    let retrieved = registry.get("test").unwrap();
    assert_eq!(retrieved.name, "test");
}
