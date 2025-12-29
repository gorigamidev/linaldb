use linal::core::dataset::registry::DatasetRegistry;
use linal::core::dataset::{ColumnSchema, Dataset};
use linal::core::tensor::{Shape, TensorId};
use linal::core::value::ValueType;

#[test]
fn test_new_tensor_dataset() {
    let mut ds = Dataset::new("my_ds");

    let id_id = TensorId::new();
    ds.add_column(
        "id".to_string(),
        linal::core::dataset::ResourceReference::tensor(id_id),
        ColumnSchema::new("id".to_string(), ValueType::Int, Shape::new(vec![10])),
    );

    let emb_id = TensorId::new();
    ds.add_column(
        "emb".to_string(),
        linal::core::dataset::ResourceReference::tensor(emb_id),
        ColumnSchema::new(
            "emb".to_string(),
            ValueType::Vector(128),
            Shape::new(vec![10, 128]),
        ),
    );

    assert_eq!(ds.name, "my_ds");
    assert_eq!(
        ds.get_reference("id"),
        Some(&linal::core::dataset::ResourceReference::tensor(id_id))
    );
    assert_eq!(
        ds.get_reference("emb"),
        Some(&linal::core::dataset::ResourceReference::tensor(emb_id))
    );
}

#[test]
fn test_dataset_registry() {
    let mut registry = DatasetRegistry::new();

    let ds = Dataset::new("test");

    registry.register(ds).unwrap();

    let retrieved = registry.get("test").unwrap();
    assert_eq!(retrieved.name, "test");
}
