use linal::core::dataset::ResourceReference;
use linal::dsl::execute_script;
use linal::TensorDb;
use std::sync::Arc;

#[test]
fn test_checkpoints() {
    let mut db = TensorDb::new();

    // 1. Setup: Create tensors and datasets via DSL
    let script = r#"
        VECTOR v1 = [1.0, 2.0, 3.0]
        LET ds1 = dataset("ds1")
        ds1.add_column("col1", v1)
        
        LET v2 = ds1.col1 * 2.0
        LET ds2 = dataset("ds2")
        ds2.add_column("col2", v2)
    "#;
    execute_script(&mut db, script).unwrap();

    // Checkpoint 1: Dataset metadata expresses references explicitly
    let ds1 = db.get_tensor_dataset("ds1").expect("ds1 should exist");
    let ref1 = ds1
        .get_reference("col1")
        .expect("col1 should have a reference");

    // Checkpoint 2: Engine resolves datasets via tensor references
    // col1 should be a direct Tensor reference
    match ref1 {
        ResourceReference::Tensor { id } => {
            let tensor_v1 = db.get("v1").unwrap();
            assert_eq!(*id, tensor_v1.id);
        }
        _ => panic!("Expected direct tensor reference for ds1.col1"),
    }

    // col2 in ds2 was created from ds1.col1 * 2.0.
    // In our current implementation, LET v2 = ds1.col1 * 2.0 creates a NEW tensor v2.
    // ds2.add_column("col2", v2) will point to that new tensor.
    let ds2 = db.get_tensor_dataset("ds2").expect("ds2 should exist");
    let ref2 = ds2
        .get_reference("col2")
        .expect("col2 should have a reference");

    match ref2 {
        ResourceReference::Tensor { .. } => {} // This is normal for now
        _ => panic!("Expected tensor reference for ds2.col2"),
    }

    // Checkpoint 3: No tensor duplication occurs (Zero-copy)
    // When we add v1 to ds1, it should NOT copy the data.
    let v1_tensor = db.get("v1").expect("v1 should exist");
    let ds1_col1_tensor_id = match ref1 {
        ResourceReference::Tensor { id } => *id,
        _ => unreachable!(),
    };
    let ds1_col1_tensor = db.active_instance().store.get(ds1_col1_tensor_id).unwrap();

    assert!(
        Arc::ptr_eq(&v1_tensor.data, &ds1_col1_tensor.data),
        "Data should be shared (zero-copy)"
    );
}

#[test]
fn test_transitive_resolution_engine() {
    let mut db = TensorDb::new();

    // Manually setup a circular-free transitive reference
    // ds_ref.col -> ds_source.col -> Tensor

    let v_data = vec![10.0, 20.0, 30.0];
    db.insert_named(
        "v_source",
        linal::core::tensor::Shape::new(vec![3]),
        v_data.clone(),
    )
    .unwrap();
    let v_source_id = db.active_instance().get_tensor_id("v_source").unwrap();

    let mut ds_source = linal::core::dataset::Dataset::new("ds_source");
    ds_source.add_column(
        "col".to_string(),
        ResourceReference::tensor(v_source_id),
        linal::core::dataset::ColumnSchema::new(
            "col".to_string(),
            linal::core::value::ValueType::Vector(3),
            linal::core::tensor::Shape::new(vec![3]),
        ),
    );
    db.active_instance_mut().register_tensor_dataset(ds_source);

    let mut ds_ref = linal::core::dataset::Dataset::new("ds_ref");
    ds_ref.add_column(
        "view_col".to_string(),
        ResourceReference::column("ds_source", "col"),
        linal::core::dataset::ColumnSchema::new(
            "view_col".to_string(),
            linal::core::value::ValueType::Vector(3),
            linal::core::tensor::Shape::new(vec![3]),
        ),
    );
    db.active_instance_mut().register_tensor_dataset(ds_ref);

    // Verify engine can resolve ds_ref.view_col
    let mut ctx = linal::engine::context::ExecutionContext::new();
    db.active_instance_mut()
        .eval_column_access(&mut ctx, "v_resolved", "ds_ref", "view_col")
        .expect("Engine should resolve transitive reference");

    let v_resolved = db.get("v_resolved").unwrap();
    assert_eq!(*v_resolved.data, v_data);
    assert!(
        Arc::ptr_eq(&v_resolved.data, &db.get("v_source").unwrap().data),
        "Transitive resolution should also be zero-copy"
    );
}
