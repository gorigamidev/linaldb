use linal::core::tensor::Shape;
use linal::dsl::execute_script;
use linal::engine::TensorDb;

#[test]
fn test_dsl_bind_tensor() {
    let mut db = TensorDb::new();

    // 1. Create tensor
    db.insert_named("original", Shape::new(vec![2]), vec![1.0, 2.0])
        .unwrap();
    let original_id = db.active_instance().get_tensor_id("original").unwrap();

    // 2. Bind alias
    execute_script(&mut db, "BIND alias TO original").unwrap();

    // 3. Verify alias points to same ID
    let alias_id = db
        .active_instance()
        .get_tensor_id("alias")
        .expect("alias should exist");
    assert_eq!(alias_id, original_id);

    // 4. Use alias in operation
    execute_script(&mut db, "LET result = alias * 2.0").unwrap();
    let result = db.get("result").unwrap();
    assert_eq!(result.data[0], 2.0);
}

#[test]
fn test_dsl_attach_tensor() {
    let mut db = TensorDb::new();

    // 1. Create tensor-first dataset and tensor
    execute_script(&mut db, "LET ds1 = dataset(\"ds1\")").unwrap();
    db.insert_named("vec1", Shape::new(vec![3]), vec![0.1, 0.2, 0.3])
        .unwrap();

    // 2. Attach
    execute_script(&mut db, "ATTACH vec1 TO ds1.embedding").unwrap();

    // 3. Verify in registration
    let ds = db
        .active_instance()
        .tensor_datasets
        .get("ds1")
        .expect("ds1 should exist");
    assert!(ds.columns.contains_key("embedding"));
}

#[test]
fn test_dsl_derive_tensor() {
    let mut db = TensorDb::new();

    // 1. Create base
    db.insert_named("a", Shape::new(vec![1]), vec![10.0])
        .unwrap();
    let a_id = db.active_instance().get_tensor_id("a").unwrap();

    // 2. Derive
    execute_script(&mut db, "DERIVE b FROM a + 5.0").unwrap();

    // 3. Verify value and lineage
    let b = db.get("b").expect("b should exist");
    assert_eq!(b.data[0], 15.0);

    let lineage = b.metadata.lineage.as_ref().expect("b should have lineage");
    assert!(lineage.inputs.contains(&a_id));
}

#[test]
fn test_dsl_retrocompatibility() {
    let mut db = TensorDb::new();

    // Standard legacy-style script
    let script = r#"
        VECTOR a = [1.0, 2.0]
        LET b = a * 10.0
        SHOW b
    "#;

    execute_script(&mut db, script).unwrap();

    let b = db.get("b").unwrap();
    assert_eq!(b.data[0], 10.0);
}
