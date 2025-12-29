use linal::dsl::execute_script;
use linal::TensorDb;

#[test]
fn symbol_resolution() {
    let mut db = TensorDb::new();

    let script = r#"
        VECTOR v1 = [1.0, 2.0, 3.0]
        LET ds = dataset("test_ds")
        ds.add_column("vec", v1)
        
        LET v2 = ds.vec * 2.0
        
        ds.add_column("vec_doubled", v2)
        
        SAVE DATASET test_ds TO "test_output.parquet"
        
        SHOW ds
    "#;

    execute_script(&mut db, script).unwrap();

    let ds = db.get_tensor_dataset("test_ds").unwrap();
    assert_eq!(ds.columns.len(), 2);

    // Check v2 value
    let v2_tensor = db.get("v2").unwrap();
    assert_eq!(*v2_tensor.data, vec![2.0, 4.0, 6.0]);

    // Clear and load back
    let mut db2 = TensorDb::new();
    let load_script = r#"
        LOAD DATASET test_ds FROM "test_output.parquet"
        SHOW test_ds
    "#;
    execute_script(&mut db2, load_script).unwrap();

    let loaded_ds = db2.get_dataset("test_ds").unwrap();
    assert_eq!(loaded_ds.len(), 3);
}
