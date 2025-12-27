use linal::dsl::execute_script;
use linal::TensorDb;

#[test]
fn test_tensor_dataset_dsl() {
    let mut db = TensorDb::new();

    let script = r#"
        VECTOR v1 = [1.0, 2.0, 3.0]
        VECTOR v2 = [4.0, 5.0, 6.0]
        
        LET ds = dataset("my_dataset")
        ds.add_column("vec1", v1)
        ds.add_column("vec2", v2)
        
        SHOW ds
    "#;

    execute_script(&mut db, script).unwrap();

    // Verify dataset in registry
    let ds = db.get_tensor_dataset("my_dataset").unwrap();
    assert_eq!(ds.name, "my_dataset");
    assert_eq!(ds.columns.len(), 2);
    assert!(ds.columns.contains_key("vec1"));
    assert!(ds.columns.contains_key("vec2"));
}
