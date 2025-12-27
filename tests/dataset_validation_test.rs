use linal::dsl::execute_script;
use linal::TensorDb;

#[test]
fn test_dataset_validation() {
    let mut db = TensorDb::new();

    let script = r#"
        VECTOR v1 = [1.0, 2.0, 3.0]
        VECTOR v2 = [1.0, 2.0]
        LET ds = dataset("test_ds")
        ds.add_column("c1", v1)
    "#;
    execute_script(&mut db, script).unwrap();

    // This should fail
    let fail_script = "ds.add_column(\"c2\", v2)";
    let result = execute_script(&mut db, fail_script);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("Column 'c2' has 2 rows, but dataset 'test_ds' has 3 rows"));
}
