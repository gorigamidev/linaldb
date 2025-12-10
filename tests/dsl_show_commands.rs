// tests/dsl_show_commands.rs

use vector_db_rs::{execute_script, TensorDb};

#[test]
fn test_show_shape_vector() {
    let mut db = TensorDb::new();

    let script = r#"
        VECTOR v = [1, 2, 3, 4, 5]
        SHOW SHAPE v
    "#;

    // This will print to stdout, but we can verify it doesn't error
    execute_script(&mut db, script).unwrap();

    // Verify the tensor exists and has correct shape
    let v = db.get("v").unwrap();
    assert_eq!(v.shape.dims, vec![5]);
    assert_eq!(v.shape.rank(), 1);
}

#[test]
fn test_show_shape_matrix() {
    let mut db = TensorDb::new();

    let script = r#"
        MATRIX m = [[1, 2, 3], [4, 5, 6]]
        SHOW SHAPE m
    "#;

    execute_script(&mut db, script).unwrap();

    let m = db.get("m").unwrap();
    assert_eq!(m.shape.dims, vec![2, 3]);
    assert_eq!(m.shape.rank(), 2);
}

#[test]
fn test_show_shape_scalar() {
    let mut db = TensorDb::new();

    let script = r#"
        VECTOR a = [1, 2, 3]
        VECTOR b = [4, 5, 6]
        MATRIX c = [[1, 2], [3, 4]]
        SHOW ALL TENSORS
    "#;

    execute_script(&mut db, script).unwrap();

    // Verify all tensors exist
    assert!(db.get("a").is_ok());
    assert!(db.get("b").is_ok());
    assert!(db.get("c").is_ok());

    let names = db.list_names();
    assert_eq!(names.len(), 3);
}

#[test]
fn test_show_all_backward_compatibility() {
    let mut db = TensorDb::new();

    let script = r#"
        VECTOR x = [1, 2]
        VECTOR y = [3, 4]
        SHOW ALL
    "#;

    // SHOW ALL should still work (backward compatibility)
    execute_script(&mut db, script).unwrap();

    let names = db.list_names();
    assert_eq!(names.len(), 2);
}

#[test]
fn test_show_all_datasets_placeholder() {
    let mut db = TensorDb::new();

    let script = r#"
        SHOW ALL DATASETS
    "#;

    // Should work but show placeholder message
    execute_script(&mut db, script).unwrap();
}

#[test]
fn test_show_commands_mixed() {
    let mut db = TensorDb::new();

    let script = r#"
        VECTOR v = [1, 2, 3]
        MATRIX m = [[1, 2], [3, 4]]
        
        SHOW v
        SHOW SHAPE v
        SHOW SHAPE m
        SHOW ALL TENSORS
        SHOW ALL
    "#;

    execute_script(&mut db, script).unwrap();
}
