// tests/dsl_matrix_ops.rs

use linal::{execute_script, TensorDb};

#[test]
fn test_dsl_vector_syntax() {
    let mut db = TensorDb::new();

    let script = r#"
        VECTOR v = [1, 2, 3]
        SHOW v
    "#;

    execute_script(&mut db, script).unwrap();

    let v = db.get("v").unwrap();
    assert_eq!(v.shape.dims, vec![3]);
    assert_eq!(v.data_ref(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_dsl_matrix_syntax() {
    let mut db = TensorDb::new();

    let script = r#"
        MATRIX A = [[1, 2], [3, 4]]
        SHOW A
    "#;

    execute_script(&mut db, script).unwrap();

    let a = db.get("A").unwrap();
    assert_eq!(a.shape.dims, vec![2, 2]);
    assert_eq!(a.data_ref(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_dsl_matrix_3x2() {
    let mut db = TensorDb::new();

    let script = r#"
        MATRIX M = [[1, 2], [3, 4], [5, 6]]
    "#;

    execute_script(&mut db, script).unwrap();

    let m = db.get("M").unwrap();
    assert_eq!(m.shape.dims, vec![3, 2]);
    assert_eq!(m.data_ref(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_dsl_matmul() {
    let mut db = TensorDb::new();

    let script = r#"
        MATRIX A = [[1, 2, 3], [4, 5, 6]]
        MATRIX B = [[7, 8], [9, 10], [11, 12]]
        LET C = MATMUL A B
        SHOW C
    "#;

    execute_script(&mut db, script).unwrap();

    let c = db.get("C").unwrap();
    assert_eq!(c.shape.dims, vec![2, 2]);
    // Expected: [[58, 64], [139, 154]]
    assert_eq!(c.data_ref(), vec![58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn test_dsl_transpose() {
    let mut db = TensorDb::new();

    let script = r#"
        MATRIX A = [[1, 2, 3], [4, 5, 6]]
        LET A_T = TRANSPOSE A
        SHOW A_T
    "#;

    execute_script(&mut db, script).unwrap();

    let a_t = db.get("A_T").unwrap();
    assert_eq!(a_t.shape.dims, vec![3, 2]);
    assert_eq!(a_t.data_ref(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn test_dsl_flatten() {
    let mut db = TensorDb::new();

    let script = r#"
        MATRIX M = [[1, 2, 3], [4, 5, 6]]
        LET flat = FLATTEN M
        SHOW flat
    "#;

    execute_script(&mut db, script).unwrap();

    let flat = db.get("flat").unwrap();
    assert_eq!(flat.shape.dims, vec![6]);
    assert_eq!(flat.data_ref(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_dsl_reshape() {
    let mut db = TensorDb::new();

    let script = r#"
        VECTOR v = [1, 2, 3, 4, 5, 6]
        LET M = RESHAPE v TO [2, 3]
        SHOW M
    "#;

    execute_script(&mut db, script).unwrap();

    let m = db.get("M").unwrap();
    assert_eq!(m.shape.dims, vec![2, 3]);
    assert_eq!(m.data_ref(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_dsl_matrix_chain_operations() {
    let mut db = TensorDb::new();

    let script = r#"
        # Create two matrices
        MATRIX A = [[1, 2], [3, 4]]
        MATRIX B = [[5, 6], [7, 8]]
        
        # Multiply them
        LET C = MATMUL A B
        
        # Transpose the result
        LET C_T = TRANSPOSE C
        
        # Flatten it
        LET flat = FLATTEN C_T
        
        SHOW flat
    "#;

    execute_script(&mut db, script).unwrap();

    // Verify the chain worked
    let flat = db.get("flat").unwrap();
    assert_eq!(flat.shape.dims, vec![4]);
    assert_eq!(flat.shape.rank(), 1);
}

#[test]
fn test_dsl_vector_operations() {
    let mut db = TensorDb::new();

    let script = r#"
        VECTOR a = [1, 2, 3]
        VECTOR b = [4, 5, 6]
        
        LET sum = ADD a b
        LET dot = CORRELATE a WITH b
        LET sim = SIMILARITY a WITH b
        
        SHOW sum
        SHOW dot
        SHOW sim
    "#;

    execute_script(&mut db, script).unwrap();

    let sum = db.get("sum").unwrap();
    assert_eq!(sum.data_ref(), vec![5.0, 7.0, 9.0]);

    let dot = db.get("dot").unwrap();
    assert_eq!(dot.data[0], 32.0); // 1*4 + 2*5 + 3*6 = 32
}

#[test]
fn test_dsl_reshape_multiple_times() {
    let mut db = TensorDb::new();

    let script = r#"
        VECTOR v = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        LET M1 = RESHAPE v TO [3, 4]
        LET M2 = RESHAPE M1 TO [4, 3]
        LET M3 = RESHAPE M2 TO [2, 6]
        LET back = RESHAPE M3 TO [12]
    "#;

    execute_script(&mut db, script).unwrap();

    let back = db.get("back").unwrap();
    let original = db.get("v").unwrap();
    assert_eq!(back.data, original.data);
}

#[test]
fn test_dsl_matmul_with_transpose() {
    let mut db = TensorDb::new();

    let script = r#"
        MATRIX A = [[1, 2, 3], [4, 5, 6]]
        LET A_T = TRANSPOSE A
        LET result = MATMUL A A_T
    "#;

    execute_script(&mut db, script).unwrap();

    let result = db.get("result").unwrap();
    assert_eq!(result.shape.dims, vec![2, 2]);
    // A * A^T where A is 2x3 and A^T is 3x2 = 2x2 result
    assert_eq!(result.data_ref(), vec![14.0, 32.0, 32.0, 77.0]);
}

#[test]
fn test_dsl_mixed_syntax() {
    let mut db = TensorDb::new();

    let script = r#"
        # Mix old and new syntax
        DEFINE old AS TENSOR [2, 2] VALUES [1, 2, 3, 4]
        MATRIX new = [[5, 6], [7, 8]]
        
        LET result = MATMUL old new
        SHOW result
    "#;

    execute_script(&mut db, script).unwrap();

    let result = db.get("result").unwrap();
    assert_eq!(result.shape.dims, vec![2, 2]);
}

#[test]
fn test_dsl_vector_to_matrix_reshape() {
    let mut db = TensorDb::new();

    let script = r#"
        VECTOR v = [1, 2, 3, 4]
        LET M = RESHAPE v TO [2, 2]
        LET v2 = FLATTEN M
    "#;

    execute_script(&mut db, script).unwrap();

    let v2 = db.get("v2").unwrap();
    let v = db.get("v").unwrap();
    assert_eq!(v2.data, v.data);
}
