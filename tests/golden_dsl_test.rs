use linal::dsl::{execute_line, DslOutput};
use linal::engine::TensorDb;

#[test]
fn test_golden_dsl_outputs() {
    let mut db = TensorDb::new();

    // 1. Vector creation
    let out = execute_line(&mut db, "VECTOR v = [1.0, 2.0]", 1).unwrap();
    assert!(matches!(out, DslOutput::Message(_)));

    // 2. SHOW vector
    let out = execute_line(&mut db, "SHOW v", 2).unwrap();
    if let DslOutput::Tensor(t) = out {
        assert_eq!(t.data_ref(), &[1.0, 2.0]);
    } else {
        panic!("Expected Tensor output from SHOW");
    }

    // 3. Matrix creation
    let out = execute_line(&mut db, "MATRIX m = [[1, 2], [3, 4]]", 3).unwrap();
    assert!(matches!(out, DslOutput::Message(_)));

    // 4. SHOW matrix
    let out = execute_line(&mut db, "SHOW m", 4).unwrap();
    if let DslOutput::Tensor(t) = out {
        assert_eq!(t.shape.dims, vec![2, 2]);
        assert_eq!(t.data_ref(), &[1.0, 2.0, 3.0, 4.0]);
    } else {
        panic!("Expected Tensor output from SHOW");
    }

    // 5. LET operation
    let out = execute_line(&mut db, "LET res = v * 10.0", 5).unwrap();
    assert!(matches!(out, DslOutput::Message(_)));

    // 6. SHOW result
    let out = execute_line(&mut db, "SHOW res", 6).unwrap();
    if let DslOutput::Tensor(t) = out {
        assert_eq!(t.data_ref(), &[10.0, 20.0]);
    } else {
        panic!("Expected Tensor output from SHOW");
    }

    // 4. BIND
    let out = execute_line(&mut db, "BIND v2 TO v", 4).unwrap();
    assert!(matches!(out, DslOutput::Message(_)));

    // 5. DATASET creation
    let out = execute_line(&mut db, "DATASET ds COLUMNS (id: INT, val: FLOAT)", 5).unwrap();
    assert!(matches!(out, DslOutput::Message(_)));

    // 6. INSERT
    let out = execute_line(&mut db, "INSERT INTO ds VALUES (1, 99.9)", 6).unwrap();
    assert!(matches!(out, DslOutput::None));

    // 7. SELECT
    let out = execute_line(&mut db, "SELECT id, val FROM ds WHERE id = 1", 7).unwrap();
    if let DslOutput::Table(ds) = out {
        assert_eq!(ds.len(), 1);
        assert_eq!(ds.schema.fields[0].name, "id");
    } else {
        panic!("Expected Table output from SELECT");
    }
}
