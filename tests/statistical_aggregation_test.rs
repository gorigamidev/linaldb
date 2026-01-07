use linal::dsl::execute_line;
use linal::engine::TensorDb;

#[test]
fn test_statistical_aggregations() {
    let mut db = TensorDb::new();

    // 1. Setup vectors
    execute_line(&mut db, "VECTOR v1 = [1, 2, 3, 4]", 1).unwrap();
    execute_line(&mut db, "VECTOR v2 = [1, 1, 1, 1]", 1).unwrap();

    // 2. Test SUM
    execute_line(&mut db, "LET s1 = SUM v1", 1).unwrap();
    let s1 = db.get("s1").unwrap();
    assert_eq!(s1.data_ref()[0], 10.0);
    assert_eq!(s1.shape.dims, vec![1]);

    // 3. Test MEAN
    execute_line(&mut db, "LET m1 = MEAN v1", 1).unwrap();
    let m1 = db.get("m1").unwrap();
    assert_eq!(m1.data_ref()[0], 2.5);

    execute_line(&mut db, "LET m2 = MEAN v2", 1).unwrap();
    let m2 = db.get("m2").unwrap();
    assert_eq!(m2.data_ref()[0], 1.0);

    // 4. Test STDEV
    // v2 = [1, 1, 1, 1], mean = 1, diffs = [0, 0, 0, 0], stdev = 0
    execute_line(&mut db, "LET sd2 = STDEV v2", 1).unwrap();
    let sd2 = db.get("sd2").unwrap();
    assert_eq!(sd2.data_ref()[0], 0.0);

    // v1 = [1, 2, 3, 4], mean = 2.5
    // diffs = [-1.5, -0.5, 0.5, 1.5]
    // squared = [2.25, 0.25, 0.25, 2.25]
    // sum = 5.0
    // variance = 5.0 / 4 = 1.25
    // stdev = sqrt(1.25) approx 1.118
    execute_line(&mut db, "LET sd1 = STDEV v1", 1).unwrap();
    let sd1 = db.get("sd1").unwrap();
    let expected_sd1 = (1.25f32).sqrt();
    assert!((sd1.data_ref()[0] - expected_sd1).abs() < 1e-5);

    // 5. Test LAZY evaluation
    execute_line(&mut db, "LAZY LET s_lazy = SUM v1", 1).unwrap();
    // SHOW triggers evaluation
    let output = execute_line(&mut db, "SHOW s_lazy", 1).unwrap();
    match output {
        linal::dsl::DslOutput::Tensor(t) => {
            assert_eq!(t.data_ref()[0], 10.0);
        }
        _ => panic!("Expected Tensor output from SHOW"),
    }
}

#[test]
fn test_matrix_aggregation() {
    let mut db = TensorDb::new();

    execute_line(&mut db, "MATRIX m = [[1, 2], [3, 4]]", 1).unwrap();
    execute_line(&mut db, "LET s = SUM m", 1).unwrap();
    let s = db.get("s").unwrap();
    assert_eq!(s.data_ref()[0], 10.0);
}
