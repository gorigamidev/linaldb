use linal::core::value::Value;
use linal::dsl::execute_line;
use linal::engine::TensorDb;

#[test]
fn test_lazy_column_basic() {
    let mut db = TensorDb::new();

    // Create dataset
    execute_line(&mut db, "DATASET test COLUMNS a: INT, b: INT", 1).unwrap();
    execute_line(&mut db, "INSERT INTO test VALUES (1, 2)", 2).unwrap();
    execute_line(&mut db, "INSERT INTO test VALUES (3, 4)", 3).unwrap();

    // Add lazy column
    let result = execute_line(&mut db, "DATASET test ADD COLUMN c = a + b LAZY", 4).unwrap();
    assert!(matches!(result, linal::dsl::DslOutput::Message(_)));

    // Query should evaluate lazy column
    let result = execute_line(&mut db, "SELECT * FROM test", 5).unwrap();
    if let linal::dsl::DslOutput::Table(ds) = result {
        assert_eq!(ds.rows.len(), 2);
        assert_eq!(ds.rows[0].values[2], Value::Int(3)); // 1 + 2
        assert_eq!(ds.rows[1].values[2], Value::Int(7)); // 3 + 4
    } else {
        panic!("Expected table output");
    }
}

#[test]
fn test_lazy_column_vs_materialized() {
    let mut db = TensorDb::new();

    // Create dataset
    execute_line(&mut db, "DATASET test COLUMNS x: INT", 1).unwrap();
    execute_line(&mut db, "INSERT INTO test VALUES (5)", 2).unwrap();
    execute_line(&mut db, "INSERT INTO test VALUES (10)", 3).unwrap();

    // Add materialized column
    execute_line(&mut db, "DATASET test ADD COLUMN doubled = x * 2", 4).unwrap();

    // Add lazy column
    execute_line(&mut db, "DATASET test ADD COLUMN tripled = x * 3 LAZY", 5).unwrap();

    // Query should show both columns evaluated
    let result = execute_line(&mut db, "SELECT * FROM test", 6).unwrap();
    if let linal::dsl::DslOutput::Table(ds) = result {
        assert_eq!(ds.rows.len(), 2);
        assert_eq!(ds.rows[0].values[1], Value::Int(10)); // 5 * 2 (materialized)
        assert_eq!(ds.rows[0].values[2], Value::Int(15)); // 5 * 3 (lazy evaluated)
        assert_eq!(ds.rows[1].values[1], Value::Int(20)); // 10 * 2 (materialized)
        assert_eq!(ds.rows[1].values[2], Value::Int(30)); // 10 * 3 (lazy evaluated)
    } else {
        panic!("Expected table output");
    }
}

#[test]
fn test_lazy_column_get_column() {
    let mut db = TensorDb::new();

    execute_line(&mut db, "DATASET test COLUMNS a: INT, b: INT", 1).unwrap();
    execute_line(&mut db, "INSERT INTO test VALUES (2, 3)", 2).unwrap();
    execute_line(&mut db, "INSERT INTO test VALUES (4, 5)", 3).unwrap();

    // Add lazy column
    execute_line(&mut db, "DATASET test ADD COLUMN sum = a + b LAZY", 4).unwrap();

    // Get dataset and test get_column
    let dataset = db.get_dataset("test").unwrap();
    let column_values = dataset.get_column("sum").unwrap();

    assert_eq!(column_values.len(), 2);
    assert_eq!(column_values[0], Value::Int(5)); // 2 + 3
    assert_eq!(column_values[1], Value::Int(9)); // 4 + 5
}

#[test]
fn test_materialize_lazy_columns() {
    let mut db = TensorDb::new();

    execute_line(&mut db, "DATASET test COLUMNS x: INT", 1).unwrap();
    execute_line(&mut db, "INSERT INTO test VALUES (7)", 2).unwrap();

    // Add lazy column
    execute_line(&mut db, "DATASET test ADD COLUMN squared = x * x LAZY", 3).unwrap();

    // Materialize
    let result = execute_line(&mut db, "MATERIALIZE test", 4).unwrap();
    assert!(matches!(result, linal::dsl::DslOutput::Message(_)));

    // Query - should still work but column is now materialized
    let result = execute_line(&mut db, "SELECT * FROM test", 5).unwrap();
    if let linal::dsl::DslOutput::Table(ds) = result {
        assert_eq!(ds.rows.len(), 1);
        assert_eq!(ds.rows[0].values[1], Value::Int(49)); // 7 * 7
    } else {
        panic!("Expected table output");
    }

    // Verify it's no longer lazy
    let dataset = db.get_dataset("test").unwrap();
    let field = dataset.schema.get_field("squared").unwrap();
    assert!(!field.is_lazy);
}

#[test]
fn test_lazy_column_with_filter() {
    let mut db = TensorDb::new();

    execute_line(&mut db, "DATASET test COLUMNS age: INT", 1).unwrap();
    execute_line(&mut db, "INSERT INTO test VALUES (25)", 2).unwrap();
    execute_line(&mut db, "INSERT INTO test VALUES (30)", 3).unwrap();
    execute_line(&mut db, "INSERT INTO test VALUES (35)", 4).unwrap();

    // Add lazy column
    execute_line(
        &mut db,
        "DATASET test ADD COLUMN next_year = age + 1 LAZY",
        5,
    )
    .unwrap();

    // Filter using lazy column
    let result = execute_line(&mut db, "SELECT * FROM test WHERE next_year > 30", 6).unwrap();
    if let linal::dsl::DslOutput::Table(ds) = result {
        assert_eq!(ds.rows.len(), 2); // 30+1=31 and 35+1=36
    } else {
        panic!("Expected table output");
    }
}

#[test]
fn test_lazy_column_empty_dataset() {
    let mut db = TensorDb::new();

    execute_line(&mut db, "DATASET test COLUMNS a: INT", 1).unwrap();

    // Should fail - can't infer type from empty dataset
    let result = execute_line(&mut db, "DATASET test ADD COLUMN b = a * 2 LAZY", 2);
    assert!(result.is_err());
}

#[test]
fn test_lazy_column_complex_expression() {
    let mut db = TensorDb::new();

    execute_line(&mut db, "DATASET test COLUMNS a: INT, b: INT, c: INT", 1).unwrap();
    execute_line(&mut db, "INSERT INTO test VALUES (1, 2, 3)", 2).unwrap();

    // Add lazy column with complex expression
    execute_line(
        &mut db,
        "DATASET test ADD COLUMN result = (a + b) * c LAZY",
        3,
    )
    .unwrap();

    let result = execute_line(&mut db, "SELECT result FROM test", 4).unwrap();
    if let linal::dsl::DslOutput::Table(ds) = result {
        assert_eq!(ds.rows.len(), 1);
        assert_eq!(ds.rows[0].values[0], Value::Int(9)); // (1 + 2) * 3
    } else {
        panic!("Expected table output");
    }
}

#[test]
fn test_lazy_column_with_float() {
    let mut db = TensorDb::new();

    execute_line(&mut db, "DATASET test COLUMNS price: FLOAT", 1).unwrap();
    execute_line(&mut db, "INSERT INTO test VALUES (10.5)", 2).unwrap();

    // Add lazy column
    execute_line(&mut db, "DATASET test ADD COLUMN tax = price * 0.1 LAZY", 3).unwrap();

    let result = execute_line(&mut db, "SELECT tax FROM test", 4).unwrap();
    if let linal::dsl::DslOutput::Table(ds) = result {
        assert_eq!(ds.rows.len(), 1);
        if let Value::Float(tax) = ds.rows[0].values[0] {
            assert!((tax - 1.05).abs() < 0.001); // 10.5 * 0.1
        } else {
            panic!("Expected Float value");
        }
    } else {
        panic!("Expected table output");
    }
}
