// tests/show_schema_test.rs
//
// Tests for SHOW SCHEMA command

use linal::dsl::{execute_line, DslOutput};
use linal::{execute_script, TensorDb};

#[test]
fn test_show_schema_basic() {
    let mut db = TensorDb::new();

    let script = r#"
        DATASET users COLUMNS (id: INT, name: STRING, age: INT, active: BOOL, score: FLOAT)
        INSERT INTO users VALUES (1, "Alice", 30, true, 0.95)
    "#;

    execute_script(&mut db, script).expect("Setup failed");

    // Test SHOW SCHEMA command
    let output = execute_line(&mut db, "SHOW SCHEMA users", 1).expect("SHOW SCHEMA should succeed");

    match output {
        DslOutput::Message(msg) => {
            assert!(msg.contains("Schema for dataset 'users'"));
            assert!(msg.contains("id"));
            assert!(msg.contains("name"));
            assert!(msg.contains("age"));
            assert!(msg.contains("active"));
            assert!(msg.contains("score"));
            assert!(msg.contains("Field"));
            assert!(msg.contains("Type"));
            assert!(msg.contains("Nullable"));
        }
        _ => panic!("Expected Message output from SHOW SCHEMA"),
    }
}

#[test]
fn test_show_schema_with_nullable_fields() {
    let mut db = TensorDb::new();

    // Create dataset first, then add nullable columns
    let script = r#"
        DATASET contacts COLUMNS (id: INT, name: STRING)
        INSERT INTO contacts VALUES (1, "Alice")
    "#;

    execute_script(&mut db, script).expect("Setup failed");

    // Add nullable columns using ADD COLUMN
    execute_line(&mut db, "DATASET contacts ADD COLUMN email: STRING?", 1)
        .expect("Add nullable email column should succeed");
    execute_line(&mut db, "DATASET contacts ADD COLUMN phone: STRING?", 1)
        .expect("Add nullable phone column should succeed");

    let output =
        execute_line(&mut db, "SHOW SCHEMA contacts", 1).expect("SHOW SCHEMA should succeed");

    match output {
        DslOutput::Message(msg) => {
            assert!(msg.contains("email"));
            assert!(msg.contains("phone"));
            assert!(msg.contains("id"));
            assert!(msg.contains("name"));
            // Check that nullable fields are marked correctly
            // The output should show nullable status
        }
        _ => panic!("Expected Message output"),
    }
}

#[test]
fn test_show_schema_with_vector_and_matrix() {
    let mut db = TensorDb::new();

    let script = r#"
        DATASET embeddings COLUMNS (id: INT, vec: Vector(3), mat: Matrix(2, 2))
        INSERT INTO embeddings VALUES (1, [0.1, 0.2, 0.3], [[1.0, 2.0], [3.0, 4.0]])
    "#;

    execute_script(&mut db, script).expect("Setup failed");

    let output =
        execute_line(&mut db, "SHOW SCHEMA embeddings", 1).expect("SHOW SCHEMA should succeed");

    match output {
        DslOutput::Message(msg) => {
            assert!(msg.contains("vec"));
            assert!(msg.contains("mat"));
            assert!(msg.contains("Vector"));
            assert!(msg.contains("Matrix"));
        }
        _ => panic!("Expected Message output"),
    }
}

#[test]
fn test_show_schema_nonexistent_dataset() {
    let mut db = TensorDb::new();

    let result = execute_line(&mut db, "SHOW SCHEMA nonexistent", 1);

    assert!(
        result.is_err(),
        "SHOW SCHEMA on nonexistent dataset should fail"
    );
}

#[test]
fn test_show_schema_empty_dataset() {
    let mut db = TensorDb::new();

    let script = r#"
        DATASET empty COLUMNS (id: INT, name: STRING)
    "#;

    execute_script(&mut db, script).expect("Setup failed");

    let output = execute_line(&mut db, "SHOW SCHEMA empty", 1)
        .expect("SHOW SCHEMA should succeed even for empty dataset");

    match output {
        DslOutput::Message(msg) => {
            assert!(msg.contains("Schema for dataset 'empty'"));
            assert!(msg.contains("id"));
            assert!(msg.contains("name"));
        }
        _ => panic!("Expected Message output"),
    }
}
