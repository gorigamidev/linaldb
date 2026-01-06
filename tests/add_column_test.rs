// tests/add_column_test.rs
//
// Tests for ADD COLUMN command

use linal::core::value::Value;
use linal::dsl::{execute_line, DslOutput};
use linal::{execute_script, TensorDb};

#[test]
fn test_add_column_basic() {
    let mut db = TensorDb::new();

    let script = r#"
        DATASET users COLUMNS (id: INT, name: STRING)
        INSERT INTO users VALUES (1, "Alice")
        INSERT INTO users VALUES (2, "Bob")
    "#;

    execute_script(&mut db, script).expect("Setup failed");

    // Add a new column
    let output = execute_line(&mut db, "DATASET users ADD COLUMN age: INT", 1)
        .expect("ADD COLUMN should succeed");

    match output {
        DslOutput::Message(msg) => {
            assert!(msg.contains("Added column 'age'"));
        }
        _ => panic!("Expected Message output"),
    }

    // Verify the column was added
    let users = db.get_dataset("users").expect("Dataset should exist");
    assert_eq!(users.schema.len(), 3); // id, name, age
    assert_eq!(users.len(), 2); // Still 2 rows

    // Verify default values (should be 0 for INT)
    let first_row = &users.rows[0];
    assert_eq!(first_row.values[2], Value::Int(0)); // age should be 0
}

#[test]
fn test_add_column_with_default_value() {
    let mut db = TensorDb::new();

    let script = r#"
        DATASET products COLUMNS (id: INT, name: STRING)
        INSERT INTO products VALUES (1, "Product A")
        INSERT INTO products VALUES (2, "Product B")
    "#;

    execute_script(&mut db, script).expect("Setup failed");

    // Add column with default value
    let output = execute_line(
        &mut db,
        "DATASET products ADD COLUMN price: FLOAT DEFAULT 99.99",
        1,
    )
    .expect("ADD COLUMN with DEFAULT should succeed");

    match output {
        DslOutput::Message(msg) => {
            assert!(msg.contains("Added column 'price'"));
        }
        _ => panic!("Expected Message output"),
    }

    // Verify default values
    let products = db.get_dataset("products").expect("Dataset should exist");
    assert_eq!(products.schema.len(), 3);

    for row in &products.rows {
        assert_eq!(row.values[2], Value::Float(99.99));
    }
}

#[test]
fn test_add_column_nullable() {
    let mut db = TensorDb::new();

    let script = r#"
        DATASET contacts COLUMNS (id: INT, name: STRING)
        INSERT INTO contacts VALUES (1, "Alice")
    "#;

    execute_script(&mut db, script).expect("Setup failed");

    // Add nullable column
    let output = execute_line(&mut db, "DATASET contacts ADD COLUMN email: STRING?", 1)
        .expect("ADD COLUMN nullable should succeed");

    match output {
        DslOutput::Message(msg) => {
            assert!(msg.contains("Added column 'email'"));
        }
        _ => panic!("Expected Message output"),
    }

    // Verify nullable column defaults to null
    let contacts = db.get_dataset("contacts").expect("Dataset should exist");
    assert_eq!(contacts.rows[0].values[2], Value::Null);
}

#[test]
fn test_add_column_duplicate_name() {
    let mut db = TensorDb::new();

    let script = r#"
        DATASET users COLUMNS (id: INT, name: STRING)
    "#;

    execute_script(&mut db, script).expect("Setup failed");

    // Try to add a column with existing name
    let result = execute_line(&mut db, "DATASET users ADD COLUMN name: STRING", 1);

    assert!(result.is_err(), "Adding duplicate column should fail");
}

#[test]
fn test_add_column_different_types() {
    let mut db = TensorDb::new();

    let script = r#"
        DATASET test COLUMNS (id: INT)
        INSERT INTO test VALUES (1)
    "#;

    execute_script(&mut db, script).expect("Setup failed");

    // Add FLOAT column
    execute_line(
        &mut db,
        "DATASET test ADD COLUMN score: FLOAT DEFAULT 0.0",
        1,
    )
    .expect("Add FLOAT column should succeed");

    // Add STRING column
    execute_line(
        &mut db,
        "DATASET test ADD COLUMN label: STRING DEFAULT \"default\"",
        1,
    )
    .expect("Add STRING column should succeed");

    // Add BOOL column
    execute_line(
        &mut db,
        "DATASET test ADD COLUMN active: BOOL DEFAULT true",
        1,
    )
    .expect("Add BOOL column should succeed");

    // Verify all columns exist
    let test = db.get_dataset("test").expect("Dataset should exist");
    assert_eq!(test.schema.len(), 4); // id, score, label, active

    // Verify default values
    let row = &test.rows[0];
    assert_eq!(row.values[1], Value::Float(0.0));
    assert_eq!(row.values[2], Value::String("default".to_string()));
    assert_eq!(row.values[3], Value::Bool(true));
}

#[test]
fn test_add_column_to_empty_dataset() {
    let mut db = TensorDb::new();

    let script = r#"
        DATASET empty COLUMNS (id: INT)
    "#;

    execute_script(&mut db, script).expect("Setup failed");

    // Add column to empty dataset
    let output = execute_line(&mut db, "DATASET empty ADD COLUMN name: STRING", 1)
        .expect("ADD COLUMN to empty dataset should succeed");

    match output {
        DslOutput::Message(msg) => {
            assert!(msg.contains("Added column 'name'"));
        }
        _ => panic!("Expected Message output"),
    }

    // Verify schema was updated
    let empty = db.get_dataset("empty").expect("Dataset should exist");
    assert_eq!(empty.schema.len(), 2);
    assert_eq!(empty.len(), 0); // Still empty
}

#[test]
fn test_add_column_vector_type() {
    let mut db = TensorDb::new();

    let script = r#"
        DATASET items COLUMNS (id: INT, name: STRING)
        INSERT INTO items VALUES (1, "Item 1")
    "#;

    execute_script(&mut db, script).expect("Setup failed");

    // Add vector column
    let output = execute_line(&mut db, "DATASET items ADD COLUMN embedding: Vector(3)", 1)
        .expect("Add Vector column should succeed");

    match output {
        DslOutput::Message(msg) => {
            assert!(msg.contains("Added column 'embedding'"));
        }
        _ => panic!("Expected Message output"),
    }

    // Verify default vector (should be [0.0, 0.0, 0.0])
    let items = db.get_dataset("items").expect("Dataset should exist");
    assert_eq!(items.rows[0].values[2], Value::Vector(vec![0.0, 0.0, 0.0]));
}

#[test]
fn test_add_column_nonexistent_dataset() {
    let mut db = TensorDb::new();

    let result = execute_line(&mut db, "DATASET nonexistent ADD COLUMN col: INT", 1);

    assert!(
        result.is_err(),
        "ADD COLUMN on nonexistent dataset should fail"
    );
}
