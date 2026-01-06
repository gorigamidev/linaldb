// tests/computed_columns_test.rs
//
// Tests for computed columns in ADD COLUMN

use linal::core::value::Value;
use linal::dsl::{execute_line, DslOutput};
use linal::{execute_script, TensorDb};

#[test]
fn test_add_computed_column_basic() {
    let mut db = TensorDb::new();

    let script = r#"
        DATASET items COLUMNS (id: Int, price: Int, qty: Int)
        INSERT INTO items VALUES (1, 10, 2)
        INSERT INTO items VALUES (2, 20, 3)
        INSERT INTO items VALUES (3, 30, 1)
    "#;

    execute_script(&mut db, script).expect("Setup failed");

    // Add computed column: total = price * qty
    let output = execute_line(&mut db, "DATASET items ADD COLUMN total = price * qty", 0)
        .expect("ADD COLUMN computed should succeed");

    match output {
        DslOutput::Message(msg) => {
            assert!(msg.contains("Added computed column 'total'"));
        }
        _ => panic!("Expected Message output"),
    }

    // Verify the computed column was added
    let items = db.get_dataset("items").expect("Dataset should exist");
    assert_eq!(items.schema.len(), 4); // id, price, qty, total
    assert_eq!(items.len(), 3);

    // Verify computed values
    // Row 1: price=10, qty=2, total=20
    assert_eq!(items.rows[0].values[3], Value::Int(20));
    // Row 2: price=20, qty=3, total=60
    assert_eq!(items.rows[1].values[3], Value::Int(60));
    // Row 3: price=30, qty=1, total=30
    assert_eq!(items.rows[2].values[3], Value::Int(30));
}

#[test]
fn test_add_computed_column_addition() {
    let mut db = TensorDb::new();

    let script = r#"
        DATASET users COLUMNS (id: Int, age: Int, bonus: Int)
        INSERT INTO users VALUES (1, 25, 5)
        INSERT INTO users VALUES (2, 30, 10)
    "#;

    execute_script(&mut db, script).expect("Setup failed");

    // Add computed column: total_age = age + bonus
    execute_line(
        &mut db,
        "DATASET users ADD COLUMN total_age = age + bonus",
        0,
    )
    .expect("ADD COLUMN computed should succeed");

    let users = db.get_dataset("users").expect("Dataset should exist");
    assert_eq!(users.rows[0].values[3], Value::Int(30)); // 25 + 5
    assert_eq!(users.rows[1].values[3], Value::Int(40)); // 30 + 10
}

#[test]
fn test_add_computed_column_subtraction() {
    let mut db = TensorDb::new();

    let script = r#"
        DATASET products COLUMNS (id: Int, price: Int, discount: Int)
        INSERT INTO products VALUES (1, 100, 20)
        INSERT INTO products VALUES (2, 200, 30)
    "#;

    execute_script(&mut db, script).expect("Setup failed");

    // Add computed column: final_price = price - discount
    execute_line(
        &mut db,
        "DATASET products ADD COLUMN final_price = price - discount",
        0,
    )
    .expect("ADD COLUMN computed should succeed");

    let products = db.get_dataset("products").expect("Dataset should exist");
    assert_eq!(products.rows[0].values[3], Value::Int(80)); // 100 - 20
    assert_eq!(products.rows[1].values[3], Value::Int(170)); // 200 - 30
}

#[test]
fn test_add_computed_column_division() {
    let mut db = TensorDb::new();

    let script = r#"
        DATASET metrics COLUMNS (id: Int, total: Int, count: Int)
        INSERT INTO metrics VALUES (1, 100, 4)
        INSERT INTO metrics VALUES (2, 150, 3)
    "#;

    execute_script(&mut db, script).expect("Setup failed");

    // Add computed column: average = total / count
    execute_line(
        &mut db,
        "DATASET metrics ADD COLUMN average = total / count",
        0,
    )
    .expect("ADD COLUMN computed should succeed");

    let metrics = db.get_dataset("metrics").expect("Dataset should exist");
    assert_eq!(metrics.rows[0].values[3], Value::Int(25)); // 100 / 4
    assert_eq!(metrics.rows[1].values[3], Value::Int(50)); // 150 / 3
}

#[test]
fn test_add_computed_column_complex_expression() {
    let mut db = TensorDb::new();

    let script = r#"
        DATASET orders COLUMNS (id: Int, price: Int, qty: Int, tax: Int)
        INSERT INTO orders VALUES (1, 10, 2, 1)
        INSERT INTO orders VALUES (2, 20, 3, 2)
    "#;

    execute_script(&mut db, script).expect("Setup failed");

    // Add computed column: total = price * qty + tax
    execute_line(
        &mut db,
        "DATASET orders ADD COLUMN total = price * qty + tax",
        0,
    )
    .expect("ADD COLUMN computed should succeed");

    let orders = db.get_dataset("orders").expect("Dataset should exist");
    // Row 1: 10 * 2 + 1 = 21
    assert_eq!(orders.rows[0].values[4], Value::Int(21));
    // Row 2: 20 * 3 + 2 = 62
    assert_eq!(orders.rows[1].values[4], Value::Int(62));
}

#[test]
fn test_add_computed_column_empty_dataset() {
    let mut db = TensorDb::new();

    let script = r#"
        DATASET empty COLUMNS (id: Int, value: Int)
    "#;

    execute_script(&mut db, script).expect("Setup failed");

    // Should fail because we can't infer type from empty dataset
    let result = execute_line(&mut db, "DATASET empty ADD COLUMN computed = value * 2", 0);
    assert!(result.is_err(), "Should fail on empty dataset");
}

#[test]
fn test_add_computed_column_duplicate_name() {
    let mut db = TensorDb::new();

    let script = r#"
        DATASET test COLUMNS (id: Int, value: Int)
        INSERT INTO test VALUES (1, 10)
    "#;

    execute_script(&mut db, script).expect("Setup failed");

    // Try to add column with existing name
    let result = execute_line(&mut db, "DATASET test ADD COLUMN value = id * 2", 0);
    assert!(result.is_err(), "Should fail on duplicate column name");
}

#[test]
fn test_add_computed_column_float_result() {
    let mut db = TensorDb::new();

    let script = r#"
        DATASET ratios COLUMNS (id: Int, numerator: Int, denominator: Int)
        INSERT INTO ratios VALUES (1, 10, 3)
        INSERT INTO ratios VALUES (2, 20, 3)
    "#;

    execute_script(&mut db, script).expect("Setup failed");

    // Add computed column that results in Float
    // Note: Integer division in Rust truncates, so 10/3 = 3, not 3.33
    // But if we use Float values, we get Float results
    execute_line(
        &mut db,
        "DATASET ratios ADD COLUMN ratio = numerator / denominator",
        0,
    )
    .expect("ADD COLUMN computed should succeed");

    let ratios = db.get_dataset("ratios").expect("Dataset should exist");
    // Integer division: 10 / 3 = 3
    assert_eq!(ratios.rows[0].values[3], Value::Int(3));
    assert_eq!(ratios.rows[1].values[3], Value::Int(6)); // 20 / 3 = 6
}
