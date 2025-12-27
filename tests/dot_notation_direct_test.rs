use std::sync::Arc;
use linal::core::tuple::{Field, Schema, Tuple};
use linal::core::value::{Value, ValueType};
use linal::engine::TensorDb;

#[test]
fn test_dataset_column_access_direct() {
    let mut db = TensorDb::new();

    // Create schema
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", ValueType::Int),
        Field::new("name", ValueType::String),
        Field::new("age", ValueType::Int),
    ]));

    // Create dataset
    db.create_dataset("users".to_string(), schema.clone())
        .unwrap();

    // Insert rows
    let row1 = Tuple::new(
        schema.clone(),
        vec![
            Value::Int(1),
            Value::String("Alice".to_string()),
            Value::Int(30),
        ],
    )
    .unwrap();
    let row2 = Tuple::new(
        schema.clone(),
        vec![
            Value::Int(2),
            Value::String("Bob".to_string()),
            Value::Int(25),
        ],
    )
    .unwrap();

    db.insert_row("users", row1).unwrap();
    db.insert_row("users", row2).unwrap();

    // Extract age column using dot notation
    db.eval_column_access("ages", "users", "age").unwrap();

    // Verify
    let tensor = db.get("ages").unwrap();
    println!("Ages tensor shape: {:?}", tensor.shape.dims);
    println!("Ages tensor data: {:?}", tensor.data);

    assert_eq!(tensor.shape.dims, vec![2]);
    assert_eq!(tensor.data_ref(), vec![30.0, 25.0]);
}

#[test]
fn test_tuple_field_access_direct() {
    let mut db = TensorDb::new();

    // Create schema
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", ValueType::Int),
        Field::new("score", ValueType::Float),
    ]));

    // Create dataset with one row (treating as tuple)
    db.create_dataset("user".to_string(), schema.clone())
        .unwrap();

    // Insert one row
    let row = Tuple::new(schema.clone(), vec![Value::Int(1), Value::Float(95.5)]).unwrap();

    db.insert_row("user", row).unwrap();

    // Access field using dot notation
    db.eval_field_access("user_score", "user", "score").unwrap();

    // Verify
    let tensor = db.get("user_score").unwrap();
    println!("Score tensor shape: {:?}", tensor.shape.dims);
    println!("Score tensor data: {:?}", tensor.data);

    assert_eq!(tensor.shape.dims, Vec::<usize>::new()); // Scalar
    assert_eq!(tensor.data_ref(), vec![95.5]);
}

#[test]
fn test_invalid_column() {
    let mut db = TensorDb::new();

    let schema = Arc::new(Schema::new(vec![Field::new("id", ValueType::Int)]));

    db.create_dataset("test".to_string(), schema).unwrap();

    // Try to access non-existent column
    let result = db.eval_column_access("x", "test", "nonexistent");

    assert!(result.is_err());
    println!("Error (expected): {:?}", result.unwrap_err());
}

#[test]
fn test_multiple_columns() {
    let mut db = TensorDb::new();

    let schema = Arc::new(Schema::new(vec![
        Field::new("x", ValueType::Int),
        Field::new("y", ValueType::Int),
        Field::new("z", ValueType::Int),
    ]));

    db.create_dataset("points".to_string(), schema.clone())
        .unwrap();

    // Insert points
    let p1 = Tuple::new(
        schema.clone(),
        vec![Value::Int(1), Value::Int(2), Value::Int(3)],
    )
    .unwrap();
    let p2 = Tuple::new(
        schema.clone(),
        vec![Value::Int(4), Value::Int(5), Value::Int(6)],
    )
    .unwrap();

    db.insert_row("points", p1).unwrap();
    db.insert_row("points", p2).unwrap();

    // Extract each column
    db.eval_column_access("x_vals", "points", "x").unwrap();
    db.eval_column_access("y_vals", "points", "y").unwrap();
    db.eval_column_access("z_vals", "points", "z").unwrap();

    // Verify
    let x = db.get("x_vals").unwrap();
    let y = db.get("y_vals").unwrap();
    let z = db.get("z_vals").unwrap();

    assert_eq!(x.data_ref(), vec![1.0, 4.0]);
    assert_eq!(y.data_ref(), vec![2.0, 5.0]);
    assert_eq!(z.data_ref(), vec![3.0, 6.0]);
}
