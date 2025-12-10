// tests/dataset_integration.rs

use std::sync::Arc;
use vector_db_rs::{Dataset, DatasetId, Field, Schema, Tuple, Value, ValueType};

#[test]
fn test_dataset_complete_workflow() {
    // Create a schema for a user dataset
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", ValueType::Int),
        Field::new("name", ValueType::String),
        Field::new("age", ValueType::Int),
        Field::new("score", ValueType::Float),
        Field::new("active", ValueType::Bool),
    ]));

    // Create dataset
    let mut dataset = Dataset::new(DatasetId(1), schema.clone(), Some("users".to_string()));

    // Add rows
    let users = vec![
        vec![
            Value::Int(1),
            Value::String("Alice".to_string()),
            Value::Int(30),
            Value::Float(0.95),
            Value::Bool(true),
        ],
        vec![
            Value::Int(2),
            Value::String("Bob".to_string()),
            Value::Int(25),
            Value::Float(0.85),
            Value::Bool(true),
        ],
        vec![
            Value::Int(3),
            Value::String("Carol".to_string()),
            Value::Int(35),
            Value::Float(0.90),
            Value::Bool(false),
        ],
        vec![
            Value::Int(4),
            Value::String("Dave".to_string()),
            Value::Int(28),
            Value::Float(0.88),
            Value::Bool(true),
        ],
    ];

    for values in users {
        let tuple = Tuple::new(schema.clone(), values).unwrap();
        dataset.add_row(tuple).unwrap();
    }

    assert_eq!(dataset.len(), 4);
    assert_eq!(dataset.metadata.name, Some("users".to_string()));
    assert_eq!(dataset.metadata.row_count, 4);

    // Test filtering: active users only
    let active_users = dataset.filter(|row| {
        if let Some(Value::Bool(active)) = row.get("active") {
            *active
        } else {
            false
        }
    });

    assert_eq!(active_users.len(), 3); // Alice, Bob, Dave

    // Test filtering: age > 27
    let older_users = dataset.filter(|row| {
        if let Some(Value::Int(age)) = row.get("age") {
            *age > 27
        } else {
            false
        }
    });

    assert_eq!(older_users.len(), 3); // Alice (30), Carol (35), Dave (28)

    // Test projection: select only name and score
    let name_score = dataset.select(&["name", "score"]).unwrap();
    assert_eq!(name_score.schema.len(), 2);
    assert_eq!(name_score.len(), 4);

    // Verify projected data
    if let Some(Value::String(name)) = name_score.rows[0].get("name") {
        assert_eq!(name, "Alice");
    } else {
        panic!("Expected string value for name");
    }

    // Test sorting by age
    let sorted_by_age = dataset.sort_by("age", true).unwrap();
    if let Some(Value::Int(youngest_age)) = sorted_by_age.rows[0].get("age") {
        assert_eq!(*youngest_age, 25); // Bob is youngest
    }

    if let Some(Value::Int(oldest_age)) = sorted_by_age.rows[3].get("age") {
        assert_eq!(*oldest_age, 35); // Carol is oldest
    }

    // Test take and skip
    let top_2 = dataset.take(2);
    assert_eq!(top_2.len(), 2);

    let skip_first = dataset.skip(1);
    assert_eq!(skip_first.len(), 3);

    // Test chaining operations: filter active users, sort by score desc, take top 2
    let top_active_by_score = dataset
        .filter(|row| {
            if let Some(Value::Bool(active)) = row.get("active") {
                *active
            } else {
                false
            }
        })
        .sort_by("score", false)
        .unwrap()
        .take(2);

    assert_eq!(top_active_by_score.len(), 2);
    // Should be Alice (0.95) and Dave (0.88)
    if let Some(Value::Float(score)) = top_active_by_score.rows[0].get("score") {
        assert_eq!(*score, 0.95);
    }
}

#[test]
fn test_dataset_with_nullable_fields() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", ValueType::Int),
        Field::new("name", ValueType::String),
        Field::new("email", ValueType::String).nullable(),
        Field::new("phone", ValueType::String).nullable(),
    ]));

    let mut dataset = Dataset::new(DatasetId(2), schema.clone(), Some("contacts".to_string()));

    // Add rows with some null values
    let contacts = vec![
        vec![
            Value::Int(1),
            Value::String("Alice".to_string()),
            Value::String("alice@example.com".to_string()),
            Value::Null,
        ],
        vec![
            Value::Int(2),
            Value::String("Bob".to_string()),
            Value::Null,
            Value::String("555-1234".to_string()),
        ],
        vec![
            Value::Int(3),
            Value::String("Carol".to_string()),
            Value::String("carol@example.com".to_string()),
            Value::String("555-5678".to_string()),
        ],
    ];

    for values in contacts {
        let tuple = Tuple::new(schema.clone(), values).unwrap();
        dataset.add_row(tuple).unwrap();
    }

    assert_eq!(dataset.len(), 3);

    // Check null counts in metadata
    let email_stats = dataset.metadata.column_stats.get("email").unwrap();
    assert_eq!(email_stats.null_count, 1); // Bob has no email

    let phone_stats = dataset.metadata.column_stats.get("phone").unwrap();
    assert_eq!(phone_stats.null_count, 1); // Alice has no phone

    // Filter contacts with email
    let with_email = dataset.filter(|row| {
        if let Some(email) = row.get("email") {
            !email.is_null()
        } else {
            false
        }
    });

    assert_eq!(with_email.len(), 2); // Alice and Carol
}

#[test]
fn test_dataset_with_vector_embeddings() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", ValueType::Int),
        Field::new("text", ValueType::String),
        Field::new("embedding", ValueType::Vector(3)),
    ]));

    let mut dataset = Dataset::new(DatasetId(3), schema.clone(), Some("documents".to_string()));

    // Add documents with embeddings
    let documents = vec![
        vec![
            Value::Int(1),
            Value::String("Hello world".to_string()),
            Value::Vector(vec![0.1, 0.2, 0.3]),
        ],
        vec![
            Value::Int(2),
            Value::String("Rust programming".to_string()),
            Value::Vector(vec![0.4, 0.5, 0.6]),
        ],
        vec![
            Value::Int(3),
            Value::String("Machine learning".to_string()),
            Value::Vector(vec![0.7, 0.8, 0.9]),
        ],
    ];

    for values in documents {
        let tuple = Tuple::new(schema.clone(), values).unwrap();
        dataset.add_row(tuple).unwrap();
    }

    assert_eq!(dataset.len(), 3);

    // Verify embeddings are stored correctly
    if let Some(Value::Vector(embedding)) = dataset.rows[0].get("embedding") {
        assert_eq!(embedding.len(), 3);
        assert_eq!(embedding[0], 0.1);
    } else {
        panic!("Expected vector value for embedding");
    }

    // Select only text and embedding
    let text_embeddings = dataset.select(&["text", "embedding"]).unwrap();
    assert_eq!(text_embeddings.schema.len(), 2);
    assert_eq!(text_embeddings.len(), 3);
}

#[test]
fn test_dataset_statistics() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", ValueType::Int),
        Field::new("value", ValueType::Float),
        Field::new("category", ValueType::String),
    ]));

    let rows = vec![
        Tuple::new(
            schema.clone(),
            vec![
                Value::Int(1),
                Value::Float(10.5),
                Value::String("A".to_string()),
            ],
        )
        .unwrap(),
        Tuple::new(
            schema.clone(),
            vec![
                Value::Int(2),
                Value::Float(20.3),
                Value::String("B".to_string()),
            ],
        )
        .unwrap(),
        Tuple::new(
            schema.clone(),
            vec![
                Value::Int(3),
                Value::Float(15.7),
                Value::String("A".to_string()),
            ],
        )
        .unwrap(),
        Tuple::new(
            schema.clone(),
            vec![
                Value::Int(4),
                Value::Float(25.1),
                Value::String("C".to_string()),
            ],
        )
        .unwrap(),
    ];

    let dataset =
        Dataset::with_rows(DatasetId(4), schema, rows, Some("metrics".to_string())).unwrap();

    // Check statistics
    assert_eq!(dataset.metadata.row_count, 4);

    // Check value column stats
    let value_stats = dataset.metadata.column_stats.get("value").unwrap();
    assert_eq!(value_stats.min, Some(Value::Float(10.5)));
    assert_eq!(value_stats.max, Some(Value::Float(25.1)));
    assert_eq!(value_stats.null_count, 0);

    // Check id column stats
    let id_stats = dataset.metadata.column_stats.get("id").unwrap();
    assert_eq!(id_stats.min, Some(Value::Int(1)));
    assert_eq!(id_stats.max, Some(Value::Int(4)));

    // Check category column stats (strings)
    let category_stats = dataset.metadata.column_stats.get("category").unwrap();
    assert_eq!(category_stats.min, Some(Value::String("A".to_string())));
    assert_eq!(category_stats.max, Some(Value::String("C".to_string())));
}

#[test]
fn test_dataset_complex_filtering() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", ValueType::Int),
        Field::new("name", ValueType::String),
        Field::new("age", ValueType::Int),
        Field::new("score", ValueType::Float),
    ]));

    let rows = vec![
        Tuple::new(
            schema.clone(),
            vec![
                Value::Int(1),
                Value::String("Alice".to_string()),
                Value::Int(30),
                Value::Float(0.95),
            ],
        )
        .unwrap(),
        Tuple::new(
            schema.clone(),
            vec![
                Value::Int(2),
                Value::String("Bob".to_string()),
                Value::Int(25),
                Value::Float(0.75),
            ],
        )
        .unwrap(),
        Tuple::new(
            schema.clone(),
            vec![
                Value::Int(3),
                Value::String("Carol".to_string()),
                Value::Int(35),
                Value::Float(0.85),
            ],
        )
        .unwrap(),
    ];

    let dataset = Dataset::with_rows(DatasetId(5), schema, rows, None).unwrap();

    // Complex filter: age >= 30 AND score > 0.8
    let filtered = dataset.filter(|row| {
        let age_ok = if let Some(Value::Int(age)) = row.get("age") {
            *age >= 30
        } else {
            false
        };

        let score_ok = if let Some(Value::Float(score)) = row.get("score") {
            *score > 0.8
        } else {
            false
        };

        age_ok && score_ok
    });

    assert_eq!(filtered.len(), 2); // Alice and Carol
}
