use linal::core::dataset_legacy::{Dataset, DatasetId};
use linal::core::tuple::{Field, Schema, Tuple};
use linal::core::value::{Value, ValueType};
use std::sync::Arc;

fn create_large_dataset(num_rows: usize) -> Dataset {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", ValueType::Int),
        Field::new("value", ValueType::Float),
        Field::new("category", ValueType::String),
    ]));

    let mut rows = Vec::with_capacity(num_rows);
    for i in 0..num_rows {
        rows.push(
            Tuple::new(
                schema.clone(),
                vec![
                    Value::Int(i as i64),
                    Value::Float((i as f32) * 1.5),
                    Value::String(if i % 2 == 0 {
                        "even".to_string()
                    } else {
                        "odd".to_string()
                    }),
                ],
            )
            .unwrap(),
        );
    }

    Dataset::with_rows(DatasetId(1), schema, rows, Some("test".to_string())).unwrap()
}

#[test]
fn test_filter_batched_correctness() {
    let dataset = create_large_dataset(5000);

    // Filter using regular method
    let filtered_regular = dataset.filter(|row| {
        if let Some(Value::Int(id)) = row.get("id") {
            *id % 2 == 0
        } else {
            false
        }
    });

    // Filter using batched method
    let filtered_batched = dataset.filter_batched(|row| {
        if let Some(Value::Int(id)) = row.get("id") {
            *id % 2 == 0
        } else {
            false
        }
    });

    // Results should be identical
    assert_eq!(filtered_regular.len(), filtered_batched.len());
    assert_eq!(filtered_regular.len(), 2500); // Half should be even

    // Verify first few rows match
    for i in 0..10 {
        let reg_id = filtered_regular.rows[i].get("id");
        let batch_id = filtered_batched.rows[i].get("id");
        assert_eq!(reg_id, batch_id);
    }
}

#[test]
fn test_filter_batched_large_dataset() {
    // Test with dataset larger than BATCH_PARALLEL_THRESHOLD (10k)
    let dataset = create_large_dataset(15_000);

    let filtered = dataset.filter_batched(|row| {
        if let Some(Value::String(cat)) = row.get("category") {
            cat == "even"
        } else {
            false
        }
    });

    assert_eq!(filtered.len(), 7500); // Half should be even
}

#[test]
fn test_map_batched_correctness() {
    let dataset = create_large_dataset(3000);

    // Map using regular method - double the value
    let mapped_regular = dataset.map(|row| {
        let id = row.get("id").cloned().unwrap();
        let value = if let Some(Value::Float(v)) = row.get("value") {
            Value::Float(v * 2.0)
        } else {
            Value::Float(0.0)
        };
        let category = row.get("category").cloned().unwrap();

        Tuple::new(row.schema.clone(), vec![id, value, category]).unwrap()
    });

    // Map using batched method
    let mapped_batched = dataset.map_batched(|row| {
        let id = row.get("id").cloned().unwrap();
        let value = if let Some(Value::Float(v)) = row.get("value") {
            Value::Float(v * 2.0)
        } else {
            Value::Float(0.0)
        };
        let category = row.get("category").cloned().unwrap();

        Tuple::new(row.schema.clone(), vec![id, value, category]).unwrap()
    });

    // Results should be identical
    assert_eq!(mapped_regular.len(), mapped_batched.len());

    // Verify values match
    for i in 0..10 {
        let reg_val = mapped_regular.rows[i].get("value");
        let batch_val = mapped_batched.rows[i].get("value");
        assert_eq!(reg_val, batch_val);
    }
}

#[test]
fn test_select_batched_correctness() {
    let dataset = create_large_dataset(5000);

    // Select using regular method
    let selected_regular = dataset.select(&["id", "category"]).unwrap();

    // Select using batched method
    let selected_batched = dataset.select_batched(&["id", "category"]).unwrap();

    // Results should be identical
    assert_eq!(selected_regular.len(), selected_batched.len());
    assert_eq!(selected_regular.schema.len(), 2);
    assert_eq!(selected_batched.schema.len(), 2);

    // Verify schema matches
    assert!(selected_regular.schema.get_field("id").is_some());
    assert!(selected_regular.schema.get_field("category").is_some());
    assert!(selected_batched.schema.get_field("id").is_some());
    assert!(selected_batched.schema.get_field("category").is_some());
}

#[test]
fn test_batches_iterator() {
    let dataset = create_large_dataset(2500);

    let mut batch_count = 0;
    let mut total_rows = 0;

    for batch in dataset.batches(1024) {
        batch_count += 1;
        total_rows += batch.len();
    }

    // Should have 3 batches: 1024 + 1024 + 452
    assert_eq!(batch_count, 3);
    assert_eq!(total_rows, 2500);
}

#[test]
fn test_empty_dataset_batched() {
    let schema = Arc::new(Schema::new(vec![Field::new("id", ValueType::Int)]));
    let dataset = Dataset::new(DatasetId(1), schema, None);

    let filtered = dataset.filter_batched(|_| true);
    assert_eq!(filtered.len(), 0);

    let mapped = dataset.map_batched(|row| row.clone());
    assert_eq!(mapped.len(), 0);
}
