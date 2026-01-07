use axum::{
    body::Body,
    http::{Request, StatusCode},
    Router,
};
use linal::core::dataset_legacy::{Dataset, DatasetId};
use linal::core::storage::{ParquetStorage, StorageEngine};
use linal::core::tuple::{Field, Schema, Tuple};
use linal::core::value::{Value, ValueType};
use linal::server::dataset_server::DatasetServer;
use std::fs;
use std::sync::Arc;
use tower::util::ServiceExt; // for `oneshot` and `ready`

/// Helper to create a test dataset
fn create_test_dataset(name: &str) -> Dataset {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", ValueType::Int),
        Field::new("val", ValueType::Float),
    ]));

    let mut dataset = Dataset::new(DatasetId(1), schema.clone(), Some(name.to_string()));

    let rows = vec![
        Tuple::new(schema.clone(), vec![Value::Int(1), Value::Float(1.1)]).unwrap(),
        Tuple::new(schema.clone(), vec![Value::Int(2), Value::Float(2.2)]).unwrap(),
    ];

    dataset.rows = rows;
    dataset.metadata.update_stats(&schema, &dataset.rows);
    dataset
}

#[tokio::test]
async fn test_dataset_delivery_http_endpoints() {
    let temp_dir = "/tmp/linal_test_delivery_http";
    let _ = fs::remove_dir_all(temp_dir);
    fs::create_dir_all(temp_dir).unwrap();

    let storage = Arc::new(ParquetStorage::new(temp_dir));
    let dataset = create_test_dataset("delivery_test");

    // 1. Save dataset (creates the package structure)
    storage.save_dataset(&dataset).unwrap();

    // 2. Setup DatasetServer
    let ds_server = DatasetServer::new(storage.clone());
    let app: Router = ds_server.router();

    // 3. Test Manifest Endpoint
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/datasets/delivery_test/manifest.json")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .unwrap();
    let body_str = String::from_utf8(body.to_vec()).unwrap();
    assert!(body_str.contains("\"name\":\"delivery_test\""));
    assert!(body_str.contains("\"version\":\"1.0\""));

    // 4. Test Schema Endpoint
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/datasets/delivery_test/schema.json")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .unwrap();
    let body_str = String::from_utf8(body.to_vec()).unwrap();

    // DatasetSchema structure has "columns"
    assert!(body_str.contains("\"columns\""));
    // ValueType::Int serializes to "Int"
    assert!(body_str.contains("\"value_type\":\"Int\""));
    assert!(body_str.contains("\"name\":\"id\""));

    // 5. Test Non-Existent Dataset
    let response = app
        .oneshot(
            Request::builder()
                .uri("/datasets/ghost/manifest.json")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);

    // Clean up
    let _ = fs::remove_dir_all(temp_dir);
}
