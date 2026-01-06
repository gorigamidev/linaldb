use axum::http::StatusCode;
use linal::engine::TensorDb;
use linal::server::start_server;
use std::sync::{Arc, RwLock};
use std::time::Duration;
use tokio::time::sleep;

#[tokio::test]
async fn test_toon_server_output() {
    // 1. Setup DB and start server in background
    let db = Arc::new(RwLock::new(TensorDb::new()));
    let port = 8095; // Use valid test port
    let db_clone = db.clone();

    // Spawn server
    tokio::spawn(async move {
        start_server(db_clone, port).await;
    });

    // Wait for server to be ready
    sleep(Duration::from_millis(1000)).await;

    // 2. Perform Request with raw DSL (new format)
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://localhost:{}/execute", port))
        .header("Content-Type", "text/plain")
        .body("VECTOR v = [1, 2, 3]")
        .send()
        .await
        .expect("Failed to send request");

    // 3. Assert Headers
    assert_eq!(resp.status(), 200);
    assert!(
        resp.headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap()
            .contains("text/toon"),
        "Content-Type should be text/toon"
    );

    // 4. Assert Body Content
    let body = resp.text().await.expect("Failed to get body");
    println!("Response Body:\n{}", body);

    // Simple checks for TOON structure
    assert!(body.contains("status: ok"));
    assert!(body.contains("result:"));
    assert!(body.contains("Message: \"Defined vector: v\""));
}

#[tokio::test]
async fn test_toon_dsl_output() {
    // 1. Setup DB and start server
    let db = Arc::new(RwLock::new(TensorDb::new()));
    let port = 8096;
    let db_clone = db.clone();

    tokio::spawn(async move {
        start_server(db_clone, port).await;
    });

    sleep(Duration::from_millis(1000)).await;

    // 2. Setup Data
    let client = reqwest::Client::new();
    // Create tensor first
    let resp_create = client
        .post(format!("http://localhost:{}/execute", port))
        .json(&serde_json::json!({
            "command": "MATRIX m = [[1, 2], [3, 4]]"
        }))
        .send()
        .await
        .unwrap();

    let create_body = resp_create.text().await.unwrap();
    println!("Create Response: {}", create_body);
    assert!(
        create_body.contains("status: ok"),
        "Creation failed: {}",
        create_body
    );

    // 3. Query it
    let resp = client
        .post(format!("http://localhost:{}/execute", port))
        .json(&serde_json::json!({
            "command": "SHOW m"
        }))
        .send()
        .await
        .unwrap();

    let body = resp.text().await.unwrap();
    println!("Matrix Body:\n{}", body);

    assert!(body.contains("Tensor:"));
    assert!(body.contains("shape:"));
    assert!(body.contains("dims[2]: 2,2")); // Check if TOON format is roughly as expected
                                            // Note: TOON format for arrays/dims might vary slightly based on toon-format crate version
                                            // but typically it's clean. Adjust assertion if failed.
}

#[tokio::test]
async fn test_json_backward_compatibility() {
    // Test that JSON format still works (with deprecation warning)
    let db = Arc::new(RwLock::new(TensorDb::new()));
    let port = 8097;
    let db_clone = db.clone();

    tokio::spawn(async move {
        start_server(db_clone, port).await;
    });

    sleep(Duration::from_millis(1000)).await;

    // Send request with JSON format (legacy)
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://localhost:{}/execute", port))
        .json(&serde_json::json!({
            "command": "VECTOR v = [1, 2, 3]"
        }))
        .send()
        .await
        .expect("Failed to send request");

    // Should still work
    assert_eq!(resp.status(), 200);
    let body = resp.text().await.expect("Failed to get body");
    println!("JSON Backward Compat Response:\n{}", body);

    assert!(body.contains("status: ok"));
    assert!(body.contains("Message: \"Defined vector: v\""));
}

#[tokio::test]
async fn test_json_format_response() {
    // Test JSON format via query parameter
    let db = Arc::new(RwLock::new(TensorDb::new()));
    let port = 8098;
    let db_clone = db.clone();

    tokio::spawn(async move {
        start_server(db_clone, port).await;
    });

    sleep(Duration::from_millis(1000)).await;

    // Request JSON format
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://localhost:{}/execute?format=json", port))
        .header("Content-Type", "text/plain")
        .body("VECTOR v = [1, 2, 3]")
        .send()
        .await
        .expect("Failed to send request");

    // Verify JSON response
    assert_eq!(resp.status(), 200);
    assert!(
        resp.headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap()
            .contains("application/json"),
        "Content-Type should be application/json"
    );

    let body = resp.text().await.expect("Failed to get body");
    println!("JSON Format Response:\n{}", body);

    // Parse as JSON
    let json: serde_json::Value = serde_json::from_str(&body).expect("Should be valid JSON");
    assert_eq!(json["status"], "ok");
    assert!(json["result"].is_object());
}

#[tokio::test]
async fn test_toon_format_explicit() {
    // Test explicit TOON format via query parameter
    let db = Arc::new(RwLock::new(TensorDb::new()));
    let port = 8099;
    let db_clone = db.clone();

    tokio::spawn(async move {
        start_server(db_clone, port).await;
    });

    sleep(Duration::from_millis(1000)).await;

    // Request TOON format explicitly
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://localhost:{}/execute?format=toon", port))
        .header("Content-Type", "text/plain")
        .body("VECTOR v = [1, 2, 3]")
        .send()
        .await
        .expect("Failed to send request");

    // Verify TOON response
    assert_eq!(resp.status(), 200);
    assert!(
        resp.headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap()
            .contains("text/toon"),
        "Content-Type should be text/toon"
    );

    let body = resp.text().await.expect("Failed to get body");
    println!("TOON Format Response:\n{}", body);

    assert!(body.contains("status: ok"));
    assert!(body.contains("Message: \"Defined vector: v\""));
}

#[tokio::test]
async fn test_invalid_format_defaults_to_toon() {
    // Test that invalid format defaults to TOON
    let db = Arc::new(RwLock::new(TensorDb::new()));
    let port = 8100;
    let db_clone = db.clone();

    tokio::spawn(async move {
        start_server(db_clone, port).await;
    });

    sleep(Duration::from_millis(1000)).await;

    // Request with invalid format
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://localhost:{}/execute?format=xml", port))
        .header("Content-Type", "text/plain")
        .body("VECTOR v = [1, 2, 3]")
        .send()
        .await
        .expect("Failed to send request");

    // Should default to TOON
    assert_eq!(resp.status(), 200);
    assert!(
        resp.headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap()
            .contains("text/toon"),
        "Content-Type should default to text/toon"
    );

    let body = resp.text().await.expect("Failed to get body");
    assert!(body.contains("status: ok"));
}
#[tokio::test]
async fn test_server_validation_empty() {
    let db = Arc::new(RwLock::new(TensorDb::new()));
    let port = 8101;
    let db_clone = db.clone();

    tokio::spawn(async move {
        start_server(db_clone, port).await;
    });

    sleep(Duration::from_millis(1000)).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://localhost:{}/execute", port))
        .header("Content-Type", "text/plain")
        .body("") // Empty body
        .send()
        .await
        .expect("Failed to send request");

    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = resp.text().await.unwrap();
    assert!(body.contains("Command cannot be empty"));
}

#[tokio::test]
async fn test_server_validation_length() {
    let db = Arc::new(RwLock::new(TensorDb::new()));
    let port = 8102;
    let db_clone = db.clone();

    tokio::spawn(async move {
        start_server(db_clone, port).await;
    });

    sleep(Duration::from_millis(1000)).await;

    let long_command = "a".repeat(16 * 1024 + 1); // 16KB + 1
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://localhost:{}/execute", port))
        .header("Content-Type", "text/plain")
        .body(long_command)
        .send()
        .await
        .expect("Failed to send request");

    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = resp.text().await.unwrap();
    assert!(body.contains("Command too long"));
}
