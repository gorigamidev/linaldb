use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::time::sleep;
use vector_db_rs::engine::TensorDb;
use vector_db_rs::server::start_server;

#[tokio::test]
async fn test_toon_server_output() {
    // 1. Setup DB and start server in background
    let db = Arc::new(Mutex::new(TensorDb::new()));
    let port = 8095; // Use valid test port
    let db_clone = db.clone();

    // Spawn server
    tokio::spawn(async move {
        start_server(db_clone, port).await;
    });

    // Wait for server to be ready
    sleep(Duration::from_millis(1000)).await;

    // 2. Perform Request
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://localhost:{}/execute", port))
        .json(&serde_json::json!({
            "command": "VECTOR v = [1, 2, 3]"
        }))
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
    let db = Arc::new(Mutex::new(TensorDb::new()));
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
