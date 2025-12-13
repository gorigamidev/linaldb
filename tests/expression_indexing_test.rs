use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::time::sleep;
use vector_db_rs::engine::TensorDb;
use vector_db_rs::server::start_server;

#[tokio::test]
async fn test_expression_indexing() {
    let db = Arc::new(Mutex::new(TensorDb::new()));
    let port = 8110;
    let db_clone = db.clone();

    tokio::spawn(async move {
        start_server(db_clone, port).await;
    });
    sleep(Duration::from_millis(500)).await;

    let client = reqwest::Client::new();
    let url = format!("http://localhost:{}/execute", port);

    // Create a vector v1 = [10, 20, 30]
    client
        .post(&url)
        .json(&serde_json::json!({
            "command": "VECTOR v1 = [10, 20, 30]"
        }))
        .send()
        .await
        .unwrap();

    // Create a vector v2 = [1, 2, 3]
    client
        .post(&url)
        .json(&serde_json::json!({
            "command": "VECTOR v2 = [1, 2, 3]"
        }))
        .send()
        .await
        .unwrap();

    // Try calculation: v1[0] + v2[1]
    // 10 + 2 = 12
    let resp = client
        .post(&url)
        .json(&serde_json::json!({
            "command": "LET sum = v1[0] + v2[1]"
        }))
        .send()
        .await
        .unwrap();

    let body = resp.text().await.unwrap();
    println!("Response: {}", body);

    // Check if it succeeded
    assert!(
        body.contains("status: ok") || body.contains("Defined variable: sum"),
        "Expression failed: {}",
        body
    );

    // Verify result
    let resp_show = client
        .post(&url)
        .json(&serde_json::json!({
            "command": "SHOW sum"
        }))
        .send()
        .await
        .unwrap();

    let show_body = resp_show.text().await.unwrap();
    println!("SHOW sum: {}", show_body);
    assert!(show_body.contains("12"));
}
