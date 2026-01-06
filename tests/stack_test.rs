use linal::engine::TensorDb;
use linal::server::start_server;
use std::sync::{Arc, RwLock};
use std::time::Duration;
use tokio::time::sleep;

#[tokio::test]
async fn test_tensor_stacking() {
    let db = Arc::new(RwLock::new(TensorDb::new()));
    let port = 8097;
    let db_clone = db.clone();

    tokio::spawn(async move {
        start_server(db_clone, port).await;
    });
    sleep(Duration::from_millis(500)).await;

    let client = reqwest::Client::new();
    let url = format!("http://localhost:{}/execute", port);

    // 1. Define vectors
    let reqs = vec!["VECTOR v1 = [1, 2]", "VECTOR v2 = [3, 4]"];
    for cmd in reqs {
        client
            .post(&url)
            .json(&serde_json::json!({"command": cmd}))
            .send()
            .await
            .unwrap();
    }

    // 2. Stack them
    let resp = client
        .post(&url)
        .json(&serde_json::json!({
            "command": "LET matrix = STACK v1 v2"
        }))
        .send()
        .await
        .unwrap();

    let body = resp.text().await.unwrap();
    assert!(body.contains("status: ok"), "STACK failed: {}", body);

    // 3. Verify result
    let resp_show = client
        .post(&url)
        .json(&serde_json::json!({
            "command": "SHOW matrix"
        }))
        .send()
        .await
        .unwrap();

    let show_body = resp_show.text().await.unwrap();
    println!("SHOW output: {}", show_body);

    // Expect rank 2, dims [2, 2]
    assert!(show_body.contains("dims[2]: 2,2"));
}
