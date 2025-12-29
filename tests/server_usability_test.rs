use linal::engine::TensorDb;
use linal::server::start_server;
use reqwest::Client;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::time::sleep;

async fn setup_server(port: u16) -> Arc<Mutex<TensorDb>> {
    let db = Arc::new(Mutex::new(TensorDb::new()));
    let db_clone = db.clone();
    tokio::spawn(async move {
        start_server(db_clone, port).await;
    });
    sleep(Duration::from_millis(1500)).await;
    db
}

#[tokio::test]
async fn test_database_lifecycle_api() {
    let port = 8201;
    let _db = setup_server(port).await;
    let client = Client::new();

    // 1. List databases (should have default)
    let resp = client
        .get(format!("http://localhost:{}/databases", port))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    let dbs = body["databases"].as_array().unwrap();
    assert!(dbs.iter().any(|d| d.as_str() == Some("default")));

    // 2. Create a new database
    let resp = client
        .post(format!("http://localhost:{}/databases/test_api_db", port))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);

    // 3. List again
    let resp = client
        .get(format!("http://localhost:{}/databases", port))
        .send()
        .await
        .unwrap();
    let body: serde_json::Value = resp.json().await.unwrap();
    let dbs = body["databases"].as_array().unwrap();
    assert!(dbs.iter().any(|d| d.as_str() == Some("test_api_db")));

    // 4. Delete the database
    let resp = client
        .delete(format!("http://localhost:{}/databases/test_api_db", port))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    // 5. Verify it's gone
    let resp = client
        .get(format!("http://localhost:{}/databases", port))
        .send()
        .await
        .unwrap();
    let body: serde_json::Value = resp.json().await.unwrap();
    let dbs = body["databases"].as_array().unwrap();
    assert!(!dbs.iter().any(|d| d.as_str() == Some("test_api_db")));
}

#[tokio::test]
async fn test_server_multitenancy() {
    let port = 8202;
    let _db = setup_server(port).await;
    let client = Client::new();

    // Create two databases
    client
        .post(format!("http://localhost:{}/databases/db_x", port))
        .send()
        .await
        .unwrap();
    client
        .post(format!("http://localhost:{}/databases/db_y", port))
        .send()
        .await
        .unwrap();

    // Define 'v' in db_x
    client
        .post(format!("http://localhost:{}/execute", port))
        .header("X-Linal-Database", "db_x")
        .header("Content-Type", "text/plain")
        .body("VECTOR v = [100]")
        .send()
        .await
        .unwrap();

    // Define 'v' in db_y as something else
    client
        .post(format!("http://localhost:{}/execute", port))
        .header("X-Linal-Database", "db_y")
        .header("Content-Type", "text/plain")
        .body("VECTOR v = [200]")
        .send()
        .await
        .unwrap();

    // Verify db_x has 100
    let resp_x = client
        .post(format!("http://localhost:{}/execute?format=json", port))
        .header("X-Linal-Database", "db_x")
        .header("Content-Type", "text/plain")
        .body("SHOW v")
        .send()
        .await
        .unwrap();
    let body_x: serde_json::Value = resp_x.json().await.unwrap();
    assert_eq!(body_x["result"]["Tensor"]["data"][0], 100.0);

    // Verify db_y has 200
    let resp_y = client
        .post(format!("http://localhost:{}/execute?format=json", port))
        .header("X-Linal-Database", "db_y")
        .header("Content-Type", "text/plain")
        .body("SHOW v")
        .send()
        .await
        .unwrap();
    let body_y: serde_json::Value = resp_y.json().await.unwrap();
    assert_eq!(body_y["result"]["Tensor"]["data"][0], 200.0);
}

#[tokio::test]
async fn test_server_scheduling() {
    let port = 8203;
    let _db = setup_server(port).await;
    let client = Client::new();

    // Create a schedule that runs every 1 second
    let resp = client
        .post(format!("http://localhost:{}/schedule", port))
        .json(&serde_json::json!({
            "name": "periodic_calc",
            "command": "VECTOR sched_v = [42]",
            "interval_secs": 1
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);
    let body: serde_json::Value = resp.json().await.unwrap();
    let task_id = body["id"].as_str().unwrap().to_string();

    // Wait for scheduler to run
    sleep(Duration::from_secs(3)).await;

    // Check if tensor was created
    let resp = client
        .post(format!("http://localhost:{}/execute?format=json", port))
        .header("Content-Type", "text/plain")
        .body("SHOW sched_v")
        .send()
        .await
        .unwrap();
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["status"], "ok");
    assert_eq!(body["result"]["Tensor"]["data"][0], 42.0);

    // Remove task
    let resp = client
        .delete(format!("http://localhost:{}/schedule/{}", port, task_id))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
}
