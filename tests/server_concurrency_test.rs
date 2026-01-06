use linal::engine::TensorDb;
use linal::server::start_server;
use std::sync::{Arc, RwLock};
use std::time::Duration;
use tokio::time::sleep;

#[tokio::test]
async fn test_server_parallel_read_concurrency() {
    // 1. Setup DB and start server
    let db = Arc::new(RwLock::new(TensorDb::new()));
    let port = 8200;
    let db_clone = db.clone();

    tokio::spawn(async move {
        start_server(db_clone, port).await;
    });

    sleep(Duration::from_millis(1000)).await;

    let client = reqwest::Client::new();
    let url = format!("http://localhost:{}/execute", port);

    // 2. Prepare some data
    client
        .post(&url)
        .body("MATRIX m = [[1, 2], [3, 4]]")
        .send()
        .await
        .unwrap();

    // 3. Launch a "long" read operation as a job (if we have one).
    // For now, let's just use regular /execute for multiple readers.
    // We want to prove that two SHOW commands can run even if one is "slow".
    // Since we don't have a SLEEP command in DSL, we'll just launch 10 simultaneous READs
    // and see if they complete quickly.

    let mut handles = vec![];
    for _ in 0..10 {
        let client_clone = client.clone();
        let url_clone = url.clone();
        handles.push(tokio::spawn(async move {
            let start = std::time::Instant::now();
            let resp = client_clone
                .post(url_clone)
                .body("SHOW m")
                .send()
                .await
                .unwrap();
            let elapsed = start.elapsed();
            (resp.status(), elapsed)
        }));
    }

    for handle in handles {
        let (status, elapsed) = handle.await.unwrap();
        assert_eq!(status, 200);
        println!("Concurrent SHOW m took: {:?}", elapsed);
    }
}

#[tokio::test]
async fn test_read_during_background_job() {
    // 1. Setup DB and start server
    let db = Arc::new(RwLock::new(TensorDb::new()));
    let port = 8201;
    let db_clone = db.clone();

    tokio::spawn(async move {
        start_server(db_clone, port).await;
    });

    sleep(Duration::from_millis(1000)).await;

    let client = reqwest::Client::new();
    let base_url = format!("http://localhost:{}", port);

    // 2. Submit a long-ish job (e.g., a query that produces some output)
    let job_resp = client
        .post(format!("{}/jobs", base_url))
        .body("MATRIX big = [[1, 2], [3, 4]]") // Not really long, but it's a job
        .send()
        .await
        .unwrap();

    let job_json: serde_json::Value = job_resp.json().await.unwrap();
    let _job_id = job_json["job_id"].as_str().unwrap();

    // 3. Immediately while job is (likely) running, do a SHOW ALL
    let exec_resp = client
        .post(format!("{}/execute", base_url))
        .body("SHOW ALL")
        .send()
        .await
        .unwrap();

    assert_eq!(exec_resp.status(), 200);
    let body = exec_resp.text().await.unwrap();
    assert!(body.contains("status: ok"));

    // Cleanup: wait for job
    sleep(Duration::from_millis(500)).await;
}
