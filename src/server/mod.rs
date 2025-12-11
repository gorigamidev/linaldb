use crate::dsl::{execute_line, DslOutput};
use crate::engine::TensorDb;
use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use toon_format::encode_default;

struct AppState {
    db: Arc<Mutex<TensorDb>>,
}

#[derive(Deserialize)]
pub struct ExecuteRequest {
    command: String,
}

#[derive(Serialize)]
pub struct ExecuteResponse {
    status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<DslOutput>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

pub async fn start_server(db: Arc<Mutex<TensorDb>>, port: u16) {
    let state = Arc::new(AppState { db });

    let app = Router::new()
        .route("/health", get(health_check))
        .route("/execute", post(execute_command))
        .with_state(state);

    let addr = format!("0.0.0.0:{}", port);
    println!("Server running at http://{}", addr);
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn health_check() -> (StatusCode, Json<serde_json::Value>) {
    (StatusCode::OK, Json(serde_json::json!({ "status": "ok" })))
}

async fn execute_command(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<ExecuteRequest>,
) -> impl IntoResponse {
    // We lock the DB for the duration of execution.
    // In a real DB we'd want finer grained locking or MVCC.
    let mut db = state.db.lock().unwrap();
    let response = match execute_line(&mut db, &payload.command, 1) {
        Ok(output) => {
            // If output is Message, effectively it's the result.
            // If output is None, result is None.
            // If output is Table/Tensor, it's result.
            let result = match output {
                DslOutput::None => None,
                _ => Some(output),
            };
            ExecuteResponse {
                status: "ok".to_string(),
                result,
                error: None,
            }
        }
        Err(e) => ExecuteResponse {
            status: "error".to_string(),
            result: None,
            error: Some(format!("{}", e)),
        },
    };

    let body = encode_default(&response)
        .unwrap_or_else(|e| format!("status: error\nerror: Serialization failed: {}", e));

    (
        StatusCode::OK,
        [(axum::http::header::CONTENT_TYPE, "text/toon")],
        body,
    )
}
