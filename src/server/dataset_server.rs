use crate::core::storage::ParquetStorage;
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::get,
    Json, Router,
};
use serde_json::Value;
use std::sync::Arc;

/// Lightweight read-only server for delivering datasets.
pub struct DatasetServer {
    pub storage: Arc<ParquetStorage>,
}

impl DatasetServer {
    pub fn new(storage: Arc<ParquetStorage>) -> Self {
        Self { storage }
    }

    pub fn router(self) -> Router {
        Router::new()
            .route("/datasets/:name/manifest.json", get(get_manifest))
            .route("/datasets/:name/schema.json", get(get_schema))
            .route("/datasets/:name/stats.json", get(get_stats))
            .route("/datasets/:name/data.parquet", get(get_data))
            .with_state(Arc::new(self))
    }
}

async fn get_manifest(
    Path(name): Path<String>,
    State(server): State<Arc<DatasetServer>>,
) -> impl IntoResponse {
    let path = format!(
        "{}/datasets/{}/manifest.json",
        server.storage.base_path(),
        name
    );
    match std::fs::read_to_string(path) {
        Ok(json) => (
            StatusCode::OK,
            Json(serde_json::from_str::<Value>(&json).unwrap()),
        ),
        Err(_) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Manifest not found"})),
        ),
    }
}

async fn get_schema(
    Path(name): Path<String>,
    State(server): State<Arc<DatasetServer>>,
) -> impl IntoResponse {
    let path = format!(
        "{}/datasets/{}/schema.json",
        server.storage.base_path(),
        name
    );
    match std::fs::read_to_string(path) {
        Ok(json) => (
            StatusCode::OK,
            Json(serde_json::from_str::<Value>(&json).unwrap()),
        ),
        Err(_) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Schema not found"})),
        ),
    }
}

async fn get_stats(
    Path(name): Path<String>,
    State(server): State<Arc<DatasetServer>>,
) -> impl IntoResponse {
    let path = format!(
        "{}/datasets/{}/stats.json",
        server.storage.base_path(),
        name
    );
    match std::fs::read_to_string(path) {
        Ok(json) => (
            StatusCode::OK,
            Json(serde_json::from_str::<Value>(&json).unwrap()),
        ),
        Err(_) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Stats not found"})),
        ),
    }
}

async fn get_data(
    Path(name): Path<String>,
    State(server): State<Arc<DatasetServer>>,
) -> impl IntoResponse {
    let path = format!(
        "{}/datasets/{}/data.parquet",
        server.storage.base_path(),
        name
    );
    match std::fs::read(path) {
        Ok(data) => (StatusCode::OK, data).into_response(),
        Err(_) => (StatusCode::NOT_FOUND, "Data not found").into_response(),
    }
}
