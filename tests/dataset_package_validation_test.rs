use linal::core::dataset::{DatasetLineage, DatasetManifest, DatasetSchema, DatasetStats};
use linal::core::dataset_legacy::{Dataset, DatasetId};
use linal::core::storage::{ParquetStorage, StorageEngine};
use linal::core::tuple::{Field, Schema, Tuple};
use linal::core::value::{Value, ValueType};
use std::fs;
use std::path::Path;
use std::sync::Arc;

/// Helper to create a test dataset with diverse types
fn create_diverse_dataset(name: &str) -> Dataset {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", ValueType::Int),
        Field::new("label", ValueType::String).nullable(),
        Field::new("score", ValueType::Float).nullable(),
        Field::new("active", ValueType::Bool),
    ]));

    let mut dataset = Dataset::new(DatasetId(100), schema.clone(), Some(name.to_string()));

    let rows = vec![
        Tuple::new(
            schema.clone(),
            vec![
                Value::Int(1),
                Value::String("A".to_string()),
                Value::Float(0.5),
                Value::Bool(true),
            ],
        )
        .unwrap(),
        Tuple::new(
            schema.clone(),
            vec![
                Value::Int(2),
                Value::String("B".to_string()),
                Value::Float(1.5),
                Value::Bool(false),
            ],
        )
        .unwrap(),
        Tuple::new(
            schema.clone(),
            vec![Value::Int(3), Value::Null, Value::Null, Value::Bool(true)],
        )
        .unwrap(),
    ];

    dataset.rows = rows;
    dataset.metadata.update_stats(&schema, &dataset.rows);

    dataset
}

#[test]
fn test_dataset_package_json_contents() {
    let temp_dir = "/tmp/linal_test_package_validation";
    let _ = fs::remove_dir_all(temp_dir);

    let storage = ParquetStorage::new(temp_dir);
    let ds_name = "validation_test";
    let dataset = create_diverse_dataset(ds_name);

    // Save dataset package
    storage.save_dataset(&dataset).unwrap();

    let base_path = format!("{}/datasets/{}", temp_dir, ds_name);

    // 1. Validate manifest.json
    let manifest_path = format!("{}/manifest.json", base_path);
    assert!(Path::new(&manifest_path).exists());
    let manifest_content = fs::read_to_string(&manifest_path).unwrap();
    let manifest: DatasetManifest = serde_json::from_str(&manifest_content).unwrap();
    assert_eq!(manifest.name, ds_name);
    assert_eq!(manifest.version, "1.0");
    assert!(manifest.formats.contains_key("parquet"));
    assert_eq!(manifest.formats.get("parquet").unwrap(), "data.parquet");

    // 2. Validate schema.json
    let schema_path = format!("{}/schema.json", base_path);
    assert!(Path::new(&schema_path).exists());
    let schema_content = fs::read_to_string(&schema_path).unwrap();
    let schema: DatasetSchema = serde_json::from_str(&schema_content).unwrap();
    assert_eq!(schema.columns.len(), 4);

    let id_col = schema.get_column("id").expect("id column missing");
    assert_eq!(id_col.value_type, ValueType::Int);

    let label_col = schema.get_column("label").expect("label column missing");
    assert_eq!(label_col.value_type, ValueType::String);
    assert!(label_col.nullable);

    // 3. Validate stats.json
    let stats_path = format!("{}/stats.json", base_path);
    assert!(Path::new(&stats_path).exists());
    let stats_content = fs::read_to_string(&stats_path).unwrap();
    let stats: DatasetStats = serde_json::from_str(&stats_content).unwrap();
    assert_eq!(stats.row_count, 3);

    let label_stats = stats.columns.get("label").expect("label stats missing");
    assert_eq!(label_stats.null_count, 1);
    // Sparsity: 1 - (1/3) = 0.666...
    assert!((label_stats.sparsity.unwrap() - 0.666666).abs() < 1e-5);

    let score_stats = stats.columns.get("score").expect("score stats missing");
    assert_eq!(score_stats.null_count, 1);

    // 4. Validate lineage.json
    let lineage_path = format!("{}/lineage.json", base_path);
    assert!(Path::new(&lineage_path).exists());
    let lineage_content = fs::read_to_string(&lineage_path).unwrap();
    let lineage: DatasetLineage = serde_json::from_str(&lineage_content).unwrap();
    assert!(!lineage.nodes.is_empty());
    assert_eq!(lineage.nodes[0].dataset_name, ds_name);
    assert_eq!(lineage.nodes[0].operation, "SAVE (Legacy)");

    // Clean up
    let _ = fs::remove_dir_all(temp_dir);
}
