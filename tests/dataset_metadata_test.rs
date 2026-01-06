use linal::dsl::{execute_line, DslOutput};
use linal::engine::TensorDb;
use std::fs;

#[test]
fn test_dataset_metadata_persistence() {
    let mut db = TensorDb::new();

    // 1. Create a dataset
    execute_line(
        &mut db,
        "DATASET test_ds COLUMNS (id: Int, name: String)",
        1,
    )
    .unwrap();
    execute_line(&mut db, "INSERT INTO test_ds VALUES (1, \"Alice\")", 2).unwrap();
    execute_line(&mut db, "INSERT INTO test_ds VALUES (2, \"Bob\")", 3).unwrap();

    // 2. Save dataset (should auto-create metadata)
    let save_output = execute_line(
        &mut db,
        "SAVE DATASET test_ds TO \"./data/test_metadata\"",
        4,
    )
    .unwrap();

    match save_output {
        DslOutput::Message(msg) => {
            // Just check that it contains the dataset name and mentions version
            assert!(msg.contains("test_ds"));
            assert!(msg.contains("Saved dataset"));
        }
        _ => panic!("Expected Message output"),
    }

    // 3. Verify metadata files exist
    // Path should be resolved to: data_dir/default/data/test_metadata/datasets/test_ds.meta.json
    // Because path "./data/test_metadata" is pushed to "./data/default"
    let legacy_metadata_path = "./data/default/data/test_metadata/datasets/test_ds.meta.json";
    let new_metadata_path = "./data/default/data/test_metadata/datasets/test_ds.metadata.json";

    assert!(
        fs::metadata(legacy_metadata_path).is_ok(),
        "Legacy metadata file (.meta.json) should exist"
    );
    assert!(
        fs::metadata(new_metadata_path).is_ok(),
        "New metadata file (.metadata.json) should exist"
    );

    // 4. Show metadata
    let show_output = execute_line(&mut db, "SHOW DATASET METADATA test_ds", 5).unwrap();

    match show_output {
        DslOutput::Message(msg) => {
            assert!(msg.contains("Version: 1"));
            assert!(msg.contains("Origin: Created"));
        }
        _ => panic!("Expected Message output"),
    }

    // Cleanup
    let _ = fs::remove_dir_all("./data/default/data/test_metadata");
}

#[test]
fn test_metadata_without_save() {
    let mut db = TensorDb::new();

    // Try to show metadata for dataset that was never saved
    let output = execute_line(&mut db, "SHOW DATASET METADATA nonexistent", 1).unwrap();

    match output {
        DslOutput::Message(msg) => {
            assert!(msg.contains("No metadata found"));
        }
        _ => panic!("Expected Message output"),
    }
}

#[test]
fn test_metadata_serialization() {
    use linal::core::dataset::{DatasetMetadata, DatasetOrigin};

    let metadata = DatasetMetadata::new(
        "test".to_string(),
        DatasetOrigin::Imported {
            source: "file.csv".to_string(),
        },
    );

    // Serialize to JSON
    let json = serde_json::to_string(&metadata).unwrap();

    // Deserialize back
    let deserialized: DatasetMetadata = serde_json::from_str(&json).unwrap();

    assert_eq!(deserialized.name, "test");
    assert_eq!(deserialized.version, 1);
    assert!(matches!(
        deserialized.origin,
        DatasetOrigin::Imported { .. }
    ));
}
