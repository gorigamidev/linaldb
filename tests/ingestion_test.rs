use linal::dsl::{execute_line, DslOutput};
use linal::engine::TensorDb;
use std::fs;

#[test]
fn test_use_dataset_from_csv() {
    let mut db = TensorDb::new();

    // 1. Create a dummy CSV file
    let temp_dir = std::env::temp_dir();
    let csv_path_buf = temp_dir.join("test_ingestion_use.csv");
    let csv_path = csv_path_buf.to_str().unwrap();
    let csv_content = "val1,val2\n1.0,2.0\n3.0,4.0\n";
    fs::write(csv_path, csv_content).unwrap();

    // 2. Test USE DATASET FROM
    let use_cmd = format!(r#"USE DATASET FROM "{}" AS use_ds"#, csv_path);
    let out = execute_line(&mut db, &use_cmd, 1).expect("Failed to execute USE DATASET FROM");

    // Verify it returns a table (materialized)
    match out {
        DslOutput::Table(table) => {
            assert_eq!(table.rows.len(), 2);
            assert_eq!(table.schema.fields.len(), 2);
            assert_eq!(table.schema.fields[0].name, "val1");
            assert_eq!(table.schema.fields[1].name, "val2");
        }
        _ => panic!("Expected Table output, got {:?}", out),
    }

    // Verify tensors are registered in store
    let names = db.active_instance().list_names();
    assert!(names.contains(&"use_ds_val1".to_string()));
    assert!(names.contains(&"use_ds_val2".to_string()));

    // Cleanup
    let _ = fs::remove_file(csv_path);
}

#[test]
fn test_import_dataset_from_csv() {
    let mut db = TensorDb::new();

    // 1. Create a dummy CSV file
    let temp_dir = std::env::temp_dir();
    let csv_path_buf = temp_dir.join("test_ingestion_import.csv");
    let csv_path = csv_path_buf.to_str().unwrap();
    let csv_content = "val1,val2\n10.0,20.0\n";
    fs::write(csv_path, csv_content).unwrap();

    // 2. Test IMPORT DATASET FROM
    let import_cmd = format!(r#"IMPORT DATASET FROM "{}" AS import_ds"#, csv_path);
    let out = execute_line(&mut db, &import_cmd, 1).expect("Failed to execute IMPORT DATASET FROM");

    match out {
        DslOutput::Message(msg) => {
            assert!(msg.contains("Imported dataset 'import_ds'"));
        }
        _ => panic!("Expected Message output, got {:?}", out),
    }

    // Cleanup
    let _ = fs::remove_file(csv_path);
}
