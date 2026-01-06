use linal::dsl::{execute_line, DslOutput};
use linal::engine::TensorDb;
use std::fs;

#[test]
fn test_csv_import_export() {
    let mut db = TensorDb::new();

    // 1. Create a dummy CSV file in a known location
    let temp_dir = std::env::temp_dir();
    let csv_path_buf = temp_dir.join("test_data.csv");
    let csv_path = csv_path_buf.to_str().unwrap();
    let csv_content = "id,name,value\n1,alpha,10.5\n2,beta,20.0\n3,gamma,30.5\n";
    fs::write(csv_path, csv_content).unwrap();

    // 2. Import CSV
    let import_cmd = format!("IMPORT CSV FROM \"{}\" AS test_ds", csv_path);
    let out = execute_line(&mut db, &import_cmd, 1).expect("Failed to execute IMPORT CSV");

    match out {
        DslOutput::Message(msg) => {
            assert!(msg.contains("Imported 3 rows"));
        }
        _ => panic!("Expected Message output, got {:?}", out),
    }

    // 3. Verify it exists
    let select_out = execute_line(&mut db, "SELECT * FROM test_ds", 2).expect("Failed to SELECT");
    match select_out {
        DslOutput::Table(table) => {
            assert_eq!(table.rows.len(), 3);
            assert_eq!(table.schema.fields.len(), 3);
            assert_eq!(table.schema.fields[0].name, "id");
            assert_eq!(table.schema.fields[1].name, "name");
            assert_eq!(table.schema.fields[2].name, "value");
        }
        _ => panic!("Expected Table output, got {:?}", select_out),
    }

    // 4. Export CSV
    let export_path_buf = temp_dir.join("test_export.csv");
    let export_path = export_path_buf.to_str().unwrap();
    let export_cmd = format!("EXPORT CSV test_ds TO \"{}\"", export_path);
    execute_line(&mut db, &export_cmd, 3).expect("Failed to execute EXPORT CSV");

    assert!(fs::metadata(export_path).is_ok());

    let exported_content = fs::read_to_string(export_path).unwrap();
    assert!(exported_content.contains("alpha"));
    assert!(exported_content.contains("10.5"));

    // Cleanup
    let _ = fs::remove_file(csv_path);
    let _ = fs::remove_file(export_path);
}

#[test]
fn test_session_reset() {
    let mut db = TensorDb::new();

    // 1. Define something
    execute_line(&mut db, "VECTOR v = [1, 2, 3]", 1).unwrap();
    assert!(db.active_instance().get("v").is_ok());

    // 2. Reset session
    execute_line(&mut db, "RESET SESSION", 2).unwrap();

    // 3. Verify it's gone
    assert!(db.active_instance().get("v").is_err());
}
