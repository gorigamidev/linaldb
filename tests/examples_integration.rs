// tests/examples_integration.rs
//
// Integration tests for example .lnl files
// These tests verify that the example scripts execute correctly end-to-end

use std::fs;
use linal::{execute_script, TensorDb};

#[test]
fn test_example_vdb_integration() {
    let mut db = TensorDb::new();
    
    // Load and execute the example.lnl file
    let script = fs::read_to_string("examples/example.lnl")
        .expect("Failed to read examples/example.lnl");
    
    // Execute the script - should not panic
    execute_script(&mut db, &script).expect("Example script execution failed");
    
    // Verify vectors were created
    let v1 = db.get("v1").expect("v1 should exist");
    assert_eq!(v1.shape.dims, vec![3]);
    assert_eq!(v1.data_ref(), vec![1.0, 2.0, 3.0]);
    
    let v2 = db.get("v2").expect("v2 should exist");
    assert_eq!(v2.data_ref(), vec![4.0, 5.0, 6.0]);
    
    // Verify vector addition result
    let v3 = db.get("v3").expect("v3 should exist");
    assert_eq!(v3.data_ref(), vec![5.0, 7.0, 9.0]);
    
    // Verify similarity was computed
    let sim = db.get("sim").expect("sim should exist");
    assert_eq!(sim.shape.dims, Vec::<usize>::new()); // Similarity returns a scalar (empty dims)
    
    // Verify matrices were created
    let m1 = db.get("m1").expect("m1 should exist");
    assert_eq!(m1.shape.dims, vec![2, 2]);
    assert_eq!(m1.data_ref(), vec![1.0, 2.0, 3.0, 4.0]);
    
    let m2 = db.get("m2").expect("m2 should exist");
    assert_eq!(m2.shape.dims, vec![2, 2]);
    
    // Verify matrix multiplication result
    let m3 = db.get("m3").expect("m3 should exist");
    assert_eq!(m3.shape.dims, vec![2, 2]);
    // m1 * m2 = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
    assert_eq!(m3.data_ref(), vec![19.0, 22.0, 43.0, 50.0]);
    
    // Verify transpose
    let m3_t = db.get("m3_t").expect("m3_t should exist");
    assert_eq!(m3_t.shape.dims, vec![2, 2]);
    assert_eq!(m3_t.data_ref(), vec![19.0, 43.0, 22.0, 50.0]);
    
    // Verify flatten
    let flat = db.get("flat").expect("flat should exist");
    assert_eq!(flat.shape.dims, vec![4]); // Flattened 2x2 matrix
    
    // Verify dataset was created
    let users = db.get_dataset("users").expect("users dataset should exist");
    assert_eq!(users.len(), 5);
    assert_eq!(users.schema.len(), 5);
    
    // Verify filtered datasets
    let active_users = db.get_dataset("active_users").expect("active_users should exist");
    assert_eq!(active_users.len(), 3); // Alice, Charlie, Eve
    
    let target_group = db.get_dataset("target_group").expect("target_group should exist");
    assert_eq!(target_group.len(), 3); // Alice(30), Charlie(35), Eve(32) - all > 28
    
    let final_list = db.get_dataset("final_list").expect("final_list should exist");
    assert_eq!(final_list.len(), 3);
    assert_eq!(final_list.schema.len(), 2); // name, score
}

#[test]
fn test_features_demo_vdb_integration() {
    let mut db = TensorDb::new();
    
    // Load and execute the features_demo.lnl file
    let script = fs::read_to_string("examples/features_demo.lnl")
        .expect("Failed to read examples/features_demo.lnl");
    
    // Execute the script - should not panic
    execute_script(&mut db, &script).expect("Features demo script execution failed");
    
    // Verify dataset was created with correct schema
    let analytics_data = db.get_dataset("analytics_data")
        .expect("analytics_data dataset should exist");
    assert_eq!(analytics_data.len(), 4);
    assert_eq!(analytics_data.schema.len(), 4);
    
    // Verify indices were created
    let indices = db.list_indices();
    assert!(indices.len() >= 2, "Should have at least 2 indices");
    
    let has_region_idx = indices.iter().any(|(ds, col, t)| 
        ds == "analytics_data" && col == "region" && t == "HASH");
    assert!(has_region_idx, "region_idx should exist");
    
    let has_emb_idx = indices.iter().any(|(ds, col, t)| 
        ds == "analytics_data" && col == "embedding" && t == "VECTOR");
    assert!(has_emb_idx, "emb_idx should exist");
}

