use linal::core::tensor::Shape;
use linal::dsl::execute_script;
use linal::engine::{context::ExecutionContext, TensorDb};

#[test]
fn test_lineage_tracking_basic() {
    let mut db = TensorDb::new();
    let mut ctx = ExecutionContext::new();
    let exec_id = ctx.execution_id();

    // 1. Create base tensor
    db.insert_named("a", Shape::new(vec![3]), vec![1.0, 2.0, 3.0])
        .unwrap();
    let a_id = db.active_instance().get_tensor_id("a").unwrap();

    // 2. Perform operation
    let script = "LET b = a * 2.0";
    db.execute_with_context(&mut ctx, script).unwrap();

    // 3. Verify lineage
    let b = db.get("b").expect("b should exist");
    let lineage = b.metadata.lineage.as_ref().expect("b should have lineage");

    assert_eq!(lineage.execution_id, exec_id);
    assert!(lineage.operation.contains("MULTIPLY") || lineage.operation.contains("SCALE"));
    assert_eq!(lineage.inputs.len(), 2);
    assert!(lineage.inputs.contains(&a_id));
}

#[test]
fn test_lineage_transitive() {
    let mut db = TensorDb::new();
    let mut ctx = ExecutionContext::new();
    let exec_id = ctx.execution_id();

    // a -> b -> c
    db.insert_named("a", Shape::new(vec![2]), vec![1.0, 2.0])
        .unwrap();
    let a_id = db.active_instance().get_tensor_id("a").unwrap();

    db.execute_with_context(&mut ctx, "LET b = a + 1.0")
        .unwrap();
    let b_id = db.active_instance().get_tensor_id("b").unwrap();

    db.execute_with_context(&mut ctx, "LET c = b * 3.0")
        .unwrap();

    let c = db.get("c").unwrap();
    let lineage_c = c.metadata.lineage.as_ref().unwrap();

    assert_eq!(lineage_c.execution_id, exec_id);
    assert_eq!(lineage_c.inputs.len(), 2);
    assert!(lineage_c.inputs.contains(&b_id));

    // Check b's lineage too
    let b = db.get("b").unwrap();
    let lineage_b = b.metadata.lineage.as_ref().unwrap();
    assert_eq!(lineage_b.inputs.len(), 2);
    assert!(lineage_b.inputs.contains(&a_id));
}

#[test]
fn test_lineage_binary_op() {
    let mut db = TensorDb::new();
    let mut ctx = ExecutionContext::new();

    db.insert_named("a", Shape::new(vec![2]), vec![1.0, 2.0])
        .unwrap();
    db.insert_named("b", Shape::new(vec![2]), vec![3.0, 4.0])
        .unwrap();
    let a_id = db.active_instance().get_tensor_id("a").unwrap();
    let b_id = db.active_instance().get_tensor_id("b").unwrap();

    db.execute_with_context(&mut ctx, "LET c = a + b").unwrap();

    let c = db.get("c").unwrap();
    let lineage = c.metadata.lineage.as_ref().unwrap();
    assert_eq!(lineage.inputs.len(), 2);
    assert!(lineage.inputs.contains(&a_id));
    assert!(lineage.inputs.contains(&b_id));
}

#[test]
fn test_lineage_persistence() {
    use std::fs;
    let mut db = TensorDb::new();
    let mut ctx = ExecutionContext::new();

    let temp_dir = "temp_lineage_test";
    let _ = fs::remove_dir_all(temp_dir);
    fs::create_dir_all(temp_dir).unwrap();

    db.insert_named("a", Shape::new(vec![2]), vec![1.0, 2.0])
        .unwrap();
    db.execute_with_context(&mut ctx, "LET b = a * 2.0")
        .unwrap();

    let b_before = db.get("b").unwrap();
    let lineage_before = b_before.metadata.lineage.as_ref().unwrap().clone();

    // Save using DSL
    let save_path = format!("{}/persistence_test", temp_dir);
    execute_script(&mut db, &format!("SAVE TENSOR b TO \"{}\"", save_path)).unwrap();

    // Load into new DB using DSL
    let mut db2 = TensorDb::new();
    execute_script(&mut db2, &format!("LOAD TENSOR b FROM \"{}\"", save_path)).unwrap();

    let b_after = db2.get("b").expect("b should exist");
    let lineage_after = b_after
        .metadata
        .lineage
        .as_ref()
        .expect("Lineage should be persisted");

    assert_eq!(lineage_after.execution_id, lineage_before.execution_id);
    assert_eq!(lineage_after.operation, lineage_before.operation);
    assert_eq!(lineage_after.inputs, lineage_before.inputs);

    let _ = fs::remove_dir_all(temp_dir);
}
