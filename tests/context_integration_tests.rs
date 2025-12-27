// Integration tests for ExecutionContext
// Tests the execute_with_context API and arena allocation

use linal::engine::context::ExecutionContext;
use linal::engine::TensorDb;

#[test]
fn test_execute_with_context_basic() {
    let mut db = TensorDb::new();
    let mut ctx = ExecutionContext::new();

    // Execute a simple vector creation with context
    let result = db.execute_with_context(&mut ctx, "VECTOR v1 = [1.0, 2.0, 3.0]");
    assert!(result.is_ok());

    // Verify the vector was created
    let tensor = db.get("v1");
    assert!(tensor.is_ok());
    let tensor = tensor.unwrap();
    assert_eq!(tensor.data_ref(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_execute_with_context_multiple_operations() {
    let mut db = TensorDb::new();
    let mut ctx = ExecutionContext::new();

    // Create two vectors
    db.execute_with_context(&mut ctx, "VECTOR v1 = [1.0, 2.0, 3.0]")
        .unwrap();
    db.execute_with_context(&mut ctx, "VECTOR v2 = [4.0, 5.0, 6.0]")
        .unwrap();

    // Add them
    db.execute_with_context(&mut ctx, "LET v3 = ADD v1 v2")
        .unwrap();

    // Verify result
    let result = db.get("v3").unwrap();
    assert_eq!(result.data_ref(), vec![5.0, 7.0, 9.0]);
}

#[test]
fn test_execute_with_context_matrix_operations() {
    let mut db = TensorDb::new();
    let mut ctx = ExecutionContext::new();

    // Create matrices
    db.execute_with_context(&mut ctx, "MATRIX m1 = [[1.0, 2.0], [3.0, 4.0]]")
        .unwrap();
    db.execute_with_context(&mut ctx, "MATRIX m2 = [[5.0, 6.0], [7.0, 8.0]]")
        .unwrap();

    // Matrix multiplication
    db.execute_with_context(&mut ctx, "LET m3 = MATMUL m1 m2")
        .unwrap();

    // Verify result exists
    assert!(db.get("m3").is_ok());
}

#[test]
fn test_execute_with_context_similarity() {
    let mut db = TensorDb::new();
    let mut ctx = ExecutionContext::new();

    // Create vectors
    db.execute_with_context(&mut ctx, "VECTOR v1 = [1.0, 0.0, 0.0]")
        .unwrap();
    db.execute_with_context(&mut ctx, "VECTOR v2 = [1.0, 0.0, 0.0]")
        .unwrap();

    // Compute similarity
    db.execute_with_context(&mut ctx, "LET sim = SIMILARITY v1 WITH v2")
        .unwrap();

    // Verify similarity is 1.0 (identical vectors)
    let result = db.get("sim").unwrap();
    assert!((result.data[0] - 1.0).abs() < 0.001);
}

#[test]
fn test_context_arena_stats() {
    let ctx = ExecutionContext::new();

    // Initial stats
    let stats_before = ctx.arena_stats();
    let initial_bytes = stats_before.allocated_bytes;

    // Allocate some temporary values
    let _val1 = ctx.alloc_temp(42);
    let _val2 = ctx.alloc_temp(3.14);
    let _slice = ctx.alloc_slice(&[1, 2, 3, 4, 5]);

    // Check that arena allocated memory
    let stats_after = ctx.arena_stats();
    assert!(stats_after.allocated_bytes > initial_bytes);
}

#[test]
fn test_context_with_dataset_operations() {
    let mut db = TensorDb::new();
    let mut ctx = ExecutionContext::new();

    // Create dataset
    db.execute_with_context(
        &mut ctx,
        "DATASET users COLUMNS (id: INT, name: STRING, score: FLOAT)",
    )
    .unwrap();

    // Insert data
    db.execute_with_context(&mut ctx, "INSERT INTO users VALUES (1, \"Alice\", 0.95)")
        .unwrap();
    db.execute_with_context(&mut ctx, "INSERT INTO users VALUES (2, \"Bob\", 0.87)")
        .unwrap();

    // Select data
    let result = db.execute_with_context(&mut ctx, "SELECT * FROM users");
    assert!(result.is_ok());
}

#[test]
fn test_backward_compatibility() {
    use linal::dsl::execute_line;

    let mut db = TensorDb::new();

    // Old API should still work
    let result = execute_line(&mut db, "VECTOR v1 = [1.0, 2.0, 3.0]", 1);
    assert!(result.is_ok());

    // New API should also work
    let mut ctx = ExecutionContext::new();
    let result = db.execute_with_context(&mut ctx, "VECTOR v2 = [4.0, 5.0, 6.0]");
    assert!(result.is_ok());

    // Both vectors should exist
    assert!(db.get("v1").is_ok());
    assert!(db.get("v2").is_ok());
}

#[test]
fn test_context_reuse() {
    let mut db = TensorDb::new();
    let mut ctx = ExecutionContext::new();

    // Use same context for multiple operations
    db.execute_with_context(&mut ctx, "VECTOR v1 = [1.0, 2.0]")
        .unwrap();
    db.execute_with_context(&mut ctx, "VECTOR v2 = [3.0, 4.0]")
        .unwrap();
    db.execute_with_context(&mut ctx, "LET v3 = ADD v1 v2")
        .unwrap();

    // All operations should succeed
    assert!(db.get("v1").is_ok());
    assert!(db.get("v2").is_ok());
    assert!(db.get("v3").is_ok());
}

#[test]
fn test_context_with_capacity() {
    let mut db = TensorDb::new();

    // Create context with specific capacity
    let mut ctx = ExecutionContext::with_capacity(1024);

    // Should work the same as default context
    db.execute_with_context(&mut ctx, "VECTOR v1 = [1.0, 2.0, 3.0]")
        .unwrap();
    assert!(db.get("v1").is_ok());
}
