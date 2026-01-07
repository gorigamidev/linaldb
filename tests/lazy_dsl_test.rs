use linal::dsl::execute_script;
use linal::engine::TensorDb;

#[test]
fn test_lazy_dsl_flow() {
    let mut db = TensorDb::new();

    // 1. Define base tensors
    let script = "
        VECTOR a = [1.0, 2.0, 3.0]
        VECTOR b = [4.0, 5.0, 6.0]
        # Define a lazy addition
        LAZY LET c = ADD a b
        # Define a lazy multiplication on top of lazy addition
        LAZY LET d = SCALE c BY 2.0
    ";

    execute_script(&mut db, script).expect("Script execution failed");

    // 2. Verify variables are registered as lazy
    let names = db.list_names();
    assert!(names.contains(&"a".to_string()));
    assert!(names.contains(&"b".to_string()));
    assert!(names.contains(&"c".to_string()));
    assert!(names.contains(&"d".to_string()));

    // 3. SHOW should trigger evaluation
    // Since we can't easily capture output here without refactoring execute_script to return it,
    // we'll just check if the variables are still there and can be evaluated.
    let show_script = "SHOW d";
    execute_script(&mut db, show_script).expect("SHOW d failed");

    // After SHOW d, 'd' and its dependency 'c' should be materialized.
    // Actually, evaluate_expression recursively materializes,
    // but DatabaseInstance::evaluate only updates the target name in the name mapping.
    // Wait, let's check evaluation logic.
}
