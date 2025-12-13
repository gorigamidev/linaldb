// use std::sync::Arc;
use vector_db_rs::core::value::Value;
use vector_db_rs::engine::TensorDb;
use vector_db_rs::query::logical::{Expr, LogicalPlan};
use vector_db_rs::query::planner::Planner;

#[test]
fn test_planner_index_selection() {
    let mut db = TensorDb::new();
    let script = r#"
    DATASET users COLUMNS (id: Int, name: String)
    CREATE INDEX name_idx ON users(name)
    INSERT INTO users VALUES (1, "Alice")
    INSERT INTO users VALUES (2, "Bob")
    "#;
    vector_db_rs::dsl::execute_script(&mut db, script).expect("Setup failed");

    let dataset = db.get_dataset("users").expect("Dataset users not found");
    let schema = dataset.schema.clone();

    // Logical Plan: Filter(name = "Alice") -> Scan(users)
    let scan = LogicalPlan::Scan {
        dataset_name: "users".to_string(),
        schema: schema.clone(),
    };
    let filter = LogicalPlan::Filter {
        input: Box::new(scan),
        predicate: Expr::BinaryExpr {
            left: Box::new(Expr::Column("name".to_string())),
            op: "=".to_string(),
            right: Box::new(Expr::Literal(Value::String("Alice".to_string()))),
        },
    };

    let planner = Planner::new(&db);
    let physical_plan = planner
        .create_physical_plan(&filter)
        .expect("Plan creation failed");

    // We verify optimizations by name type check or just by execution?
    // In Rust without reflection, it's hard to check "is this struct IndexScanExec?".
    // But we can check results.
    let results = physical_plan.execute(&db).expect("Execution failed");
    assert_eq!(results.len(), 1);
    if let Some(Value::String(name)) = results[0].get("name") {
        assert_eq!(name, "Alice");
    } else {
        panic!("Wrong result");
    }
}

#[test]
fn test_explain_command() {
    let mut db = TensorDb::new();
    let script = r#"
    DATASET users COLUMNS (id: Int, name: String)
    CREATE INDEX name_idx ON users(name)
    "#;
    vector_db_rs::dsl::execute_script(&mut db, script).expect("Setup failed");

    let explain_cmd = r#"EXPLAIN DATASET users_filtered FROM users FILTER name = "Bob""#;
    let output = vector_db_rs::dsl::execute_line(&mut db, explain_cmd, 0);

    match output {
        Ok(vector_db_rs::dsl::DslOutput::Message(msg)) => {
            assert!(msg.contains("--- Logical Plan ---"));
            assert!(msg.contains("--- Physical Plan ---"));
            // println!("{}", msg); // Can't print in test unless failed or --nocapture
        }
        _ => panic!("Expected message output from EXPLAIN, got {:?}", output),
    }

    // Verify SEARCH explain too
    let _explain_search = r#"EXPLAIN SEARCH res FROM users QUERY [1.0, 2.0] ON name K=1"#;
    // Note: SEARCH requires Vector(N) column and index. Our dataset has name: String.
    // It should fail gracefully or we should setup proper dataset.
    // Let's setup proper dataset for search test.
    let setup_vector = r#"
    DATASET vectors COLUMNS (id: Int, v: Vector(2))
    CREATE VECTOR INDEX v_idx ON vectors(v)
    "#;
    vector_db_rs::dsl::execute_script(&mut db, setup_vector).expect("Vector setup failed");

    let explain_search_cmd = r#"EXPLAIN SEARCH res FROM vectors QUERY [1.0, 1.0] ON v K=5"#;
    let output_search = vector_db_rs::dsl::execute_line(&mut db, explain_search_cmd, 0);

    match output_search {
        Ok(vector_db_rs::dsl::DslOutput::Message(msg)) => {
            assert!(msg.contains("VectorSearch"));
        }
        _ => panic!(
            "Expected message output from EXPLAIN SEARCH, got {:?}",
            output_search
        ),
    }
}
