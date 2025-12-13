use vector_db_rs::engine::TensorDb;

#[test]
fn test_filter_optimization_workflow() {
    let mut db = TensorDb::new();

    let script = r#"
    DATASET users COLUMNS (id: Int, name: String, age: Int)
    
    CREATE INDEX name_idx ON users(name)

    INSERT INTO users VALUES (1, "Alice", 30)
    INSERT INTO users VALUES (2, "Bob", 25)
    INSERT INTO users VALUES (3, "Alice", 35)
    INSERT INTO users VALUES (4, "Charlie", 40)

    DATASET alice_users FROM users FILTER name = "Alice"
    SHOW alice_users
    "#;

    vector_db_rs::dsl::execute_script(&mut db, script).expect("Script execution failed");

    let alice_users = db.get_dataset("alice_users").expect("Dataset not found");
    assert_eq!(alice_users.len(), 2);

    // Check content (order might vary if index doesn't preserve it, but usually it might)
    // For now just check we got the right rows
    let ages: Vec<i64> = alice_users
        .rows
        .iter()
        .map(|r| match r.get("age") {
            Some(vector_db_rs::core::value::Value::Int(i)) => *i,
            _ => 0,
        })
        .collect();

    assert!(ages.contains(&30));
    assert!(ages.contains(&35));
}

#[test]
fn test_vector_search_workflow() {
    let mut db = TensorDb::new();

    let script = r#"
    DATASET vectors COLUMNS (id: Int, embedding: Vector(3))
    CREATE VECTOR INDEX vec_idx ON vectors(embedding)

    INSERT INTO vectors VALUES (1, [1.0, 0.0, 0.0])
    INSERT INTO vectors VALUES (2, [0.0, 1.0, 0.0])
    INSERT INTO vectors VALUES (3, [0.0, 0.0, 1.0])
    "#;

    vector_db_rs::dsl::execute_script(&mut db, script).expect("Script execution failed");

    // Search for nearest to [1.0, 0.1, 0.0] -> Should be id 1
    // SEARCH target FROM source QUERY vector ON column K=k
    let search_script = r#"
    SEARCH results FROM vectors QUERY [1.0, 0.1, 0.0] ON embedding K=1
    SHOW results
    "#;

    vector_db_rs::dsl::execute_script(&mut db, search_script)
        .expect("Search script execution failed");

    let results = db
        .get_dataset("results")
        .expect("Results dataset not found");
    assert_eq!(results.len(), 1);
    // Id 1 should be the result
    if let Some(vector_db_rs::core::value::Value::Int(id)) = results.rows[0].get("id") {
        assert_eq!(*id, 1);
    } else {
        panic!("Result row does not have id");
    }
}
