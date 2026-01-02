use criterion::{black_box, criterion_group, criterion_main, Criterion};
use linal::dsl::execute_line;
use linal::engine::db::TensorDb;

fn dataset_basics_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("dataset_ops");

    group.bench_function("dataset_creation", |b| {
        let mut db = TensorDb::new();
        b.iter(|| execute_line(&mut db, black_box("LET ds = dataset('bench_ds')"), 1).unwrap());
    });

    group.bench_function("add_column_metadata", |b| {
        let mut db = TensorDb::new();
        execute_line(&mut db, "VECTOR v1 = [1.0, 2.0, 3.0]", 1).unwrap();
        execute_line(&mut db, "LET ds = dataset('ds1')", 2).unwrap();

        b.iter(|| {
            // Note: We use a fresh column name each time if we want to avoid duplicate errors,
            // or we can allow it if the engine handles it fast.
            // Let's use a unique name in the loop if needed, but for metadata bench we just want the overhead.
            let _ = execute_line(&mut db, black_box("ds.add_column('new_col', v1)"), 3);
        });
    });

    group.finish();
}

fn column_access_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("column_access");
    let mut db = TensorDb::new();

    // Setup
    execute_line(&mut db, "VECTOR v1 = [1.0, 2.0, 3.0, 4.0, 5.0]", 1).unwrap();
    execute_line(&mut db, "LET ds = dataset('ds1')", 2).unwrap();
    execute_line(&mut db, "ds.add_column('col1', v1)", 3).unwrap();

    group.bench_function("direct_variable_access", |b| {
        b.iter(|| execute_line(&mut db, black_box("LET x = v1 * 2.0"), 4).unwrap());
    });

    group.bench_function("dataset_column_access", |b| {
        b.iter(|| execute_line(&mut db, black_box("LET x = ds.col1 * 2.0"), 4).unwrap());
    });

    group.finish();
}

fn materialization_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("materialization");

    for size in [128, 512, 4096].iter() {
        group.bench_with_input(
            criterion::BenchmarkId::from_parameter(size),
            size,
            |b, &size| {
                let mut db = TensorDb::new();
                let values: Vec<String> = (0..size).map(|i| format!("{}.0", i)).collect();
                let v_def = format!("VECTOR v1 = [{}]", values.join(", "));
                execute_line(&mut db, &v_def, 1).unwrap();
                execute_line(&mut db, "LET ds = dataset('ds1')", 2).unwrap();
                execute_line(&mut db, "ds.add_column('col1', v1)", 3).unwrap();

                b.iter(|| {
                    // Internal materialization call
                    let _ = db.materialize_tensor_dataset("ds1").unwrap();
                });
            },
        );
    }
    group.finish();
}

fn batched_operations_benchmark(c: &mut Criterion) {
    use linal::core::dataset_legacy::{Dataset, DatasetId};
    use linal::core::tuple::{Field, Schema, Tuple};
    use linal::core::value::{Value, ValueType};
    use std::sync::Arc;

    let mut group = c.benchmark_group("batched_ops");

    // Create datasets of different sizes
    for size in [1_000, 10_000, 50_000].iter() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", ValueType::Int),
            Field::new("value", ValueType::Float),
        ]));

        let mut rows = Vec::with_capacity(*size);
        for i in 0..*size {
            rows.push(
                Tuple::new(
                    schema.clone(),
                    vec![Value::Int(i as i64), Value::Float((i as f32) * 1.5)],
                )
                .unwrap(),
            );
        }

        let dataset = Dataset::with_rows(
            DatasetId(1),
            schema.clone(),
            rows,
            Some("bench".to_string()),
        )
        .unwrap();

        // Benchmark filter
        group.bench_with_input(
            criterion::BenchmarkId::new("filter_regular", size),
            &dataset,
            |b, ds| {
                b.iter(|| {
                    ds.filter(|row| {
                        if let Some(Value::Int(id)) = row.get("id") {
                            *id % 2 == 0
                        } else {
                            false
                        }
                    })
                });
            },
        );

        group.bench_with_input(
            criterion::BenchmarkId::new("filter_batched", size),
            &dataset,
            |b, ds| {
                b.iter(|| {
                    ds.filter_batched(|row| {
                        if let Some(Value::Int(id)) = row.get("id") {
                            *id % 2 == 0
                        } else {
                            false
                        }
                    })
                });
            },
        );

        // Benchmark map
        group.bench_with_input(
            criterion::BenchmarkId::new("map_regular", size),
            &dataset,
            |b, ds| {
                b.iter(|| {
                    ds.map(|row| {
                        let id = row.get("id").cloned().unwrap();
                        let value = if let Some(Value::Float(v)) = row.get("value") {
                            Value::Float(v * 2.0)
                        } else {
                            Value::Float(0.0)
                        };
                        Tuple::new(row.schema.clone(), vec![id, value]).unwrap()
                    })
                });
            },
        );

        group.bench_with_input(
            criterion::BenchmarkId::new("map_batched", size),
            &dataset,
            |b, ds| {
                b.iter(|| {
                    ds.map_batched(|row| {
                        let id = row.get("id").cloned().unwrap();
                        let value = if let Some(Value::Float(v)) = row.get("value") {
                            Value::Float(v * 2.0)
                        } else {
                            Value::Float(0.0)
                        };
                        Tuple::new(row.schema.clone(), vec![id, value]).unwrap()
                    })
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    dataset_basics_benchmark,
    column_access_benchmark,
    materialization_benchmark,
    batched_operations_benchmark
);
criterion_main!(benches);
