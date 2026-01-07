// Phase 0 Baseline Benchmarks - Query Operations
// Using correct DSL syntax from examples
// Enhanced with batch operations

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use linal::dsl::execute_line;
use linal::engine::db::TensorDb;

fn dataset_creation_benchmark(c: &mut Criterion) {
    c.bench_function("create_dataset", |bench| {
        bench.iter_batched(
            TensorDb::new,
            |mut db| {
                execute_line(&mut db, black_box("DATASET users COLUMNS (id: INT, name: STRING, age: INT, active: BOOL, score: FLOAT)"), 1).unwrap()
            },
            criterion::BatchSize::SmallInput
        );
    });
}

fn insert_benchmark(c: &mut Criterion) {
    c.bench_function("single_insert", |bench| {
        bench.iter_batched(
            || {
                let mut db = TensorDb::new();
                execute_line(&mut db, "DATASET users COLUMNS (id: INT, name: STRING, age: INT, active: BOOL, score: FLOAT)", 1).unwrap();
                db
            },
            |mut db| {
                execute_line(&mut db, black_box("INSERT INTO users VALUES (1, \"Alice\", 30, true, 0.95)"), 1).unwrap()
            },
            criterion::BatchSize::SmallInput
        );
    });
}

// Batch insert benchmarks at different sizes
fn batch_insert_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_insert");

    for batch_size in [10, 100, 1000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(batch_size), batch_size, |bench, &batch_size| {
            bench.iter_batched(
                || {
                    let mut db = TensorDb::new();
                    execute_line(&mut db, "DATASET users COLUMNS (id: INT, name: STRING, age: INT, active: BOOL, score: FLOAT)", 1).unwrap();
                    db
                },
                |mut db| {
                    for i in 0..batch_size {
                        execute_line(
                            &mut db,
                            &format!("INSERT INTO users VALUES ({}, \"User{}\", {}, true, 0.{})", i, i, 20 + (i % 50), 50 + (i % 50)),
                            1
                        ).unwrap();
                    }
                },
                criterion::BatchSize::SmallInput
            );
        });
    }
    group.finish();
}

fn select_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("select");

    for row_count in [100, 1000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(row_count), row_count, |bench, &row_count| {
            let mut db = TensorDb::new();
            // Setup: create dataset with specified rows
            execute_line(&mut db, "DATASET users COLUMNS (id: INT, name: STRING, age: INT, active: BOOL, score: FLOAT)", 1).unwrap();
            for i in 0..row_count {
                execute_line(
                    &mut db,
                    &format!("INSERT INTO users VALUES ({}, \"User{}\", {}, true, 0.{})", i, i, 20 + (i % 50), 50 + (i % 50)),
                    1
                ).unwrap();
            }
            bench.iter(|| {
                execute_line(&mut db, black_box("SELECT * FROM users"), 1).unwrap()
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    dataset_creation_benchmark,
    insert_benchmark,
    batch_insert_benchmark,
    select_benchmark
);
criterion_main!(benches);
