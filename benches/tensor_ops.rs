// Phase 0 Baseline Benchmarks - Tensor Operations
// Using correct DSL syntax from examples
// Enhanced with incremental vector sizes for comprehensive baseline

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use linal::dsl::execute_line;
use linal::engine::db::TensorDb;

// Vector creation benchmarks at different sizes
fn vector_creation_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_creation");

    for size in [128, 512, 4096].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, &size| {
            bench.iter_batched(
                TensorDb::new,
                |mut db| {
                    let values: Vec<String> = (0..size).map(|i| format!("{}.0", i)).collect();
                    let vector_def = format!("VECTOR v1 = [{}]", values.join(", "));
                    execute_line(&mut db, black_box(&vector_def), 1).unwrap()
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

// Vector addition at different sizes
fn vector_addition_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_add");

    for size in [128, 512, 4096, 100_000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, &size| {
            let mut db = TensorDb::new();
            // Use 0.0 to avoid huge string alloc overhead in setup
            let values: Vec<String> = (0..size).map(|_| "1.0".to_string()).collect();
            let v1_def = format!("VECTOR v1 = [{}]", values.join(", "));
            let v2_def = format!("VECTOR v2 = [{}]", values.join(", "));
            execute_line(&mut db, &v1_def, 1).unwrap();
            execute_line(&mut db, &v2_def, 1).unwrap();

            bench.iter(|| execute_line(&mut db, black_box("LET v3 = ADD v1 v2"), 1).unwrap());
        });
    }
    group.finish();
}

// Vector multiplication at different sizes
fn vector_multiply_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_multiply");

    for size in [128, 512, 4096, 100_000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, &size| {
            let mut db = TensorDb::new();
            let values: Vec<String> = (0..size).map(|_| "1.0".to_string()).collect();
            let v1_def = format!("VECTOR v1 = [{}]", values.join(", "));
            let v2_def = format!("VECTOR v2 = [{}]", values.join(", "));
            execute_line(&mut db, &v1_def, 1).unwrap();
            execute_line(&mut db, &v2_def, 1).unwrap();

            bench.iter(|| execute_line(&mut db, black_box("LET v3 = v1 * v2"), 1).unwrap());
        });
    }
    group.finish();
}

// Similarity computation at different sizes
fn similarity_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("similarity");

    // Skip 100k for similarity for now as it's O(N) but might be slow to setup
    for size in [128, 512, 4096].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, &size| {
            let mut db = TensorDb::new();
            let values: Vec<String> = (0..size).map(|i| format!("{}.0", i % 100 + 1)).collect();
            let v1_def = format!("VECTOR v1 = [{}]", values.join(", "));
            let v2_def = format!("VECTOR v2 = [{}]", values.join(", "));
            execute_line(&mut db, &v1_def, 1).unwrap();
            execute_line(&mut db, &v2_def, 1).unwrap();

            bench.iter(|| {
                execute_line(&mut db, black_box("LET sim = SIMILARITY v1 WITH v2"), 1).unwrap()
            });
        });
    }
    group.finish();
}

// Matrix operations
fn matrix_operations_benchmark(c: &mut Criterion) {
    c.bench_function("matrix_creation", |bench| {
        bench.iter_batched(
            TensorDb::new,
            |mut db| {
                execute_line(
                    &mut db,
                    black_box("MATRIX m1 = [[1.0, 2.0], [3.0, 4.0]]"),
                    1,
                )
                .unwrap()
            },
            criterion::BatchSize::SmallInput,
        );
    });

    c.bench_function("matrix_multiply_small", |bench| {
        let mut db = TensorDb::new();
        execute_line(&mut db, "MATRIX m1 = [[1.0, 2.0], [3.0, 4.0]]", 1).unwrap();
        execute_line(&mut db, "MATRIX m2 = [[5.0, 6.0], [7.0, 8.0]]", 1).unwrap();

        bench.iter(|| execute_line(&mut db, black_box("LET m3 = MATMUL m1 m2"), 1).unwrap());
    });

    c.bench_function("matrix_multiply_100x100", |bench| {
        let mut db = TensorDb::new();
        // Construct 100x100 matrix string
        let size = 100;
        let mut rows = Vec::new();
        for _ in 0..size {
            let row: Vec<String> = (0..size).map(|_| "1.0".to_string()).collect();
            rows.push(format!("[{}]", row.join(",")));
        }
        let matrix_str = format!("[{}]", rows.join(","));

        execute_line(&mut db, &format!("MATRIX m1 = {}", matrix_str), 1).unwrap();
        execute_line(&mut db, &format!("MATRIX m2 = {}", matrix_str), 1).unwrap();

        bench.iter(|| execute_line(&mut db, black_box("LET m3 = MATMUL m1 m2"), 1).unwrap());
    });
}

criterion_group!(
    benches,
    vector_creation_benchmark,
    vector_addition_benchmark,
    vector_multiply_benchmark,
    similarity_benchmark,
    matrix_operations_benchmark
);
criterion_main!(benches);
