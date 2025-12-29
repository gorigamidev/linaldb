use criterion::{black_box, criterion_group, criterion_main, Criterion};
use linal::core::tensor::{Shape, Tensor, TensorId};
use linal::engine::context::ExecutionContext;
#[cfg(feature = "zero-copy")]
use std::sync::Arc;

// Benchmark allocation: ExecutionContext arena vs Standard Vec
fn allocation_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("allocation");

    // Benchmark standard Vec allocation
    group.bench_function("vec_alloc_1000", |b| {
        b.iter(|| {
            let vec: Vec<f32> = black_box(Vec::with_capacity(1000));
            black_box(vec)
        })
    });

    // Benchmark ExecutionContext arena allocation
    // Note: This benchmarks the *overhead* of tracking + allocation
    group.bench_function("arena_alloc_1000", |b| {
        let ctx = ExecutionContext::new();
        b.iter(|| {
            // In a real scenario we'd alloc slices, but here we test the overhead
            // of the tracking mechanism primarily
            let _val = ctx.alloc_slice(black_box(&[0.0; 1000]));
        })
    });

    group.finish();
}

// Benchmark zero-copy mechanics
fn zero_copy_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("zero_copy");

    let data_vec: Vec<f32> = (0..10_000).map(|x| x as f32).collect();
    #[cfg(feature = "zero-copy")]
    let data_arc = Arc::new(data_vec.clone());
    let shape = Shape::new(vec![10000]);

    // Baseline: Cloning a tensor (standard behavior)
    group.bench_function("tensor_clone_10k", |b| {
        // Setup a tensor
        let id = TensorId::new();
        let metadata = linal::core::tensor::TensorMetadata::new(id, None);
        let t = Tensor::new(id, shape.clone(), data_vec.clone(), metadata).unwrap();
        b.iter(|| {
            // Measure deep copy cost
            let _clone = black_box(t.clone());
        })
    });

    // New: Zero-copy sharing
    #[cfg(feature = "zero-copy")]
    group.bench_function("tensor_share_10k", |b| {
        // Setup a shared tensor
        let id = TensorId::new();
        let metadata = std::sync::Arc::new(linal::core::tensor::TensorMetadata::new(id, None));
        let t = Tensor::from_shared(id, shape.clone(), data_arc.clone(), metadata).unwrap();
        b.iter(|| {
            // Measure Arc cloning cost (should be near instant)
            let _shared = black_box(t.share());
        })
    });

    group.finish();
}

criterion_group!(benches, allocation_benchmark, zero_copy_benchmark);
criterion_main!(benches);
