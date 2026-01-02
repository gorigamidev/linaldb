use linal::core::backend::ComputeBackend;
use linal::core::backend::CpuBackend;
use linal::core::tensor::{Shape, Tensor, TensorId, TensorMetadata};
use linal::engine::context::ExecutionContext;

#[test]
fn test_timestamp_reuse_in_context() {
    let mut ctx = ExecutionContext::new();
    let initial_time = ctx.created_at;

    // Simulate some work taking time
    std::thread::sleep(std::time::Duration::from_millis(10));

    // Create tensors
    let backend = CpuBackend::new();
    let shape = Shape::new(vec![4]);
    let data = vec![1.0, 2.0, 3.0, 4.0];

    // Initial tensors can have their own metadata/timestamps, that's fine
    let meta_a = TensorMetadata::new(TensorId::new(), None);
    let a = Tensor::new(TensorId::new(), shape.clone(), data.clone(), meta_a).unwrap();
    let b = Tensor::new(
        TensorId::new(),
        shape.clone(),
        data.clone(),
        TensorMetadata::new(TensorId::new(), None),
    )
    .unwrap();

    // Perform op
    let result = backend.add(&mut ctx, &a, &b, TensorId::new()).unwrap();

    // Verify result metadata has EXACTLY the same timestamp as context
    // This confirms we bypassed Utc::now() syscalls inside the kernel
    assert_eq!(
        result.metadata.created_at, initial_time,
        "Tensor metadata timestamp should match context creation time"
    );
}

#[test]
fn test_simd_allocation_safety() {
    // Large enough to verify the functionality of Vec::with_capacity + set_len
    // We want to ensure that using uninitialized memory doesn't lead to issues
    // and that we are correctly overwriting it.
    let len = 100_000;
    let mut ctx = ExecutionContext::new();
    let backend = CpuBackend::new();

    let data_a: Vec<f32> = (0..len).map(|i| i as f32).collect();
    let data_b: Vec<f32> = (0..len).map(|i| (i as f32) * 2.0).collect();

    let shape = Shape::new(vec![len]);
    let a = Tensor::new(
        TensorId::new(),
        shape.clone(),
        data_a.clone(),
        TensorMetadata::new(TensorId::new(), None),
    )
    .unwrap();
    let b = Tensor::new(
        TensorId::new(),
        shape.clone(),
        data_b.clone(),
        TensorMetadata::new(TensorId::new(), None),
    )
    .unwrap();

    let result = backend.add(&mut ctx, &a, &b, TensorId::new()).unwrap();

    // Verify data correctness
    let res_data = result.data_ref();
    for i in 0..len {
        let expected = data_a[i] + data_b[i];
        if (res_data[i] - expected).abs() > f32::EPSILON {
            panic!(
                "Mismatch at index {}: expected {}, got {}",
                i, expected, res_data[i]
            );
        }
    }
}

#[test]
fn test_zero_copy_reshape() {
    let mut ctx = ExecutionContext::new();
    let backend = CpuBackend::new();

    // Create initial tensor 2x2
    let shape = Shape::new(vec![2, 2]);
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let meta_a = TensorMetadata::new(TensorId::new(), None);
    let a = Tensor::new(TensorId::new(), shape, data, meta_a).unwrap();

    // Reshape to 4
    let new_shape = Shape::new(vec![4]);
    let b = backend
        .reshape(&mut ctx, &a, new_shape, TensorId::new())
        .unwrap();

    // Verify data equality
    assert_eq!(a.data_ref(), b.data_ref());

    // Verify ZERO-COPY: pointers must be identical
    let ptr_a = a.data.as_ptr();
    let ptr_b = b.data.as_ptr();
    assert_eq!(
        ptr_a, ptr_b,
        "Reshaped tensor should share the same memory location"
    );
}

#[test]
fn test_zero_copy_transpose() {
    let mut ctx = ExecutionContext::new();
    let backend = CpuBackend::new();

    // Create matrix 2x3
    // [[1, 2, 3],
    //  [4, 5, 6]]
    let shape = Shape::new(vec![2, 3]);
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let meta_a = TensorMetadata::new(TensorId::new(), None);
    let a = Tensor::new(TensorId::new(), shape, data, meta_a).unwrap();

    // Transpose to 3x2
    // [[1, 4],
    //  [2, 5],
    //  [3, 6]]
    let b = backend.transpose(&mut ctx, &a, TensorId::new()).unwrap();

    // Verify shape
    assert_eq!(b.shape.dims, vec![3, 2]);

    // Verify ZERO-COPY: pointers must be identical
    let ptr_a = a.data.as_ptr();
    let ptr_b = b.data.as_ptr();
    assert_eq!(
        ptr_a, ptr_b,
        "Transposed tensor should share the same memory location"
    );

    // Verify strides (swapped)
    // A strides: [3, 1]
    // B strides: [1, 3]
    assert_eq!(a.strides, vec![3, 1]);
    assert_eq!(b.strides, vec![1, 3]);

    // Verify content access (logical)
    // b[0, 1] should be 4.0
    // b[1, 0] should be 2.0
    use linal::engine::kernels::index;
    assert_eq!(index(&b, &[0, 1]).unwrap(), 4.0);
    assert_eq!(index(&b, &[1, 0]).unwrap(), 2.0);
}

#[test]
fn test_zero_copy_slice() {
    // Create matrix 3x4
    // 0  1  2  3
    // 4  5  6  7
    // 8  9 10 11
    let shape = Shape::new(vec![3, 4]);
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let meta_a = TensorMetadata::new(TensorId::new(), None);
    let a = Tensor::new(TensorId::new(), shape, data, meta_a).unwrap();

    // Slice rows 1..3 (rows 1 and 2)
    // 4  5  6  7
    // 8  9 10 11
    use linal::engine::kernels;
    let b = kernels::slice(&a, 0, 1, 3, TensorId::new()).unwrap();

    // Verify shape
    assert_eq!(b.shape.dims, vec![2, 4]);

    // Verify ZERO-COPY
    assert_eq!(a.data.as_ptr(), b.data.as_ptr());
    assert_eq!(b.offset, a.offset + 1 * 4); // offset by 1 row (4 elements)

    use linal::engine::kernels::index;
    assert_eq!(index(&b, &[0, 0]).unwrap(), 4.0);
    assert_eq!(index(&b, &[1, 3]).unwrap(), 11.0);

    // Slice columns of B: cols 1..3
    // 5  6
    // 9 10
    let c = kernels::slice(&b, 1, 1, 3, TensorId::new()).unwrap();

    assert_eq!(c.shape.dims, vec![2, 2]);
    assert_eq!(c.strides, vec![4, 1]); // Strides preserved from original 3x4

    // Offset relative to B: + 1 column (1 element)
    // Total offset = 4 (row 1) + 1 (col 1) = 5
    assert_eq!(c.offset, b.offset + 1);
    assert_eq!(c.offset, 5);

    assert_eq!(index(&c, &[0, 0]).unwrap(), 5.0);
    assert_eq!(index(&c, &[1, 1]).unwrap(), 10.0);
}
