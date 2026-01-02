use linal::core::tensor::{Shape, Tensor, TensorId, TensorMetadata};
use linal::engine::kernels;

fn create_large_tensor(size: usize, val: f32) -> Tensor {
    let id = TensorId::new();
    let shape = Shape::new(vec![size]);
    let data = vec![val; size];
    let metadata = TensorMetadata::new(id, None);
    Tensor::new(id, shape, data, metadata).expect("Failed to create tensor")
}

fn create_large_matrix(rows: usize, cols: usize, val: f32) -> Tensor {
    let id = TensorId::new();
    let shape = Shape::new(vec![rows, cols]);
    let data = vec![val; rows * cols];
    let metadata = TensorMetadata::new(id, None);
    Tensor::new(id, shape, data, metadata).expect("Failed to create tensor")
}

#[test]
fn test_parallel_add_large() {
    let size = 100_000; // Above PARALLEL_THRESHOLD (50k)
    let t1 = create_large_tensor(size, 1.0);
    let t2 = create_large_tensor(size, 2.0);

    let res = kernels::add(&t1, &t2, TensorId::new()).unwrap();

    assert_eq!(res.shape.num_elements(), size);
    assert_eq!(res.data_ref()[0], 3.0);
    assert_eq!(res.data_ref()[size - 1], 3.0);
}

#[test]
fn test_parallel_scalar_mul_large() {
    let size = 100_000;
    let t1 = create_large_tensor(size, 2.0);

    let res = kernels::scalar_mul(&t1, 3.0, TensorId::new()).unwrap();

    assert_eq!(res.data_ref()[0], 6.0);
    assert_eq!(res.data_ref()[size - 1], 6.0);
}

#[test]
fn test_parallel_matmul_large() {
    // 500x500 = 250,000 elements > 50,000 threshold
    let rows = 500;
    let cols = 500;

    let a = create_large_matrix(rows, cols, 1.0);
    let b = create_matrix_identity(rows);

    let res = kernels::matmul(&a, &b, TensorId::new()).unwrap();

    // A * I = A
    assert_eq!(res.shape.dims, vec![rows, cols]);
    assert_eq!(res.data_ref()[0], 1.0);
    assert_eq!(res.data_ref()[rows * cols - 1], 1.0);
}

fn create_matrix_identity(size: usize) -> Tensor {
    let id = TensorId::new();
    let shape = Shape::new(vec![size, size]);
    let mut data = vec![0.0; size * size];
    for i in 0..size {
        data[i * size + i] = 1.0;
    }
    let metadata = TensorMetadata::new(id, None);
    Tensor::new(id, shape, data, metadata).expect("Failed to create identity matrix")
}
