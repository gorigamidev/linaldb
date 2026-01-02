// tests/matrix_operations.rs

use linal::{flatten, index, matmul, reshape, slice, transpose, Shape, Tensor, TensorId};

fn create_matrix(_id: u64, rows: usize, cols: usize, data: Vec<f32>) -> Tensor {
    let id = TensorId::new();
    let shape = Shape::new(vec![rows, cols]);
    let metadata = linal::core::tensor::TensorMetadata::new(id, None);
    Tensor::new(id, shape, data, metadata).unwrap()
}

fn create_vector(_id: u64, data: Vec<f32>) -> Tensor {
    let id = TensorId::new();
    let shape = Shape::new(vec![data.len()]);
    let metadata = linal::core::tensor::TensorMetadata::new(id, None);
    Tensor::new(id, shape, data, metadata).unwrap()
}

#[test]
fn test_matmul_basic() {
    // A: 2x3 matrix
    let a = create_matrix(1, 2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    // B: 3x2 matrix
    let b = create_matrix(2, 3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);

    // C = A * B should be 2x2
    let c = matmul(&a, &b, TensorId::new()).unwrap();

    assert_eq!(c.shape.dims, vec![2, 2]);

    // Expected result:
    // C[0,0] = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
    // C[0,1] = 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
    // C[1,0] = 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
    // C[1,1] = 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154
    assert_eq!(c.data_ref(), vec![58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn test_matmul_identity() {
    // A: 2x2 matrix
    let a = create_matrix(1, 2, 2, vec![1.0, 2.0, 3.0, 4.0]);

    // I: 2x2 identity matrix
    let identity = create_matrix(2, 2, 2, vec![1.0, 0.0, 0.0, 1.0]);

    // A * I should equal A
    let result = matmul(&a, &identity, TensorId::new()).unwrap();
    assert_eq!(result.data, a.data);
}

#[test]
fn test_matmul_dimension_mismatch() {
    let a = create_matrix(1, 2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = create_matrix(2, 2, 2, vec![1.0, 2.0, 3.0, 4.0]); // Wrong dimensions

    let result = matmul(&a, &b, TensorId::new());
    assert!(result.is_err());
}

#[test]
fn test_transpose() {
    // A: 2x3 matrix
    let a = create_matrix(1, 2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    // A^T should be 3x2
    let a_t = transpose(&a, TensorId::new()).unwrap();

    assert_eq!(a_t.shape.dims, vec![3, 2]);
    // Logical verification for zero-copy transpose
    assert_eq!(index(&a_t, &[0, 0]).unwrap(), 1.0);
    assert_eq!(index(&a_t, &[0, 1]).unwrap(), 4.0);
    assert_eq!(index(&a_t, &[1, 0]).unwrap(), 2.0);
    assert_eq!(index(&a_t, &[1, 1]).unwrap(), 5.0);
    assert_eq!(index(&a_t, &[2, 0]).unwrap(), 3.0);
    assert_eq!(index(&a_t, &[2, 1]).unwrap(), 6.0);
}

#[test]
fn test_transpose_square() {
    // Square matrix
    let a = create_matrix(1, 3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);

    let a_t = transpose(&a, TensorId::new()).unwrap();

    assert_eq!(a_t.shape.dims, vec![3, 3]);
    // Logical verification
    assert_eq!(index(&a_t, &[0, 1]).unwrap(), 4.0); // was [1, 0]
    assert_eq!(index(&a_t, &[1, 0]).unwrap(), 2.0); // was [0, 1]
    assert_eq!(index(&a_t, &[0, 2]).unwrap(), 7.0); // was [2, 0]
}

#[test]
fn test_reshape() {
    // 2x3 matrix
    let a = create_matrix(1, 2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    // Reshape to 3x2
    let reshaped = reshape(&a, Shape::new(vec![3, 2]), TensorId::new()).unwrap();
    assert_eq!(reshaped.shape.dims, vec![3, 2]);
    // Reshape preserves data order, so data_ref check is valid if logic is correct?
    // Actually, reshape is zero-copy too, but it doesn't change order, just dims.
    // If original was contiguous, reshaped might see same data.
    // However, [2,3] row-major -> [1,2,3,4,5,6]
    // [3,2] row-major -> [1,2,3,4,5,6]
    // Logical values change:
    // Old: (0,0)=1, (0,1)=2, (0,2)=3, (1,0)=4, (1,1)=5, (1,2)=6
    // New: (0,0)=1, (0,1)=2, (1,0)=3, (1,1)=4, (2,0)=5, (2,1)=6
    // data_ref check is valid here because reshape strictly preserves physical order.
    assert_eq!(reshaped.data_ref(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    // Reshape to 1x6
    let reshaped2 = reshape(&a, Shape::new(vec![1, 6]), TensorId::new()).unwrap();
    assert_eq!(reshaped2.shape.dims, vec![1, 6]);

    // Reshape to 6x1
    let reshaped3 = reshape(&a, Shape::new(vec![6, 1]), TensorId::new()).unwrap();
    assert_eq!(reshaped3.shape.dims, vec![6, 1]);
}

#[test]
fn test_reshape_invalid() {
    let a = create_matrix(1, 2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    // Try to reshape to incompatible size
    let result = reshape(&a, Shape::new(vec![2, 2]), TensorId::new());
    assert!(result.is_err());
}

#[test]
fn test_flatten() {
    // 2x3 matrix
    let a = create_matrix(1, 2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let flat = flatten(&a, TensorId::new()).unwrap();

    assert_eq!(flat.shape.dims, vec![6]);
    assert_eq!(flat.shape.rank(), 1);
    assert_eq!(flat.data_ref(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_slice_vector() {
    let v = create_vector(1, vec![1.0, 2.0, 3.0, 4.0, 5.0]);

    // Slice [1:4]
    let sliced = slice(&v, 0, 1, 4, TensorId::new()).unwrap();

    assert_eq!(sliced.shape.dims, vec![3]);
    // Logical verification
    assert_eq!(index(&sliced, &[0]).unwrap(), 2.0);
    assert_eq!(index(&sliced, &[1]).unwrap(), 3.0);
    assert_eq!(index(&sliced, &[2]).unwrap(), 4.0);
}

#[test]
fn test_slice_matrix_rows() {
    let m = create_matrix(1, 3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    // Slice rows [1:3]
    let sliced = slice(&m, 0, 1, 3, TensorId::new()).unwrap();

    assert_eq!(sliced.shape.dims, vec![2, 2]);
    // Logical verification
    // Original rows: [1,2], [3,4], [5,6]
    // Sliced rows 1,2: [3,4], [5,6]
    assert_eq!(index(&sliced, &[0, 0]).unwrap(), 3.0);
    assert_eq!(index(&sliced, &[0, 1]).unwrap(), 4.0);
    assert_eq!(index(&sliced, &[1, 0]).unwrap(), 5.0);
    assert_eq!(index(&sliced, &[1, 1]).unwrap(), 6.0);
}

#[test]
fn test_slice_matrix_cols() {
    let m = create_matrix(1, 2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    // Slice columns [1:3]
    let sliced = slice(&m, 1, 1, 3, TensorId::new()).unwrap();

    assert_eq!(sliced.shape.dims, vec![2, 2]);
    // Logical verification
    // Original: [[1, 2, 3], [4, 5, 6]]
    // Sliced cols 1,2: [[2, 3], [5, 6]]
    assert_eq!(index(&sliced, &[0, 0]).unwrap(), 2.0);
    assert_eq!(index(&sliced, &[0, 1]).unwrap(), 3.0);
    assert_eq!(index(&sliced, &[1, 0]).unwrap(), 5.0);
    assert_eq!(index(&sliced, &[1, 1]).unwrap(), 6.0);
}

#[test]
fn test_index_vector() {
    let v = create_vector(1, vec![10.0, 20.0, 30.0, 40.0]);

    assert_eq!(index(&v, &[0]).unwrap(), 10.0);
    assert_eq!(index(&v, &[2]).unwrap(), 30.0);
    assert_eq!(index(&v, &[3]).unwrap(), 40.0);
}

#[test]
fn test_index_matrix() {
    let m = create_matrix(1, 2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    assert_eq!(index(&m, &[0, 0]).unwrap(), 1.0);
    assert_eq!(index(&m, &[0, 2]).unwrap(), 3.0);
    assert_eq!(index(&m, &[1, 1]).unwrap(), 5.0);
    assert_eq!(index(&m, &[1, 2]).unwrap(), 6.0);
}

#[test]
fn test_index_out_of_bounds() {
    let v = create_vector(1, vec![1.0, 2.0, 3.0]);

    let result = index(&v, &[5]);
    assert!(result.is_err());
}

#[test]
fn test_matrix_chain_operations() {
    // Create a 2x3 matrix
    let a = create_matrix(1, 2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    // Transpose to 3x2
    let a_t = transpose(&a, TensorId::new()).unwrap();
    assert_eq!(a_t.shape.dims, vec![3, 2]);

    // Multiply A * A^T (2x3 * 3x2 = 2x2)
    let result = matmul(&a, &a_t, TensorId::new()).unwrap();
    assert_eq!(result.shape.dims, vec![2, 2]);

    // Expected:
    // [1,2,3] * [1,4]   = 1+4+9 = 14,  1+8+18 = 27
    // [4,5,6]   [2,5]   = 4+10+18 = 32, 16+25+36 = 77
    //           [3,6]
    assert_eq!(result.data_ref(), vec![14.0, 32.0, 32.0, 77.0]);
}

#[test]
fn test_flatten_then_reshape() {
    let m = create_matrix(1, 2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    // Flatten
    let flat = flatten(&m, TensorId::new()).unwrap();
    assert_eq!(flat.shape.dims, vec![6]);

    // Reshape back to 3x2
    let reshaped = reshape(&flat, Shape::new(vec![3, 2]), TensorId::new()).unwrap();
    assert_eq!(reshaped.shape.dims, vec![3, 2]);
    assert_eq!(reshaped.data_ref(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}
