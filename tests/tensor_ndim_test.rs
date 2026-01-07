use linal::core::tensor::{Shape, Tensor, TensorId, TensorMetadata};
use linal::engine::kernels::{add, flatten, multiply, scalar_mul};

#[test]
fn test_rank_3_add() {
    let id_a = TensorId::new();
    let id_b = TensorId::new();
    let shape = Shape::new(vec![2, 2, 2]); // total 8
    let data_a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let data_b = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];

    let a = Tensor::new(id_a, shape.clone(), data_a, TensorMetadata::new(id_a, None)).unwrap();
    let b = Tensor::new(id_b, shape.clone(), data_b, TensorMetadata::new(id_b, None)).unwrap();

    let res = add(&a, &b, TensorId::new()).unwrap();
    assert_eq!(res.shape.dims, vec![2, 2, 2]);
    assert_eq!(
        res.data_ref(),
        &[11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 77.0, 88.0]
    );
}

#[test]
fn test_rank_3_strided_multiply() {
    let id = TensorId::new();
    let shape = Shape::new(vec![2, 2, 2]);
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let a = Tensor::new(id, shape.clone(), data, TensorMetadata::new(id, None)).unwrap();

    // Create a 1.0 tensor of same shape
    let b = Tensor::new(
        TensorId::new(),
        shape.clone(),
        vec![1.0; 8],
        TensorMetadata::new(TensorId::new(), None),
    )
    .unwrap();

    // Simple verification
    let res = multiply(&a, &b, TensorId::new()).unwrap();
    assert_eq!(res.data_ref(), a.data_ref());
}

#[test]
fn test_rank_4_scalar_mul() {
    let id = TensorId::new();
    let shape = Shape::new(vec![2, 1, 2, 1]); // total 4
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let a = Tensor::new(id, shape.clone(), data, TensorMetadata::new(id, None)).unwrap();

    let res = scalar_mul(&a, 2.0, TensorId::new()).unwrap();
    assert_eq!(res.data_ref(), &[2.0, 4.0, 6.0, 8.0]);
}

#[test]
fn test_rank_3_flatten() {
    let id = TensorId::new();
    let shape = Shape::new(vec![2, 2, 2]);
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let a = Tensor::new(id, shape.clone(), data, TensorMetadata::new(id, None)).unwrap();

    let res = flatten(&a, TensorId::new()).unwrap();
    assert_eq!(res.shape.dims, vec![8]);
    assert_eq!(res.data_ref(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
}
