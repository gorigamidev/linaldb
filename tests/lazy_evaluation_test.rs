use linal::core::tensor::{Expression, Shape, Tensor, TensorId, TensorMetadata};
use linal::engine::kernels::evaluate_expression;

#[test]
fn test_lazy_add_multiply() {
    let id_a = TensorId::new();
    let id_b = TensorId::new();
    let shape = Shape::new(vec![2]);
    let a = Tensor::new(
        id_a,
        shape.clone(),
        vec![1.0, 2.0],
        TensorMetadata::new(id_a, None),
    )
    .unwrap();
    let b = Tensor::new(
        id_b,
        shape.clone(),
        vec![10.0, 20.0],
        TensorMetadata::new(id_b, None),
    )
    .unwrap();

    // Expression: (A + B) * 2.0
    let expr = Expression::ScalarMul(
        Box::new(Expression::Add(
            Box::new(Expression::Literal(a)),
            Box::new(Expression::Literal(b)),
        )),
        2.0,
    );

    let res = evaluate_expression(&expr, chrono::Utc::now()).unwrap();

    assert_eq!(res.data_ref(), &[22.0, 44.0]);
}

#[test]
fn test_lazy_matmul() {
    let id_a = TensorId::new();
    let id_b = TensorId::new();

    // 2x1 * 1x2 -> 2x2
    let a = Tensor::new(
        id_a,
        Shape::new(vec![2, 1]),
        vec![2.0, 3.0],
        TensorMetadata::new(id_a, None),
    )
    .unwrap();
    let b = Tensor::new(
        id_b,
        Shape::new(vec![1, 2]),
        vec![4.0, 5.0],
        TensorMetadata::new(id_b, None),
    )
    .unwrap();

    let expr = Expression::MatMul(
        Box::new(Expression::Literal(a)),
        Box::new(Expression::Literal(b)),
    );

    let res = evaluate_expression(&expr, chrono::Utc::now()).unwrap();

    assert_eq!(res.shape.dims, vec![2, 2]);
    // [2 * 4, 2 * 5]
    // [3 * 4, 3 * 5]
    assert_eq!(res.data_ref(), &[8.0, 10.0, 12.0, 15.0]);
}
