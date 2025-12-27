use linal::core::tensor::{Shape, Tensor, TensorId};
use std::sync::Arc;

#[test]
fn test_from_shared() {
    let data = Arc::new(vec![1.0, 2.0, 3.0, 4.0]);
    let shape = Shape::new(vec![2, 2]);
    let id = TensorId(1);

    let tensor =
        Tensor::from_shared(id, shape.clone(), data.clone()).expect("Should create tensor");

    assert_eq!(tensor.len(), 4);
    assert_eq!(tensor.rank(), 2);
    assert_eq!(tensor.data_ref(), &[1.0, 2.0, 3.0, 4.0]);

    // Verify it is actually using the shared memory
    assert!(Arc::ptr_eq(&tensor.data, &data));
}

#[test]
fn test_share_from_owned() {
    let data = vec![1.0, 2.0, 3.0];
    let shape = Shape::new(vec![3]);
    let id = TensorId(2);
    let tensor = Tensor::new(id, shape, data).expect("Should create tensor");

    // calling share on tensor should return reference to same Arc
    let shared = tensor.share();
    assert_eq!(*shared, vec![1.0, 2.0, 3.0]);

    // In new implementation, tensor.share() just clones the Arc
    assert!(Arc::ptr_eq(&tensor.data, &shared));
}

#[test]
fn test_share_from_shared() {
    let data = Arc::new(vec![10.0, 20.0]);
    let shape = Shape::new(vec![2]);
    let id = TensorId(3);
    let tensor = Tensor::from_shared(id, shape, data.clone()).expect("Should create tensor");

    // calling share on already shared tensor should return clone of Arc (cheap)
    let shared_again = tensor.share();
    assert!(Arc::ptr_eq(&data, &shared_again));
}

#[test]
fn test_copy_on_write() {
    let data = Arc::new(vec![1.0, 2.0, 3.0]);
    let shape = Shape::new(vec![3]);
    let id = TensorId(4);
    let mut tensor = Tensor::from_shared(id, shape, data.clone()).expect("Should create tensor");

    // Initial state
    assert!(Arc::ptr_eq(&tensor.data, &data));

    // Mutate data - should trigger COW because we hold 'data' Arc
    let mut_data = tensor.data_mut();
    mut_data[0] = 100.0;

    // Verify data changed
    assert_eq!(tensor.data_ref()[0], 100.0);

    // Original Arc data should remain unchanged
    assert_eq!(*data, vec![1.0, 2.0, 3.0]);

    // Tensor's data Arc should now be different from original
    assert!(!Arc::ptr_eq(&tensor.data, &data));
}
