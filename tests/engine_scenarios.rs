use linal::engine::context::ExecutionContext;
use linal::{BinaryOp, Shape, TensorDb, UnaryOp};

#[test]
fn basic_scenario() {
    let mut db = TensorDb::new();

    let shape_vec3 = Shape::new(vec![3]);

    db.insert_named("a", shape_vec3.clone(), vec![1.0, 0.0, 0.0])
        .unwrap();
    db.insert_named("b", shape_vec3.clone(), vec![0.0, 1.0, 0.0])
        .unwrap();
    db.insert_named("c", shape_vec3.clone(), vec![1.0, 1.0, 0.0])
        .unwrap();

    let mut ctx = ExecutionContext::new();

    db.eval_binary(&mut ctx, "c1", "a", "b", BinaryOp::Add)
        .unwrap();
    db.eval_binary(&mut ctx, "corr_ac", "a", "c", BinaryOp::Correlate)
        .unwrap();
    db.eval_binary(&mut ctx, "sim_ac", "a", "c", BinaryOp::Similarity)
        .unwrap();
    db.eval_unary(&mut ctx, "a_half", "a", UnaryOp::Scale(0.5))
        .unwrap();

    let c1 = db.get("c1").unwrap();
    let corr_ac = db.get("corr_ac").unwrap();
    let sim_ac = db.get("sim_ac").unwrap();
    let a_half = db.get("a_half").unwrap();

    assert_eq!(c1.shape.dims, vec![3]);
    assert_eq!(c1.data_ref(), vec![1.0, 1.0, 0.0]);

    assert_eq!(corr_ac.shape.rank(), 0);
    assert!((corr_ac.data[0] - 1.0).abs() < 1e-6);

    let expected = 1.0 / 2f32.sqrt();
    assert!((sim_ac.data[0] - expected).abs() < 1e-6);

    assert_eq!(a_half.data_ref(), vec![0.5, 0.0, 0.0]);
}
