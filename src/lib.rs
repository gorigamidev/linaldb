// src/lib.rs

pub mod tensor;
pub mod ops;
pub mod store;
pub mod engine;
pub mod dsl;

// Re-exports para tener una API limpia desde fuera del crate
pub use tensor::{Tensor, TensorId, Shape};
pub use store::{InMemoryTensorStore, StoreError};
pub use ops::{
    add,
    sub,
    multiply,
    divide,
    scalar_mul,
    dot_1d,
    l2_norm_1d,
    distance_1d,
    cosine_similarity_1d,
    normalize_1d,
    add_relaxed,
    sub_relaxed,
    multiply_relaxed,
    divide_relaxed,
};
pub use engine::{TensorDb, BinaryOp, UnaryOp, EngineError, TensorKind};
pub use dsl::{DslError, execute_script, execute_line};