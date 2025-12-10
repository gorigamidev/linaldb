// src/lib.rs

pub mod dataset;
pub mod dataset_store;
pub mod dsl;
pub mod engine;
pub mod ops;
pub mod store;
pub mod tensor;
pub mod tuple;
pub mod value;

// Re-exports para tener una API limpia desde fuera del crate
pub use dataset::{ColumnStats, Dataset, DatasetId, DatasetMetadata};
pub use dataset_store::{DatasetStore, DatasetStoreError};
pub use dsl::{execute_line, execute_script, DslError};
pub use engine::{BinaryOp, EngineError, TensorDb, TensorKind, UnaryOp};
pub use ops::{
    add,
    add_relaxed,
    cosine_similarity_1d,
    distance_1d,
    divide,
    divide_relaxed,
    dot_1d,
    flatten,
    index,
    l2_norm_1d,
    // Matrix operations
    matmul,
    multiply,
    multiply_relaxed,
    normalize_1d,
    reshape,
    scalar_mul,
    slice,
    sub,
    sub_relaxed,
    transpose,
};
pub use store::{InMemoryTensorStore, StoreError};
pub use tensor::{Shape, Tensor, TensorId};
pub use tuple::{Field, Schema, Tuple};
pub use value::{Value, ValueType};
