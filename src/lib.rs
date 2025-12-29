// src/lib.rs

pub mod core;

pub mod dsl;
pub mod engine;
pub mod query;
pub mod server;
pub mod utils;

// Re-exports replacing legacy modules with core aliases
pub use core::dataset;
pub use core::dataset_legacy;
pub use core::store as dataset_store;
pub use core::store;
pub use core::tensor;
pub use core::tuple;
pub use core::value;

// Re-exports para tener una API limpia desde fuera del crate
pub use dataset::{Dataset as TensorDataset, DatasetRegistry, DatasetSchema};
pub use dataset_legacy::{ColumnStats, Dataset, DatasetId, DatasetMetadata};
pub use dataset_store::{DatasetStore, DatasetStoreError};
pub use dsl::{execute_line, execute_script, DslError};
pub use engine::{BinaryOp, EngineError, TensorDb, TensorKind, UnaryOp};
// Re-export kernels as ops for compatibility
pub use engine::kernels as ops;
pub use engine::kernels::{
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
