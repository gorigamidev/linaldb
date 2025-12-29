use crate::core::store::{DatasetStoreError, StoreError};

#[derive(Debug)]
pub enum EngineError {
    Store(StoreError),
    NameNotFound(String),
    InvalidOp(String),
    DatasetError(DatasetStoreError),
    DatasetNotFound(String),
    ExecutionError(String),
    ConstraintViolation(String),
    ReferenceError(String),
}

impl From<StoreError> for EngineError {
    fn from(e: StoreError) -> Self {
        EngineError::Store(e)
    }
}

impl From<DatasetStoreError> for EngineError {
    fn from(e: DatasetStoreError) -> Self {
        EngineError::DatasetError(e)
    }
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EngineError::Store(e) => write!(f, "Store error: {}", e),
            EngineError::NameNotFound(name) => write!(f, "Tensor name not found: {}", name),
            EngineError::InvalidOp(msg) => write!(f, "Invalid operation: {}", msg),
            EngineError::DatasetError(e) => write!(f, "Dataset error: {}", e),
            EngineError::DatasetNotFound(name) => write!(f, "Dataset not found: {}", name),
            EngineError::ExecutionError(msg) => write!(f, "Execution error: {}", msg),
            EngineError::ConstraintViolation(msg) => write!(f, "Constraint violation: {}", msg),
            EngineError::ReferenceError(msg) => write!(f, "Reference error: {}", msg),
        }
    }
}

impl std::error::Error for EngineError {}
