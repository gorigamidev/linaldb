# VectorDB Modular Architecture Restructuring Plan
## Overview
Restructure the codebase from a flat module structure to a hierarchical, modular architecture that follows OOP principles and Rust best practices.

## Goals:

Clear separation of concerns
Modular design for easy feature addition
Minimal coupling between modules
Clean import structure (avoid barrel exports)
Maintain all 78 passing tests

## Proposed Directory Structure
src/
├── main.rs                    # Entry point (if needed)
├── lib.rs                     # Root library file (minimal, just module declarations)
│
├── core/                      # Core data structures and storage
│   ├── mod.rs
│   ├── tensor.rs             # Tensor, TensorId, Shape
│   ├── value.rs              # Value, ValueType
│   ├── tuple.rs              # Tuple, Schema, Field
│   ├── dataset.rs            # Dataset, DatasetMetadata
│   └── store/
│       ├── mod.rs
│       ├── tensor_store.rs   # InMemoryTensorStore
│       └── dataset_store.rs  # DatasetStore
│
├── engine/                    # Execution engine
│   ├── mod.rs
│   ├── db.rs                 # TensorDb (main database)
│   ├── operations.rs         # BinaryOp, UnaryOp enums
│   ├── error.rs              # EngineError
│   └── executor/
│       ├── mod.rs
│       ├── tensor_ops.rs     # Tensor operation execution
│       ├── matrix_ops.rs     # Matrix operation execution
│       └── dataset_ops.rs    # Dataset operation execution
│
├── dsl/                       # DSL parsing and execution
│   ├── mod.rs
│   ├── error.rs              # DslError
│   ├── parser.rs             # Main parser logic
│   └── handlers/
│       ├── mod.rs
│       ├── tensor.rs         # DEFINE, VECTOR, MATRIX handlers
│       ├── operations.rs     # LET handler
│       ├── dataset.rs        # DATASET, INSERT INTO handlers
│       └── introspection.rs  # SHOW handlers
│
├── ops/                       # Low-level operations (keep as is)
│   └── mod.rs                # All tensor/matrix operations
│
└── lib/                       # Shared utilities
    ├── mod.rs
    └── parsing.rs            # Common parsing utilities