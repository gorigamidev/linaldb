# LINAL Architecture

This document provides a comprehensive overview of the LINAL engine architecture, its components, and design decisions.

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Core Components](#core-components)
4. [Execution Flow](#execution-flow)
5. [Storage Layer](#storage-layer)
6. [Query Processing](#query-processing)
7. [Type System](#type-system)
8. [Design Principles](#design-principles)

---

## Overview

LINAL is an in-memory analytical engine designed for linear algebra operations, structured data analysis, and machine learning workloads. It combines:

- **Tensor computation** (vectors, matrices, higher-dimensional tensors)
- **Structured datasets** (SQL-like tables with heterogeneous types)
- **Query optimization** (index-aware execution, predicate pushdown)
- **Persistence** (Parquet for datasets, JSON for tensors)

The engine is built in Rust with a modular architecture that separates concerns into distinct layers.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                      Application Layer                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │   CLI    │  │  Server  │  │   REPL   │  │  Scripts │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  │
└───────┼─────────────┼─────────────┼─────────────┼────────┘
        │             │             │             │
        └─────────────┴─────────────┴─────────────┘
                           │
        ┌──────────────────┴──────────────────┐
        │          DSL Layer (Parser)         │
        │  ┌──────────────────────────────┐   │
        │  │  Command Parsing & Routing   │   │
        │  │  Expression Evaluation       │   │
        │  │  Error Handling              │   │
        │  └──────────────────────────────┘   │
        └──────────────────┬──────────────────┘
                           │
        ┌──────────────────┴──────────────────┐
        │       Query Planning & Execution    │
        │  ┌──────────────┐  ┌──────────────┐ │
        │  │   Logical    │→ │   Physical   │ │
        │  │    Plan      │  │    Plan      │ │
        │  └──────────────┘  └──────────────┘ │
        │  ┌──────────────────────────────┐   │
        │  │      Query Optimizer         │   │
        │  │  - Index Selection           │   │
        │  │  - Predicate Pushdown        │   │
        │  └──────────────────────────────┘   │
        └──────────────────┬──────────────────┘
                           │
        ┌──────────────────┴──────────────────┐
        │         Engine Layer (TensorDb)     │
        │  ┌──────────────────────────────┐   │
        │  │   Database Instance Mgmt     │   │
        │  │   - Multi-database support   │   │
        │  │   - Context switching        │   │
        │  └──────────────────────────────┘   │
        └──────────────────┬──────────────────┘
                           │
        ┌──────────────────┴──────────────────┐
        │          Storage Layer              │
        │  ┌──────────────┐  ┌──────────────┐ │
        │  │   Tensor     │  │   Dataset    │ │
        │  │   Store      │  │   Store      │ │
        │  └──────────────┘  └──────────────┘ │
        │  ┌──────────────┐  ┌──────────────┐ │
        │  │   Hash       │  │   Vector     │ │
        │  │   Index      │  │   Index      │ │
        │  └──────────────┘  └──────────────┘ │
        └──────────────────┬──────────────────┘
                           │
        ┌──────────────────┴──────────────────┐
        │        Persistence Layer            │
        │  ┌──────────────┐  ┌──────────────┐ │
        │  │   Parquet    │  │     JSON     │ │
        │  │  (Datasets)  │  │   (Tensors)  │ │
        │  └──────────────┘  └──────────────┘ │
        └─────────────────────────────────────┘
```

---

## Core Components

### 1. Core Module (`src/core/`)

The core module contains fundamental data structures and abstractions:

#### `tensor.rs`

- **Tensor**: Multi-dimensional array with shape `[d1, d2, ..., dn]` and f32 data
- **TensorId**: Unique identifier for tensors
- **Shape**: Dimension specification supporting scalars, vectors, matrices, and higher-order tensors

#### `value.rs`

- **Value**: Enum representing all possible data types:
  - `Int`, `Float`, `String`, `Bool`
  - `Vector(usize)`, `Matrix(usize, usize)`, `Tensor(Shape)`
- **ValueType**: Type information for schema definitions

#### `tuple.rs`

- **Tuple**: Row representation with named fields
- **Schema**: Column definitions with types and constraints
- **Field**: Individual column specification

#### `dataset/` (Reference Graph)

- **Dataset**: A semantic view over existing tensors or other dataset columns. It does not own data directly but stores a map of `ResourceReference`s.
- **ResourceReference**: An enum representing a link to either a specific `TensorId` or a `(dataset, column)` pair in another dataset.
- **DatasetGraph**: A component responsible for resolving references. It supports **Transitive Resolution** (e.g., resolving a view of a view) and implements **Circular Dependency Detection** to prevent infinite resolution loops.
- **ColumnRole**: Metadata categorizing columns by their semantic purpose (e.g., `Feature`, `Target`, `Weight`, `Guid`).
- **Zero-Copy Guarantees**: Adding a column is an O(1) metadata operation. Underlying tensor data is shared via atomic reference counting (`Arc`), ensuring no data duplication.
- **Materialization**: While datasets are views in-memory, they can be materialized into physical rows and persisted via standard Parquet for portability.

#### `dataset_legacy.rs` (Row-Based)

- **Dataset**: Traditional row-oriented collection of `Tuple`s.
- **DatasetMetadata**: Versioning, timestamps, custom metadata.
- **ColumnStats**: Statistics for query optimization.

#### `store/`

- **InMemoryTensorStore**: In-memory storage for tensors.
- **DatasetStore**: In-memory storage for legacy datasets.

#### `index/`

- **HashIndex**: Exact match lookups (equality predicates)
- **VectorIndex**: Similarity search (cosine, Euclidean distance)

#### `storage.rs`

- **StorageEngine**: Trait for persistence abstraction
- **ParquetStorage**: Parquet-based dataset persistence
- **JsonStorage**: JSON-based tensor persistence

### 2. Engine Module (`src/engine/`)

The engine module orchestrates execution:

#### `db.rs`

- **TensorDb**: Main database engine managing multiple database instances
- **DatabaseInstance**: Isolated database with its own stores
- Features:
  - Multi-database support with context switching
  - Automatic recovery from disk on startup
  - Configuration via `linal.toml`

#### `operations.rs`

- **BinaryOp**: Binary operations (ADD, SUBTRACT, MULTIPLY, DIVIDE, etc.)
- **UnaryOp**: Unary operations (TRANSPOSE, FLATTEN, RESHAPE, etc.)
- **TensorKind**: Classification of tensor types

#### `kernels.rs`

- Low-level computational kernels:
  - Element-wise operations
  - Matrix multiplication (MATMUL)
  - Vector operations (dot product, cosine similarity, L2 distance)
  - Broadcasting and relaxed mode operations

#### `error.rs`

- **EngineError**: Unified error type for engine operations

### 3. DSL Module (`src/dsl/`)

The DSL module handles command parsing and execution:

#### `mod.rs`

- **execute_line()**: Execute a single DSL command
- **execute_script()**: Execute a script file
- **DslOutput**: Structured output format

#### `handlers/`

Command-specific handlers:

- **tensor.rs**: DEFINE, VECTOR, MATRIX, SHOW commands
- **dataset.rs**: DATASET, INSERT INTO, SELECT, FILTER, etc.
- **operations.rs**: LET, binary/unary operations
- **index.rs**: CREATE INDEX, CREATE VECTOR INDEX
- **search.rs**: SEARCH (vector similarity)
- **persistence.rs**: SAVE, LOAD, LIST commands
- **instance.rs**: CREATE DATABASE, USE, DROP DATABASE
- **metadata.rs**: SET DATASET METADATA
- **explain.rs**: EXPLAIN, EXPLAIN PLAN
- **introspection.rs**: SHOW commands

#### `error.rs`

- **DslError**: DSL-specific error types

### 4. Query Module (`src/query/`)

The query module implements query planning and optimization:

#### `logical.rs`

- **LogicalPlan**: High-level query representation
- Operations: Scan, Filter, Project, Aggregate, GroupBy, Limit

#### `physical.rs`

- **PhysicalPlan**: Executable query plan
- **Executor**: Executes physical plans with index-aware execution

#### `planner.rs`

- **QueryPlanner**: Converts logical plans to physical plans
- **Optimizer**: Applies optimizations:
  - Index selection
  - Predicate pushdown
  - Projection pruning

### 5. Server Module (`src/server/`)

HTTP server implementation:

- REST API endpoint (`POST /execute`)
- OpenAPI/Swagger documentation (`/swagger-ui`)
- Query timeout (30s)
- Request validation (size limits, non-empty checks)
- Support for TOON and JSON output formats

### 6. Utils Module (`src/utils/`)

Utility functions:

- **parsing.rs**: String parsing helpers

---

## Execution Flow

### 1. Command Parsing

```
DSL Command → Parser → Command AST → Route to Handler
```

Example: `SELECT * FROM users WHERE id > 10`

- Parser identifies `SELECT` command
- Routes to `dataset.rs` handler
- Parses query components (columns, table, filter, etc.)

### 2. Query Planning (for SELECT queries)

```
SELECT Query → Logical Plan → Physical Plan → Execution
```

1. **Logical Plan**: High-level representation

   ```
   Project(columns: [*])
     └─ Filter(predicate: id > 10)
         └─ Scan(table: users)
   ```

2. **Optimization**: Apply optimizations
   - Check for indexes on `id`
   - Push predicate to index scan if available

3. **Physical Plan**: Executable plan

   ```
   IndexScan(index: id_idx, predicate: > 10)
     └─ Project(columns: [*])
   ```

4. **Execution**: Execute physical plan
   - Use index for fast lookup
   - Apply projection
   - Return results

### 3. Expression Evaluation

Expressions are evaluated recursively:

- **Literals**: Direct value
- **Variables**: Lookup in tensor/dataset store
- **Binary Operations**: Evaluate operands, apply operation
- **Indexing**: `tensor[i, j]`, `tuple.field`, `dataset.column`

### 4. Aggregation

GROUP BY queries:

1. Group rows by grouping columns
2. Apply aggregation functions (SUM, AVG, COUNT, MIN, MAX)
3. Support element-wise aggregation for vectors/matrices
4. Apply HAVING clause filter

---

## Storage Layer

### In-Memory Storage

- **TensorStore**: HashMap-based storage keyed by TensorId
- **DatasetStore**: HashMap-based storage with name and ID indexes
- **Indices**: Maintained automatically on INSERT

### Persistence

#### Datasets (Parquet)

- Columnar format for efficient storage
- Metadata stored separately in JSON
- Schema preserved

#### Tensors (JSON)

- Full tensor serialization.
- Shape and data preserved.
- Suitable for weights and model parameters.

#### Tensor-First Datasets (In-Memory)

- **Zero-Copy Architecture**: Datasets reference tensors in the `TensorStore` by ID. Adding a column is an O(1) metadata operation.
- **Math Integration**: Columns are exposed as standard LINAL symbols via dot notation. `LET x = ds.vec * 2.0` resolves `ds.vec` to its underlying `TensorId` and executes normally.
- **Reverse Integration**: Results of any tensor operation can be added back to a dataset as a new column, maintaining the zero-copy chain.
- **Persistence**: While primarily in-memory views, they can be persisted to Parquet using the `SAVE DATASET` command, which triggers on-demand materialization.

#### Safety & Integrity

- **Row Count Validation**: The engine strictly enforces that all columns within a tensor-first dataset have a consistent "row count" (dimension 0 of the tensor). This prevents malformed data from entering analytical pipelines.
- **Dangling Reference Detection**: Since datasets reference tensors by `TensorId`, the engine performs on-demand audits. The `SHOW` command generates **Health Warnings** if a dataset column points to a tensor that has been deleted from the `TensorStore`.
- **Zero-Copy Guarantees**: Metadata-only operations ensure that datasets never duplicate underlying vector data, preserving memory and cache locality.

### Recovery

On engine startup:

1. Scan `data_dir` for database directories
2. Load dataset metadata from JSON
3. Load tensor metadata
4. Lazy-load actual data on first access

---

## Query Processing

### Index-Aware Execution

1. **Index Selection**: Planner checks for applicable indexes
   - HashIndex for equality predicates (`WHERE id = 5`)
   - VectorIndex for similarity search (`WHERE embedding ~= [...]`)

2. **Predicate Pushdown**: Filters applied as early as possible
   - Use index to filter before scanning full dataset

3. **Execution**: Physical plan uses index when available
   - IndexScan instead of full table scan
   - Significant performance improvement for filtered queries

### Aggregation Execution

1. **Grouping**: Hash-based grouping by grouping columns
2. **Aggregation**: Apply aggregation functions per group
   - Element-wise for vectors/matrices
   - Scalar for numeric types
3. **HAVING**: Filter groups after aggregation

---

## Type System

### Type Hierarchy

```
Value
├── Scalar
│   ├── Int
│   ├── Float
│   ├── String
│   └── Bool
└── Tensor
    ├── Vector(n)
    ├── Matrix(m, n)
    └── Tensor(Shape)
```

### Type Inference

- **Arithmetic**: Int + Float → Float
- **Broadcasting**: Scalar * Vector → Vector
- **Matrix Operations**: Matrix * Matrix → Matrix (if compatible)

### Type Safety

- Compile-time type checking in expressions
- Runtime validation for dataset operations
- Clear error messages for type mismatches

---

## Design Principles

### 1. Modularity

- Clear separation of concerns
- Minimal coupling between modules
- Easy to extend and test

### 2. Performance

- In-memory first design
- Index-aware query execution
- Efficient tensor operations

### 3. Expressiveness

- Rich type system
- SQL-inspired querying
- Linear algebra operations

### 4. Usability

- Human-friendly DSL
- Multiple access methods (CLI, REPL, Server)
- Comprehensive error messages

### 5. Extensibility

- Trait-based abstractions (StorageEngine, Index)
- Plugin-friendly architecture
- Easy to add new operations

---

## Configuration

Engine configuration via `linal.toml`:

```toml
[storage]
data_dir = "./data"
default_db = "default"
```

- **data_dir**: Root directory for persistence
- **default_db**: Default database name

---

## Error Handling

Unified error types:

- **EngineError**: Engine-level errors
- **DslError**: DSL parsing/execution errors
- **DatasetStoreError**: Dataset operation errors

All errors propagate with context for debugging.

---

## Future Enhancements

- GPU-backed tensor execution
- Distributed execution
- Columnar execution engine
- Python/WASM integration
- Native ML operators (KNN, clustering, PCA)

---

For more details, see:

- [DSL Reference](DSL_REFERENCE.md)
- [Tasks & Implementation](Tasks_implementations.md)
- [Changelog](../CHANGELOG.md)
