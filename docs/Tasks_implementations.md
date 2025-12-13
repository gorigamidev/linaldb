# VectorDB Evolution: Dataset & Tuple Support

## Phase 0: Preparation

- [x] Fix Cargo.toml edition (2024 → 2021)
- [ ] Add ADD COLUMN for computed fields
- [ ] Add GROUP BY with COMPUTE aggregations
- [x] Add matrix operations (MATMUL, TRANSPOSE, RESHAPE, FLATTEN)
- [ ] Add indexing syntax (m[0, *], tuple.field)
- [ ] Update SHOW command for all types
- [x] Add SHOW SHAPE introspection
- [ ] Add SHOW SCHEMA introspection

## Phase 1: Dataset Store ✅

- [x] Implement DatasetStore.rs for dataset management
- [x] Add name-based and ID-based access
- [x] Add insert, get, remove operations
- [x] Add duplicate name validation
- [x] Add comprehensive unit tests (4 tests passing)

## Phase 2: Engine Integration ✅

- [x] Integrate DatasetStore into TensorDb
- [x] Add create_dataset and insert_row methods to TensorDb
- [x] Map EngineError to DatasetStoreError

## Phase 3: DSL Dataset Operations ✅

- [x] DATASET command (Dataset creation)
- [x] INSERT INTO command (Row insertion)
- [x] SELECT / FILTER / ORDER BY / LIMIT commands (Querying)

## Restructuring (Architectural Overhaul) ✅

- [x] Restructure src/ into modular components (core, engine, dsl, utils)
- [x] Create core module (tensor, value, tuple, dataset, store)
- [x] Create engine module (db, operations, error)
- [x] Create dsl module (parser, error, handlers) 
- [x] Create utils module (parsing) 
- [x] Clean up lib.rs exports (unified API)
- [x] Ensure all tests pass (Unit + Integration)

## Cleanup & Finalization ✅

- [x] Delete legacy files (src/tensor.rs, src/dataset.rs, etc.)
- [x] Remove empty directories
- [x] Fix regressions in DSL Handlers (Restore VECTOR/MATRIX syntax, TRANSPOSE/FLATTEN ops)
- [x] Verify all test suites (dsl_matrix_ops, dsl_show_commands, dsl_dataset_complete)

## Phase 4: Server & CLI (Current Focus) ✅

- [x] Add dependencies (clap, tokio, axum, serde)
- [x] Refactor DSL for structured output (DslOutput)
- [x] Implement CLI with subcommands (repl, run, server)
- [x] Implement REST API (POST /execute)
- [x] Verify Server and CLI

## Phase 5: TOON Integration & Server Refactor

- [x] Add toon-format dependency
- [x] Implement Serialize for core types (Tensor, Dataset, DslOutput)
- [x] Update server to return TOON format
- [x] Verify TOON output
- [x] Create automated tests for TOON header and body
- [x] Clean up project root (move docs, delete temp files)
- [x] Update README.md with TOON and Server details

---------------
# VectorDB / Tensor Engine – Extended Roadmap (Post-Phase 5.5)

This section refines and extends the roadmap starting from Phase 5.5, with the explicit goal of reaching a **public, usable engine** that can operate as:

- An in-memory analytical engine
- A CLI-first scientific database
- An HTTP-accessible compute service
- A foundation for ML / AI / statistical workloads

---

## Phase 5.5: Feature Catch-up (DSL & Core Completeness)

> Goal: Close all DSL and data-model gaps required for serious analytical usage.

- [x] Implement STACK operation for tensors
- [x] Implement SHOW SCHEMA \<dataset\>
- [x] Implement ADD COLUMN for datasets
  - Support computed columns (`ADD COLUMN x = a + b`)
  - Lazy vs materialized column evaluation
- [x] Add indexing syntax in expressions
  - Tensor indexing: `m[0, *]`, `m[:, 1]`
  - Tuple access: `row.field`, `dataset.column`
- [x] Improve expression typing and error messages
- [x] Extend SHOW to fully cover:
  - Scalars, vectors, matrices, tensors
  - Tuples and datasets

---

## Phase 6: Indexing & Access Paths

> Goal: Enable fast lookups and similarity search as first-class concepts.

- [x] Define `Index` trait
- [x] Implement `HashIndex`
  - Exact match lookups
  - Equality-based filters
- [x] Implement `VectorIndex`
  - Cosine similarity
  - Euclidean distance
- [x] Add CREATE INDEX DSL command
  - `CREATE INDEX idx_name ON dataset(column)`
  - `CREATE VECTOR INDEX idx_name ON dataset(embedding)`
- [x] Update INSERT to maintain indices
- [x] Add SHOW INDEXES command

---

## Phase 7: Query Planning & Optimization

> Goal: Move from naive execution to planned execution.

- [ ] Introduce logical query plan
- [ ] Introduce physical execution plan
- [ ] Integrate index-aware execution
- [ ] Implement basic query optimizer:
  - Index selection
  - Predicate pushdown
- [ ] Add EXPLAIN / EXPLAIN PLAN DSL command

---

## Phase 8: Aggregations & GROUP BY

> Goal: Enable analytical workloads comparable to SQL engines.

- [ ] Implement GROUP BY execution
- [ ] Aggregation functions:
  - SUM
  - AVG
  - COUNT
  - MIN / MAX
- [ ] Aggregations over:
  - Scalars
  - Vectors (element-wise)
  - Matrices (axis-based)
- [ ] Support HAVING clause
- [ ] Allow aggregations over computed columns

---

## Phase 9: Persistence Layer (Disk-backed Engine)

> Goal: Make the engine restartable, durable, and production-viable.

- [ ] Define persistence abstraction (StorageEngine trait)
- [ ] Implement Parquet-based storage
- [ ] Serialize:
  - DatasetStore
  - TensorStore
  - Schema metadata
  - Index metadata
- [ ] Implement LOAD / SAVE DSL commands
- [ ] Support partial loading (lazy datasets)
- [ ] Add full persistence test suite

---

## Phase 10: Engine Lifecycle & Instance Management

> Goal: Make VectorDB behave like a real database engine.

- [ ] Engine instances
  - Named databases
  - Isolated DatasetStores
- [ ] CREATE DATABASE / DROP DATABASE
- [ ] USE database
- [ ] Engine configuration file support
- [ ] Startup/shutdown hooks
- [ ] Graceful recovery from disk

---

## Phase 11: CLI & Server Hardening

> Goal: Prepare the engine for public usage.

- [ ] Improve REPL UX
  - Multiline input
  - History
  - Auto-completion
- [ ] Add CLI commands:
  - `vectordb init`
  - `vectordb load`
  - `vectordb serve`
- [ ] Server enhancements:
  - Streaming responses
  - Query timeout
  - Request validation
- [ ] OpenAPI / Swagger documentation

---

## Phase 12: Public Readiness

> Goal: Make the project open-source ready and approachable.

- [ ] Write architectural documentation
- [ ] Add end-to-end examples
- [ ] Benchmark in-memory vs persisted workloads
- [ ] Add contribution guidelines
- [ ] Add security considerations
- [ ] Prepare first public release (v0.1.0)

---

## Phase 13: Project Identity & Naming (LINAL)

This phase formalizes the public identity of the project, aligning naming, tooling, and file formats with the long-term vision of a linear algebra–centric analytical engine.

### Naming Decisions

- **Project Name**: **LINAL**
  - Derived from *Linear Algebra*
  - Emphasizes vectors, matrices, tensors, and numerical computation
  - Technical, unambiguous, and aligned with ML / AI / scientific computing audiences

- **Engine**: LINAL Engine
- **CLI Binary**: `linal`
- **DSL Name**: LINAL Script

### File Extension

- **`.lnl`** is the canonical extension for LINAL scripts
  - Used for experiments, tests, pipelines, and analytical workflows
  - Example:
    ```bash
    linal run analysis.lnl
    ```

### Scope Clarification

LINAL is positioned as:

- An **in-memory analytical engine**
- Focused on:
  - Linear algebra (vectors, matrices, tensors)
  - Structured datasets
  - SQL-inspired querying combined with algebraic operations
- Designed for:
  - Machine Learning
  - AI research
  - Statistical analysis
  - Scientific and numerical computing

### Non-Goals (for now)

- LINAL is **not** positioned as a traditional OLTP database
- LINAL prioritizes **analytical workflows, computation, and expressiveness** over transactional guarantees

### Documentation & Repository Updates

- Update README to reflect LINAL branding
- Rename example scripts to use `.lnl`
- Ensure CLI help and server documentation reference LINAL consistently
- Keep internal architecture unchanged (naming-only refactor)

---

## Long-term Vision (Post v1)

- GPU-backed tensor execution
- Distributed execution
- Columnar execution engine
- Integration with Python / WASM
- Native ML operators (KNN, clustering, PCA)
