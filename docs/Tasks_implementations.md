# LINAL Implementation History (Historical Record)

> [!NOTE]
> This document serves as a historical record of the development phases of LINAL. For the current system architecture, please refer to [ARCHITECTURE.md](ARCHITECTURE.md).

## Phase 0: Preparation

- [x] Fix Cargo.toml edition (2024 → 2021)
- [x] Add ADD COLUMN for datasets (with DEFAULT values and nullable support)
- [x] Add GROUP BY with aggregations (SUM, AVG, COUNT, MIN, MAX)
- [x] Add matrix operations (MATMUL, TRANSPOSE, RESHAPE, FLATTEN)
- [x] Add indexing syntax (m[0, *], tuple.field, dataset.column)
- [x] Update SHOW command for all types (tensors, datasets, schemas, indexes)
- [x] Add SHOW SHAPE introspection
- [x] Add SHOW SCHEMA introspection

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

# LINAL / Tensor Engine – Extended Roadmap (Post-Phase 5.5)

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
  - [x] Support computed columns (`ADD COLUMN x = a + b`) - Materialized evaluation ✅
  - [x] Lazy vs materialized column evaluation ✅
    - [x] Materialized evaluation (immediate computation) ✅
    - [x] Lazy evaluation (`ADD COLUMN x = expr LAZY`) ✅
    - [x] Automatic lazy evaluation in queries ✅
    - [x] MATERIALIZE command to convert lazy to materialized ✅
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

- [x] Introduce logical query plan
- [x] Introduce physical execution plan
- [x] Integrate index-aware execution
- [x] Implement basic query optimizer:
  - Index selection
  - Predicate pushdown
- [x] Add EXPLAIN / EXPLAIN PLAN DSL command

---

## Phase 8: Aggregations & GROUP BY

> Goal: Enable analytical workloads comparable to SQL engines.

- [x] Implement GROUP BY execution
- [x] Aggregation functions:
  - SUM ✅
  - AVG ✅ (fully implemented with sum/count tracking)
  - COUNT ✅
  - MIN / MAX ✅
- [X] Aggregations over:
  - Scalars
  - Vectors (element-wise)
  - Matrices (axis-based)
- [x] Support HAVING clause
- [X] Allow aggregations over computed columns

---

## Phase 8.5: Interface Standardization (TOON + DSL) ✅

> Goal: Standardize all interfaces to use DSL commands as input and TOON as output format.

- [x] Server API refactor
  - Accept raw DSL text via `text/plain` content type
  - Maintain JSON backward compatibility with deprecation warnings
  - Always return TOON format
- [x] CLI enhancement
  - Add `--format` flag to REPL and Run commands
  - Support `display` (default) and `toon` output formats
  - Implement TOON serialization in CLI
- [x] Test updates
  - Update server tests to use raw DSL
  - Add backward compatibility test for JSON
  - Verify all 113 tests passing
- [x] Documentation
  - Create walkthrough document
  - Document migration path from JSON to raw DSL

### Future Enhancement: Response Format Selection ✅

- [x] Add response format parameter to server API
  - Support `?format=toon` (default) and `?format=json` query parameters
  - Allow clients to request JSON response format if needed
  - Implement JSON serialization for `DslOutput`
- [x] Update server endpoint signature
  - Parse query parameters for format selection
  - Serialize response based on requested format
  - Maintain TOON as default for consistency

---

## Phase 9: Persistence Layer (Disk-backed Engine) ✅

> Goal: Make the engine restartable, durable, and production-viable.

- [x] Define persistence abstraction (StorageEngine trait)
- [x] Implement Parquet-based storage
- [x] Serialize:
  - [x] DatasetStore (Parquet + JSON metadata)
  - [x] TensorStore (JSON format)
  - [x] Schema metadata
- [x] Implement SAVE DSL commands
  - [x] SAVE DATASET
  - [x] SAVE TENSOR
- [x] Implement LOAD DSL commands
  - [x] LOAD DATASET (Parquet -> Dataset conversion)
  - [x] LOAD TENSOR
- [x] Implement LIST commands
  - [x] LIST DATASETS
  - [x] LIST TENSORS
- [x] Add full persistence test suite

---

## Phase 10: Engine Lifecycle & Instance Management ✅

> Goal: Make LINAL behave like a real database engine.

- [x] Engine instances
  - [x] Named databases
  - [x] Isolated DatasetStores
- [x] CREATE DATABASE / DROP DATABASE
- [x] USE database
- [x] Engine configuration file support (`linal.toml`)
- [x] Startup/shutdown hooks (Startup recovery implemented)
- [x] Graceful recovery from disk

## Phase 10.5: Robust Metadata System ✅

- [x] Add `chrono` dependency for ISO timestamps
- [x] Update `DatasetMetadata` struct with versioning, `updated_at`, and `extra` fields
- [x] Implement `SET DATASET METADATA` DSL command
- [x] Verify persistence and readability of new metadata format

## Phase 10.6: CLI Parity & Multi-line Support ✅

- [x] Refactor script runner for multi-line support
- [x] Add routing for `ALTER DATASET` in DSL
- [x] Fix `GROUP BY` type inference for grouping columns
- [x] Verify all past features via comprehensive smoke test

---

## Phase 11: CLI & Server Hardening ✅

> Goal: Prepare the engine for public usage.

- [x] Improve REPL UX
  - [x] Multiline input
  - [x] History
  - [x] Auto-completion
- [x] Add CLI commands:
  - [x] `linal init`
  - [x] `linal load`
  - [x] `linal serve`
- [x] Server enhancements:
  - [x] Query timeout (30s)
  - [x] Request validation (Size limits, non-empty checks)
- [x] OpenAPI / Swagger documentation (`/swagger-ui`)

---

## Phase 12: Public Readiness ✅

> Goal: Make the project open-source ready and approachable.

- [x] Write architectural documentation
- [x] Add end-to-end examples
- [x] Benchmark in-memory vs persisted workloads
- [x] Add contribution guidelines
- [x] Add security considerations
- [x] Prepare first public release (v0.1.0)

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
