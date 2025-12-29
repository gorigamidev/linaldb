# ⚡ LINAL

**LINAL** is an experimental, in-memory analytical engine built for scientific computing, machine learning, and structured data analysis. It bridges the gap between Relational Databases and Tensor Computation, providing a SQL-like DSL that treats Vectors and Matrices as first-class citizens.

---

## Key Capabilities

### 1. Hybrid Data Model

Store structured data (Integers, Strings) alongside mathematical types (Vectors, Matrices) in the same dataset.

```rust
DATASET analytics COLUMNS (
    id: Int,
    region: String,
    features: Matrix(4, 4),  // 4x4 Matrix
    embedding: Vector(128)   // 128-dim Vector
)
```

### 2. Dataset as Reference Graph (New!)

Datasets in LINAL are now formal **Reference Graphs**. They serve as semantic views over existing data, referencing actual tensors or other dataset columns without duplication. This enables powerful features like virtual datasets and shared data structures.

```sql
-- Create tensors
VECTOR prices = [10.0, 20.0, 30.0]
VECTOR qtys   = [5.0, 10.0, 2.0]

-- Create a zero-copy dataset view
-- This stores ResourceReferences, not the data itself.
LET ds = dataset("sales_view")
ds.add_column("price", prices)
ds.add_column("quantity", qtys)

-- Derived Views: Datasets can point to other datasets
-- ds2.total -> ds1.price * ds1.quantity (Zero-copy chain)
LET revenue = ds.price * ds.quantity
ds.add_column("total", revenue)

-- Materialize & Save to Parquet
-- Regular SAVE materializes into Parquet for portability.
SAVE DATASET sales_view TO "sales.parquet"
```

### 3. Analytical DSL

Perform complex selection, filtering, and aggregation on all data types.

```sql
-- Select specific columns including matrix data
SELECT region, features FROM analytics LIMIT 5

-- Filter using standard predicates
SELECT * FROM analytics WHERE region = "North"

-- Add computed columns dynamically
DATASET analytics ADD COLUMN total = price * quantity
DATASET analytics ADD COLUMN discount_price = price - discount

-- Schema introspection
SHOW SCHEMA analytics
```

### 3. Matrix & Vector Aggregations

LINAL supports element-wise aggregations on complex types with full SQL-like aggregation functions.

```sql
-- Element-wise Summation of Matrices by Region
SELECT region, SUM(features) 
FROM analytics 
GROUP BY region

-- Average (AVG) aggregation with proper sum/count tracking
SELECT region, AVG(score) 
FROM analytics 
GROUP BY region

-- Aggregation on Computed Expressions
-- Scales the matrix by 2 before summing
SELECT region, SUM(features * 2.0) 
FROM analytics 
GROUP BY region

-- Full aggregation suite: SUM, AVG, COUNT, MIN, MAX
SELECT region, 
       SUM(amount) as total,
       AVG(amount) as average,
       COUNT(*) as count,
       MIN(amount) as minimum,
       MAX(amount) as maximum
FROM analytics 
GROUP BY region
```

### 4. Vector Similarity Search

Built-in support for vector indexing and similarity search (KNN).

```sql
-- Create a Vector Index
CREATE VECTOR INDEX emb_idx ON analytics(embedding)

-- Find top 5 similar vectors
SEARCH analytics 
WHERE embedding ~= [0.1, 0.2, ... 128 values ...] 
LIMIT 5
```

### 5. Multi-Database Engine

Manage multiple isolated database instances within a single cluster.

```sql
-- Create and switch context
CREATE DATABASE analytics
USE analytics

-- List all databases
SHOW DATABASES

-- Data isolation: users in 'default' != users in 'analytics'
USE default
SHOW ALL DATASETS
```

### 6. Robust Metadata System

Add custom, searchable metadata and versioning to your datasets.

```sql
-- Set custom metadata tags and versions
SET DATASET users METADATA version = "1.0.0"
SET DATASET users METADATA tags = "scientific,test"

-- Metadata is automatically tracked with ISO-8601 timestamps
-- (created_at, updated_at) and custom 'extra' fields.
```

### 7. Explicit Semantics & Lineage (New!)

LINAL now supports declarative resource management and built-in execution lineage.

* **BIND**: Create zero-copy aliases for tensors or datasets.

    ```sql
    BIND scores_alias TO original_scores
    ```

* **ATTACH**: Declaratively link independent tensors into dataset columns.

    ```sql
    ATTACH vector_weights TO neural_ds.weights
    ```

* **DERIVE**: Explicitly compute new tensors while preserving full provenance (Lineage).

    ```sql
    DERIVE normalized_v FROM v / 10.0
    ```

#### Execution Lineage

Every tensor created via `DERIVE` or `LET` automatically tracks its origin, operation, and input IDs. This metadata is persistent and can be inspected to audit analytical workflows.

## Multi-Paradigm Access

LINAL provides a unified interface across all access methods.

### 1. Interactive REPL (Shell)

Designed for live data exploration. Supports command history and flexible output formatting.

```bash
# Start the interactive shell
cargo run -- repl

# Use machine-readable TOON output
cargo run -- repl --format=toon
```

### 2. Script Execution (Automation)

The `run` command executes `.lnl` script files. Scripts support **multi-line commands** (e.g., complex `DATASET` definitions) using balanced parentheses logic.

```bash
# Run a script file
cargo run -- run examples/smoke_test.lnl

# Scripts handle multi-line definitions gracefully:
# DATASET users COLUMNS (
#    id: INT,
#    embedding: VECTOR(128)
# )
```

### 3. HTTP Server

```bash
# Start server on port 8080
cargo run -- server --port 8080
```

**Server API** - Unified interface with DSL commands and flexible output:

*Input:* Raw DSL commands (text/plain)

```bash
# TOON response (default)
curl -X POST "http://localhost:8080/execute" \
  -H "Content-Type: text/plain" \
  -d "VECTOR v = [1, 2, 3]"

# JSON response (opt-in)
curl -X POST "http://localhost:8080/execute?format=json" \
  -H "Content-Type: text/plain" \
  -d "SHOW v"
```

*Response Formats:*

* **TOON** (default): Token-Oriented Object Notation - human and machine readable
* **JSON** (opt-in): Standard JSON format via `?format=json` query parameter

---

## Recent Features

### DSL Semantic Expansion & Lineage (v0.1.7)

Introduction of formal semantics to improve legibility and traceability:

* **Declarative Keywords**: Added `BIND`, `ATTACH`, and `DERIVE` for explicit resource handling.
* **Lineage Tracking**: Automatic recording of tensor derivation history (Phase 3 & 4 of formalization roadmap).
* **Persistent Provenance**: Lineage metadata is serialized/deserialized ensuring audits survive database restarts.

### CLI & Server Hardening (v0.1.4 - LINAL)

Significant improvements to the user experience and engine robustness:

**Professional REPL (LINAL Shell):**

* Integrated `rustyline` for persistent command history and multi-line support.
* Balanced parentheses logic for entering complex datasets directly in the REPL.
* `colored` output for improved readability and error reporting.

**Administrative CLI Commands:**

* `linal init`: Automated setup for `./data` and `linal.toml`.
* `linal load <file> <dataset>`: Direct Parquet ingestion via CLI.
* `linal serve`: Shorthand for starting the HTTP server.

**Server Robustness & API Docs:**

* **Query Timeouts**: Long-running queries automatically cancel after 30s.
* **Request Validation**: Size limits and non-empty checks for all incoming commands.
* **OpenAPI / Swagger UI**: Built-in interactive documentation available at `/swagger-ui`.

---

### Interface Standardization (v0.1.2)

Unified interface across all access methods with flexible output formats:

**Server API Improvements:**

* Raw DSL commands as input (no JSON wrapper needed)
* TOON format as default output
* JSON format available via `?format=json` query parameter
* Backward compatible with legacy JSON request format

**CLI Enhancements:**

* `--format` flag for REPL and script execution
* Choose between `display` (human-readable) or `toon` (machine-readable) output
* Perfect for automation and scripting workflows

**Example:**

```bash
# CLI with TOON output
cargo run -- repl --format=toon

# Server with JSON response
curl -X POST "http://localhost:8080/execute?format=json" \
  -H "Content-Type: text/plain" \
  -d "SELECT * FROM users LIMIT 5"
```

### AVG Aggregation (v0.1.2)

Full implementation of AVG aggregation with proper sum/count tracking:

* Supports Int, Float, Vector, and Matrix types
* Automatic type conversion (Int → Float for precision)
* Works with GROUP BY and computed expressions

### Computed Columns (v0.1.2)

Add computed columns dynamically using expressions with support for both materialized and lazy evaluation:

**Materialized Columns** (evaluated immediately):

```sql
-- Add computed column with expression
DATASET products ADD COLUMN total = price * quantity
DATASET sales ADD COLUMN profit = revenue - cost
```

**Lazy Columns** (evaluated on access):

```sql
-- Lazy columns store the expression and evaluate only when accessed
DATASET analytics ADD COLUMN total = price * quantity LAZY
DATASET analytics ADD COLUMN complex_calc = (a + b) * c LAZY

-- Query results automatically evaluate lazy columns
SELECT total, complex_calc FROM analytics

-- Materialize lazy columns to convert them to regular columns
MATERIALIZE analytics
```

### Lazy Column Evaluation (v0.1.2)

Lazy columns provide on-demand computation, storing expressions instead of pre-computed values:

* **Storage Efficiency**: Only store expressions, not computed values
* **On-Demand Evaluation**: Values computed when accessed in queries
* **Materialization**: Convert lazy columns to materialized with `MATERIALIZE` command
* **Automatic Evaluation**: Query execution automatically evaluates lazy columns

### Schema Introspection

```sql
-- View dataset schema
SHOW SCHEMA analytics

-- List all datasets
SHOW ALL DATASETS

-- View all indexes
SHOW INDEXES analytics
```

### Persistence & Lifecycle (v0.1.3)

Native disk-backed persistence with automated database discovery and configuration.

**Database Management:**

```sql
CREATE DATABASE research
USE research
DROP DATABASE obsolete_db
SHOW DATABASES
```

**Dataset & Tensor Persistence:**

```sql
-- Save data (defaults to database-specific path in linal.toml)
SAVE DATASET users
SAVE TENSOR weights

-- Load data back
LOAD DATASET users
LOAD TENSOR weights

-- List what's on disk
LIST DATASETS
LIST TENSORS
```

**Configuration (`linal.toml`):**
Customize the engine behavior and storage locations.

```toml
[storage]
data_dir = "./data"
default_db = "default"
```

**Key Features:**

* **Auto-Discovery**: Engine automatically discovers and recovers databases from `data_dir` on startup.
* **Database Isolation**: Persistence is siloed per database (e.g., `./data/analytics/` vs `./data/default/`).
* **Standard Formats**: Datasets use **Apache Parquet** for efficiency; Tensors use JSON for weights and metadata.
* **Seamless Recovery**: Databases created in one session are immediately available in the next.

---

## Quick Start

### Installation

```bash
git clone https://github.com/yourusername/linal.git
cd linal
cargo build --release
```

### Running Examples

We have several example scripts demonstrating LINAL's capabilities:

```bash
# Comprehensive feature demo
cargo run -- run examples/features_demo.lnl

# End-to-end workflow example
cargo run -- run examples/end_to_end.lnl

# Performance benchmark
cargo run -- run examples/benchmark.lnl
```

### Interactive REPL

```bash
$ cargo run
> DEFINE v = [1, 2, 3]
> SHOW v + 1
[2, 3, 4]
```

---

## Architecture

* **Storage Engine**: In-memory columnar/row hybrid store with specialized indices (HashIndex, VectorIndex).
* **Query Engine**: Logical -> Physical plan optimization with predicate pushdown and index-aware execution.
* **Type System**: Strong typing with inference for arithmetic expressions (`Matrix + Float = Matrix`).
* **Aggregation Engine**: Full SQL aggregation support (SUM, AVG, COUNT, MIN, MAX) with element-wise operations on vectors and matrices.
* **Schema Evolution**: Dynamic column addition with computed columns support (`ADD COLUMN x = expression`).

## Documentation

* [Architecture](docs/ARCHITECTURE.md) - System architecture and design
* [DSL Reference](docs/DSL_REFERENCE.md) - Complete DSL command reference
* [Roadmap & Status](docs/Tasks_implementations.md) - Development roadmap
* [Changelog](CHANGELOG.md) - Version history
* [Contributing](CONTRIBUTING.md) - How to contribute
* [Security](SECURITY.md) - Security policy and considerations

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2025 gorigami (gorigami.xyz)

---

**LINAL**: *Where SQL meets Linear Algebra.*
