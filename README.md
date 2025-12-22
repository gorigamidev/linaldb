# ⚡ VectorDB

**VectorDB** is an experimental, in-memory analytical engine built for scientific computing, machine learning, and structured data analysis. It bridges the gap between Relational Databases and Tensor Computation, providing a SQL-like DSL that treats Vectors and Matrices as first-class citizens.

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

### 2. Analytical DSL
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
VectorDB supports element-wise aggregations on complex types with full SQL-like aggregation functions.

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

### 5. Multi-Paradigm Access
*   **REPL**: Interactive shell for exploration (`cargo run`).
*   **Scripting**: Run `.vdb` files (`cargo run -- run script.vdb`).
*   **HTTP Server**: JSON-based API returning **TOON** (Token-Oriented Object Notation) for efficient LLM integration.

---

## Recent Features

### AVG Aggregation (v0.1.2)
Full implementation of AVG aggregation with proper sum/count tracking:
- Supports Int, Float, Vector, and Matrix types
- Automatic type conversion (Int → Float for precision)
- Works with GROUP BY and computed expressions

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
- **Storage Efficiency**: Only store expressions, not computed values
- **On-Demand Evaluation**: Values computed when accessed in queries
- **Materialization**: Convert lazy columns to materialized with `MATERIALIZE` command
- **Automatic Evaluation**: Query execution automatically evaluates lazy columns

### Schema Introspection
```sql
-- View dataset schema
SHOW SCHEMA analytics

-- List all datasets
SHOW ALL DATASETS

-- View all indexes
SHOW INDEXES analytics
```

---

## Quick Start

### Installation
```bash
git clone https://github.com/yourusername/vector-db-rs.git
cd vector-db-rs
cargo build --release
```

### Running the Example
We have a comprehensive feature demo script included.
```bash
# Run the features demo
cargo run -- run examples/features_demo.vdb
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

*   **Storage Engine**: In-memory columnar/row hybrid store with specialized indices (HashIndex, VectorIndex).
*   **Query Engine**: Logical -> Physical plan optimization with predicate pushdown and index-aware execution.
*   **Type System**: Strong typing with inference for arithmetic expressions (`Matrix + Float = Matrix`).
*   **Aggregation Engine**: Full SQL aggregation support (SUM, AVG, COUNT, MIN, MAX) with element-wise operations on vectors and matrices.
*   **Schema Evolution**: Dynamic column addition with computed columns support (`ADD COLUMN x = expression`).

## Documentation

*   [DSL Reference](docs/DSL_REFERENCE.md)
*   [Roadmap & Status](docs/ROADMAP.md)
*   [TOON Format](TOON_FORMAT.md)

---

**VectorDB**: *Where SQL meets Linear Algebra.*
