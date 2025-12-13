# ⚡ VectorDB

**VectorDB** is an experimental, in-memory analytical engine built for scientific computing, machine learning, and structured data analysis. It bridges the gap between Relational Databases and Tensor Computation, providing a SQL-like DSL that treats Vectors and Matrices as first-class citizens.

---

## 🌟 Key Capabilities

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
```

### 3. Matrix & Vector Aggregations
VectorDB supports element-wise aggregations on complex types.

```sql
-- Element-wise Summation of Matrices by Region
SELECT region, SUM(features) 
FROM analytics 
GROUP BY region

-- Aggregation on Computed Expressions
-- Scales the matrix by 2 before summing
SELECT region, SUM(features * 2.0) 
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

## 🏗 Architecture

*   **Storage Engine**: In-memory columnar/row hybrid store with specialized indices (HashIndex, VectorIndex).
*   **Query Engine**: Logical -> Physical plan optimization with predicate pushdown.
*   **Type System**: Strong typing with inference for arithmetic expressions (`Matrix + Float = Matrix`).

## � Documentation

*   [DSL Reference](docs/DSL_REFERENCE.md)
*   [Roadmap & Status](docs/ROADMAP.md)
*   [TOON Format](TOON_FORMAT.md)

---

**VectorDB**: *Where SQL meets Linear Algebra.*
