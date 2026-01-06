# LINAL DSL Reference

**LINAL Script** is a high-performance, SQL-inspired language for tensor algebra and relational analytics. This document serves as the complete technical specification for all keywords, operators, and built-in functions.

---

## 1. Data Types & Literals

LINAL supports both standard relational types and multi-dimensional numeric structures.

### Relational Types

- `Int`: 64-bit signed integer.
- `Float`: 32-bit floating point (standard for tensor values).
- `String`: UTF-8 character sequence.
- `Bool`: `true` or `false`.
- `Null`: Represents a missing value. Use the `?` suffix in `DATASET` definitions for nullable columns (e.g., `score: Float?`).

### Tensor Types

Defined with specific dimensionality:

- `Vector(N)`: A 1D tensor with `N` elements.
- `Matrix(R, C)`: A 2D tensor with `R` rows and `C` columns.
- `Tensor(d1, d2, ...)`: An N-dimensional tensor.

---

## 2. Resource Definition

Create and initialize numeric resources and structured schemas.

### VECTOR / MATRIX

Quick shorthand for defining tensors.

```sql
VECTOR v = [1.0, 2.0, 3.0]
MATRIX m = [[1, 2], [3, 4]]
```

### DEFINE

Explicit tensor definition for higher dimensions.

```sql
DEFINE t AS TENSOR(2, 2, 2) VALUES [1, 2, 3, 4, 5, 6, 7, 8]
```

### DATASET

Define a persistent relational structure.

```sql
DATASET diagnostics COLUMNS (
    id: Int,
    region: String,
    score: Float?,           -- Nullable column
    features: Vector(128)    -- Embedded tensor
)
```

---

## 3. Numerical DSL (Core Algebra)

LINAL provides two ways to perform math: Functional keywords and Infix operators.

### Functional Keywords

- `ADD a b`: Element-wise addition.
- `SUBTRACT a b`: Element-wise subtraction.
- `MULTIPLY a b`: Element-wise multiplication (Hadamard product).
- `DIVIDE a b`: Element-wise division.
- `MATMUL a b`: Standard matrix multiplication.
- `TRANSPOSE a`: Swap dimensions of a matrix/tensor.
- `RESHAPE a TO [dims]`: Change shape without copying data.
- `FLATTEN a`: Convert multidimensional tensor to a 1D vector.
- `NORMALIZE a`: Scales vector to unit length (L2 norm).
- `SCALE a BY n`: Multiplies all elements by a scalar `n`.
- `STACK t1 t2 ...`: Combines tensors along Axis 0.

### Infix Operators

Standard math notation for scalar and tensor variables:

```sql
LET result = (v_a + v_b) / 2.0
LET scaled = m_a * 10
```

### Advanced Operators

- `CORRELATE a WITH b`: Pearson correlation between two vectors.
- `SIMILARITY a WITH b`: Cosine similarity score [-1.0, 1.0].
- `DISTANCE a TO b`: Euclidean distance between points.

---

## 4. Query & Engineering (SQL)

### SELECT

Query datasets with familiar syntax.

```sql
SELECT region, AVG(score) 
FROM diagnostics 
WHERE id > 100 
GROUP BY region 
HAVING AVG(score) > 0.5 
LIMIT 10
```

- **Aggregate Functions**: `SUM`, `AVG`, `COUNT`, `MIN`, `MAX`.
- **Filtering**: `WHERE` or `FILTER` can be used interchangeably.

### Semantic Transforms (Zero-Copy)

- `BIND alias TO resource`: Create a semantic link (alias) to a tensor or dataset.
- `ATTACH tensor TO ds.col`: Link an independent tensor into a dataset column.
- `DERIVE target FROM expr`: Create a new resource with full automated lineage tracking.

### Schema Evolution

- `ALTER DATASET ds ADD COLUMN col: type [DEFAULT val]`
- `ALTER DATASET ds ADD COLUMN col = expression [LAZY]`
- `MATERIALIZE ds`: Physicalize all `LAZY` columns in a dataset.

---

## 5. Persistence & Ingestion

Load and save data across different formats.

- `IMPORT CSV FROM "path" AS name`: Auto-infer schema and load CSV.
- `EXPORT CSV name TO "path"`: Save dataset to CSV.
- `SAVE DATASET name [TO "path"]`: Persist to Parquet (includes metadata/lineage).
- `LOAD DATASET name [FROM "path"]`: Restore a persisted dataset.
- `LIST DATASETS`: Show available datasets in the current database context.

---

## 6. Instance & Session Management

### Database Management

LINAL supports multi-platform isolated instances.

```sql
CREATE DATABASE research
USE research
DROP DATABASE obsolete_db
SHOW DATABASES
```

### RESET SESSION

Clears all in-memory registers (Tensors and Datasets) for the current session.

---

## 7. Diagnostics

- `SHOW <name>`: Display contents/schema of any resource.
- `SHOW LINEAGE <name>`: Display the graph of computations that produced the resource.
- `SHOW SCHEMA <dataset>`: Display column names and types.
- `EXPLAIN <query>`: Show the logical execution plan.
- `AUDIT DATASET <name>`: Perform a health check on referential integrity.

---

## 8. Server & Job Management

For remote execution and production workloads.

- **Background Jobs**: Submit commands via `POST /jobs` to get a `job_id`.
- **Status Polling**: `GET /jobs/:id` returns `Pending`, `Running`, or `Completed`.
- **Graceful Shutdown**: Server handles `SIGINT`/`SIGTERM` to safely close connections.

---

**LINALDB**: *Where SQL meets Linear Algebra.*
Copyright (c) 2025 gorigami (gorigami.xyz)
Licensed under the LinalDB Community License v1.0
