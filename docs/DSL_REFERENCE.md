# LINAL DSL Reference

**LINAL Script** is a SQL-inspired language designed for working with tensors, vectors, matrices, and structured datasets. It bridges the gap between traditional data querying and numerical computation.

---

## 1. Core Principles

1. **SQL Familiarity**: Standard SELECT/WHERE/JOIN syntax for data operations.
2. **First-Class Algebra**: Vectors and matrices are treated as primitive types with native operators.
3. **Zero-Copy Semantics**: Operations on large datasets favor references and metadata views over data duplication.
4. **Lineage Tracking**: Automatic tracking of computational history for all derived tensors.

---

## 2. Resource Management

### BIND

Create a local alias for an existing tensor or dataset.

```sql
BIND scores_alias TO original_scores
```

### ATTACH

Link an independent tensor into a dataset column (Zero-copy).

```sql
ATTACH vector_weights TO neural_ds.weights
```

### DERIVE

Create a new tensor from an expression while preserving full provenance.

```sql
DERIVE normalized_v FROM v / 10.0
```

### RESET

Clear the current session (all in-memory tensors and datasets).

```sql
RESET SESSION
```

---

## 3. Tensor & Matrix Operations

### DEFINE / VECTOR / MATRIX

Create new numeric resources.

```sql
VECTOR v = [1.0, 2.0, 3.0]
MATRIX m = [[1, 2], [3, 4]]
DEFINE t = TENSOR(2, 2, 2) [1, 2, 3, 4, 5, 6, 7, 8]
```

### LET

Perform algebraic operations.

```sql
LET res = v * 2.5
LET combined = m + matrix_b
```

### Algebra Handlers

- `MATMUL`: Matrix multiplication
- `TRANSPOSE`: Swap dimensions
- `RESHAPE`: Change shape without copying
- `FLATTEN`: Convert to 1D vector
- `STACK`: Combine tensors along axis

---

## 4. Dataset Operations

### DATASET

Define a new dataset schema.

```sql
DATASET analytics COLUMNS (
    id: Int,
    region: String,
    embedding: Vector(128)
)
```

### INSERT INTO

Add data to a dataset.

```sql
INSERT INTO analytics (id, region, embedding) VALUES (1, "North", [0.1, ...])
```

### SELECT (Querying)

Standard SQL querying with support for tensor columns.

```sql
SELECT region, SUM(embedding) 
FROM analytics 
WHERE region = "North" 
GROUP BY region
```

### ALTER DATASET

Add columns dynamically.

```sql
ALTER DATASET analytics ADD COLUMN total = price * quantity
ALTER DATASET analytics ADD COLUMN score = complex_expr LAZY
```

### MATERIALIZE

Convert a LAZY column into a physical, materialized column.

```sql
MATERIALIZE analytics
```

---

## 5. Introspection & Diagnostics

### SHOW

Inspect any engine resource.

```sql
SHOW v
SHOW SCHEMA analytics
SHOW LINEAGE result
SHOW ALL DATASETS
### SHOW DATASET METADATA

Inspect a dataset's current metadata (including version, origin, timestamps, and custom tags).

```sql
SHOW DATASET METADATA analytics
```

### LIST DATASET VERSIONS

Show the history of schema and metadata changes for a specific dataset.

```sql
LIST DATASET VERSIONS analytics
```

### SET DATASET METADATA

Update specific metadata fields for a dataset. Supported fields: `author`, `description`, `tag` (can be called multiple times to add multiple tags).

```sql
SET DATASET analytics METADATA author = "Nicolas"
SET DATASET analytics METADATA description = "Analysis of Q1 results"
SET DATASET analytics METADATA tag = "production"
SET DATASET analytics METADATA tag = "v1-stable"
```

### AUDIT

Verify the health and referential integrity of a dataset.

```sql
AUDIT DATASET diagnostics
```

### EXPLAIN

Visualize the execution plan for a query.

```sql
EXPLAIN SELECT * FROM users WHERE id > 10
```

---

## 6. Persistence

- `SAVE DATASET <name> [TO "path"]`: Persist dataset and its metadata to Parquet/JSON. If path is omitted, use the managed database directory.
- `SAVE TENSOR <name> [TO "path"]`: Persist tensor data to JSON.
- `LOAD DATASET <target> [FROM "source_path"]`: Recover a dataset. If `FROM` is specified, the dataset is loaded from the source path (which can be a filename or directory) and registered in-memory as `<target>`.
- `LIST DATASETS`: Show all persisted datasets in the current database context.
- `IMPORT CSV FROM "path" AS <name>`: Import CSV with automatic schema inference.
- `EXPORT CSV <name> TO "path"`: Export dataset to CSV format.

---

## 7. Instance Management

Manage multiple isolated database instances.

```sql
CREATE DATABASE research
USE research
DROP DATABASE obsolete_db
SHOW DATABASES
```

---

## 8. Server & Job Management

LINAL supports asynchronous background execution and high-concurrency server modes.

### Background Jobs

Manage long-running DSL commands as isolated background tasks.

- **Submit**: `POST /jobs` (REST API) executes a DSL command in the background.
- **Cancel**: `DELETE /jobs/:id` cancels a pending job.

### Server Commands (CLI)

Control the LINAL server instance directly from the command line.

```bash
# Check if server is running and healthy
linal server status

# Start server on specific port
linal server start --port 8080
```

### High-Concurrency Features

- **Parallel Reads**: Analytical commands (`SHOW`, `SELECT`, `EXPLAIN`, `AUDIT`) run in parallel using shared locks.
- **Thread Safety**: State-modifying operations automatically acquire exclusive write locks to ensure consistency.
- **Graceful Shutdown**: The server safely finishes active requests before exiting on SIGINT/SIGTERM.

---

**LINAL**: *Where SQL meets Linear Algebra.*
