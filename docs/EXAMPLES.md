# LINAL Example Gallery

Discover how to use LINAL for various analytical and numerical workloads. All examples listed here can be found in the `examples/` directory.

---

## 1. Core Workflows

### [End-to-End Analytics](file:///Users/nicolasbalaguera/dev/linaldb/linal-db-rs/examples/end_to_end.lnl)

Demonstrates a complete pipeline: creating a database, loading data, performing vector math, and saving the results to Parquet.

```bash
linal run examples/end_to_end.lnl
```

### [Feature Demo](file:///Users/nicolasbalaguera/dev/linaldb/linal-db-rs/examples/features_demo.lnl)

A tour of all major LINAL capabilities, from `DERIVE` to `GROUP BY` aggregations.

---

## 2. Numerical Computation

### [Algebraic Ops](file:///Users/nicolasbalaguera/dev/linaldb/linal-db-rs/examples/matrix_ops_demo.lnl)

Focuses on `MATMUL`, `TRANSPOSE`, and complex tensor transformations.

```bash
linal run examples/matrix_ops_demo.lnl
```

### [Similarity Search](file:///Users/nicolasbalaguera/dev/linaldb/linal-db-rs/examples/vector_search_demo.lnl)

Shows how to create vector indexes and perform similarity lookups using the `SEARCH` command.

---

## 3. Introspection & Auditing

---

## 4. Server & Concurrency

### Background Job Execution

Learn how to submit long-running analytical tasks to the server's background scheduler and poll for results. See [DSL_REFERENCE.md](DSL_REFERENCE.md#8-server--job-management) for details.

```bash
# Submit a job
curl -X POST "http://localhost:8080/jobs" -d "SHOW ALL"
```

---

## 5. Quick Start Snippets

### Calculate Cosine Similarity

```sql
VECTOR a = [1, 0, 0]
VECTOR b = [0, 1, 0]
LET sim = COSINE_SIMILARITY(a, b)
SHOW sim
```

### Aggregate Vector Data

```sql
SELECT region, AVG(features) 
FROM sales_data 
GROUP BY region
```

---

*For more implementation details, see [ARCHITECTURE.md](ARCHITECTURE.md).*
