# 🎨 LINAL Example Gallery

Discover how to use LINAL for various analytical, numerical, and machine learning workloads. All referenced scripts can be found in the `examples/` directory.

---

## 🏗️ 1. Core Analytics Pipeline

Learn how to move from raw data to insights using LINAL's data portability features.

### [Data Ingestion & Export](file:///Users/nicolasbalaguera/dev/linaldb/linal-db-rs/examples/export_import_csv.lnl)

Demonstrates schema inference, CSV import/export, and session management.

```sql
-- Infer schema and import
IMPORT CSV FROM "./data/sample_data.csv" AS frameworks

-- Materialize a computed column and export
ALTER DATASET frameworks ADD COLUMN score = value * 100
MATERIALIZE frameworks
EXPORT CSV frameworks TO "./data/exported_results.csv"
```

### [End-to-End Workflow](file:///Users/nicolasbalaguera/dev/linaldb/linal-db-rs/examples/end_to_end.lnl)

A comprehensive pipeline: Database creation → Feature engineering → Parquet persistence.

---

## 🧮 2. Numerical Computation

LINAL's specialized kernels make heavy linear algebra expressive and fast.

### [Matrix & Vector Operations](file:///Users/nicolasbalaguera/dev/linaldb/linal-db-rs/examples/matrix_operations.lnl)

Focuses on the algebraic DSL: `MATMUL`, `TRANSPOSE`, and `RESHAPE`.

```sql
MATRIX A = [[1, 2, 3], [4, 5, 6]]
MATRIX B = [[7, 8], [9, 10], [11, 12]]

-- Matrix multiplication (2x3 * 3x2 = 2x2)
LET C = MATMUL A B

-- Transpose and Flatten
LET A_T = TRANSPOSE A
LET flat = FLATTEN A_T
```

---

## 🔗 3. Semantic Reference Graphs

Leverage zero-copy architecture to build complex data relationships without memory overhead.

### [Zero-Copy Provenance](file:///Users/nicolasbalaguera/dev/linaldb/linal-db-rs/examples/reference_graph.lnl)

Demonstrates how `dataset` objects act as a view over independent tensors and datasets.

```sql
LET raw = dataset("raw_metrics")
raw.add_column("temp", v_temp)
raw.add_column("pressure", v_pressure)

-- Create a derived view without copying data
DERIVE normalized FROM v_temp / 100.0
BIND alias_ds TO raw
```

---

## 🌐 4. Managed Service & Background Jobs

Scale your analysis using LINAL's server-side execution and job management.

### [Managed Service Demo](file:///Users/nicolasbalaguera/dev/linaldb/linal-db-rs/examples/managed_service_demo.lnl)

Shows how to use multi-tenant database contexts and remote execution.

### Asynchronous Background Jobs

Submit long-running tasks via the REST API to keep your client responsive.

```bash
# Submit a background analytics job
curl -X POST "http://localhost:8080/jobs" \
     -H "Content-Type: text/plain" \
     -d "SELECT region, AVG(score) FROM results GROUP BY region"

# Response: {"job_id": "8c3f..."}
```

---

## 🔍 5. Introspection & Auditing

Maintain data quality with first-class lineage and integrity tools.

### [Lineage & Audit Demo](file:///Users/nicolasbalaguera/dev/linaldb/linal-db-rs/examples/introspection_demo.lnl)

Trace the history of any result and verify reference graph integrity.

```sql
-- Trace where a tensor came from
SHOW LINEAGE weighted_sum

-- Audit a dataset for dangling references
AUDIT DATASET clinical_results
```

---

*For technical implementation details, see [ARCHITECTURE.md](ARCHITECTURE.md).*
