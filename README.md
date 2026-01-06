# ⚡ LINALDB: The Tensor-First Analytical Engine

**LINALDB** is a high-performance, in-memory analytical engine built to bridge the gap between relational data engineering and scientific computing. It provides a SQL-inspired DSL that treats vectors, matrices, and multi-dimensional tensors as first-class citizens.

---

## One Mental Model: SQL meets Linear Algebra

LINALDB is designed for developers and researchers who need the structure of a database with the mathematical power of a tensor library.

- **Semantic Transformations**: Build zero-copy views using Reference Graphs.
- **Algebraic DSL**: Perform complex matrix and vector math directly in your queries.
- **Local-First & Portable**: Use it as an embedded library (like SQLite) or a managed server.
- **Enterprise Ready**: High-concurrency reads (`RwLock`), asynchronous background jobs, and multi-tenant isolation.

---

## 30-Second Quick Start

Get LINALDB running on your machine:

```bash
# Clone and build
git clone https://github.com/gorigami/linaldb.git
cd linaldb && cargo build --release

# 1. Start the interactive REPL
./target/release/linaldb repl

# 2. Run a smoke test script
./target/release/linaldb run examples/end_to_end.lnl

# 3. Start the managed server
./target/release/linaldb serve --port 8080
```

---

## Core Capabilities

### 1. Unified Hybrid Data Model

Store structured fields alongside high-dimensional tensors in the same dataset.

```sql
DATASET diagnostics COLUMNS (
    id: Int,
    region: String,
    features: Matrix(4, 4),  -- Native Matrix support
    embedding: Vector(128)   -- Native Vector support
)
```

### 2. Zero-Copy Reference Graphs

Create semantic views without duplicating data. LINALDB tracks lineage and provenance automatically.

```sql
-- Create a zero-copy alias
BIND scores_alias TO original_scores

-- Attach independent tensors to a dataset
ATTACH weights_vec TO model_ds.weights

-- Derive new resources with full lineage
DERIVE normalized FROM v / 10.0
```

### 3. High-Concurrency Analytics

Multi-platform server with parallel execution and background workload management.

```bash
# Check server health
linaldb server status

# Submit a long-running job to the background
curl -X POST "http://localhost:8080/jobs" -d "SHOW ALL"
```

---

## 📖 Documentation Hub

LINALDB is extensively documented to help you scale from local experiments to production services.

- **[Architecture](docs/ARCHITECTURE.md)**: Deep dive into the internal engine design.
- **[DSL Reference](docs/DSL_REFERENCE.md)**: Complete guide to keywords, operators, and syntax.
- **[Performance & Benchmarks](docs/BENCHMARKS.md)**: How we achieve 2.5x speedups via SIMD and Rayon.
- **[Example Gallery](docs/EXAMPLES.md)**: Curated snippets for common ML and analytical workflows.
- **[Error Reference](docs/ERROR_REFERENCE.md)**: Troubleshooting guide for engine and DSL errors.

---

## ⚖️ License

LINALDB is licensed under the **LinalDB Community License v1.0**.

- ✅ **Permitted**: Personal use, research, education, and internal organizational use.
- ⚠️ **Restricted**: Commercial redistribution, managed services (DBaaS/SaaS), and direct monetization require a separate **Commercial License**.

For commercial licensing inquiries, please contact: [develop@gorigami.xyz](mailto:develop@gorigami.xyz)

---

**LINALDB**: *Where SQL meets Linear Algebra.*
Copyright (c) 2025 gorigami (gorigami.xyz)
