# TensorDB DSL Reference

**TensorDB DSL** is a human-oriented language for working with tensors, vectors, matrices, tuples and datasets using a declarative syntax.

It combines:
- The structure of SQL
- The expressiveness of linear algebra
- The clarity of dataset-based workflows

into a single, coherent model.

---

## Design Philosophy

TensorDB DSL follows three core rules:

### 1. Logical order over historical syntax

Instructions are written in the order a human reasons about data:

```
DATASET → FILTER → SELECT → ORDER → LIMIT
```

**Not** in reversed SQL order.

### 2. Strong abstraction without hiding meaning

- No hidden side effects
- No overloaded symbols
- Operations express intent clearly:

```txt
FILTER age > 25
SELECT id, score
```

### 3. One language, multiple levels

Same language for:
- Math experiments
- Feature engineering
- Analytics workflows
- Dataset transformations

---

## Core Types

### Tensor

Base numeric structure.

```txt
TENSOR weights [3, 4] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
```

May be:
- `STRICT` – enforces shape compatibility
- `NORMAL` – supports relaxed broadcasting (default)

### Vector

Specialized tensor (rank-1):

```txt
VECTOR v = [1, 2, 3]
```

Equivalent to:

```txt
TENSOR v [3] = [1, 2, 3]
```

### Matrix

Two-dimensional tensor (rank-2):

```txt
MATRIX m = [
  [1, 2],
  [3, 4]
]
```

### Tuple

Structured record:

```txt
TUPLE user = (id: 1, age: 25, score: 0.8)
```

Tuples are used as:
- Dataset rows
- Composite keys
- Structured parameters

### Dataset

Table-like structure over tuples:

```txt
DATASET users COLUMNS (id, age, score) = [
  (1, 25, 0.9),
  (2, 31, 0.7)
]
```

---

## Transformations

Transformations are written as **blocks** instead of pipelines.

### Filtering and Projection

```txt
DATASET result FROM users
FILTER age > 20
SELECT id, score
ORDER BY score DESC
```

Equivalent to SQL:

```sql
SELECT id, score FROM users WHERE age > 20 ORDER BY score DESC
```

but written in **logical order**.

### Mathematical Operations

```txt
sum = a + b
```

or:

```txt
sum = ADD a b
```

### Similarity and Distance

```txt
sim  = SIMILARITY u, v
corr = CORRELATE u, v
dist = DISTANCE u, v
norm = NORMALIZE v
```

### Dataset Enrichment

Mix tensors and datasets seamlessly:

```txt
sim_scores = SIMILARITY_ROWS features WITH vector_w

DATASET enriched FROM users
ADD COLUMN similarity FROM sim_scores
FILTER similarity > 0.8
```

---

## Strict vs Relaxed Execution

### Strict Mode

```txt
TENSOR a STRICT [3] = [1, 2, 3]
TENSOR b [5] = [1, 2, 3, 4, 5]

a + b   # ERROR (shape mismatch)
```

### Relaxed Mode

```txt
[1, 1, 1] + [1, 1, 1, 1, 1]
→ [2, 2, 2, 1, 1]
```

Shorter tensor is padded implicitly.

---

## Indexing

### Matrix

```txt
row = m[0, *]   # First row
col = m[*, 1]   # Second column
elem = m[2, 3]  # Single element
```

### Tuple

```txt
user.age
user.score
```

### Dataset

```txt
users.age
users.embedding
```

---

## Introspection

```txt
SHOW TENSOR v
SHOW DATASET users
SHOW ALL
```

**Planned:**

```txt
SHOW SHAPE tensor
SHOW SCHEMA dataset
SHOW TYPES tuple
```

---

## Complete Examples

### Example 1: Vector Similarity Search

```txt
# Define product embeddings
DATASET products COLUMNS (id, name, category, embedding: VECTOR[128]) = [
  (1, "Laptop", "Electronics", [0.1, 0.2, ...]),
  (2, "Mouse", "Electronics", [0.15, 0.22, ...]),
  (3, "Desk", "Furniture", [0.8, 0.1, ...])
]

# Query vector
VECTOR query = [0.12, 0.21, ...]

# Find similar products (logical order: FROM → COMPUTE → FILTER → ORDER → LIMIT)
DATASET similar FROM products
ADD COLUMN sim = SIMILARITY embedding, query
FILTER sim > 0.7
ORDER BY sim DESC
LIMIT 5
SELECT id, name, sim

SHOW similar
```

### Example 2: Matrix Operations

```txt
# Define matrices
MATRIX A = [
  [1, 2, 3],
  [4, 5, 6]
]

MATRIX B = [
  [7, 8],
  [9, 10],
  [11, 12]
]

# Matrix multiplication (A: 2x3, B: 3x2 → C: 2x2)
MATRIX C = MATMUL A B

# Transpose
MATRIX A_T = TRANSPOSE A

# Extract rows/columns
VECTOR row1 = A[0, *]
VECTOR col2 = A[*, 1]

SHOW C
SHOW A_T
```

### Example 3: Dataset Analytics

```txt
# Sales dataset
DATASET sales COLUMNS (product_id, category, region, amount, date)

# Aggregate by category (logical order)
DATASET category_stats FROM sales
GROUP BY category
COMPUTE 
  total = SUM amount,
  avg = AVG amount,
  count = COUNT *
ORDER BY total DESC

# Filter high-value categories
DATASET top_categories FROM category_stats
FILTER total > 10000
SELECT category, total, avg

SHOW top_categories
```

---

## Compilation Model

TensorDB DSL compiles into:

1. **Parsed AST**
2. **Typed Intermediate Representation**
3. **Optimized execution plan**
4. **Runtime tensor engine**

---

## Philosophy

**SQL explains structure.**  
**TensorDB explains intention.**
