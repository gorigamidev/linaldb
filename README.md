# vector-db-rs

`vector-db-rs` is an experimental **in-memory tensor database and DSL engine** written in Rust.

The goal of the project is to explore how a **small, high-level “pseudocode in English” language** can sit on top of a tensor store and provide building blocks for **vector/tensor operations, ML-style feature engineering and experimentation**.

> This is a learning / playground project, not a production database (yet 😉).

---

## Features (current)

- ✅ **In-memory tensor store**
  - `Shape` with arbitrary dimensions `[d1, d2, ...]`
  - `Tensor` (dense, `f32`, row-major)
  - `TensorDb` engine that manages named tensors with modes (`NORMAL` / `STRICT`)

- ✅ **Two tensor modes**
  - `TENSOR` → Normal (relaxed semantics)
  - `STRICT TENSOR` → Strict (shapes must match)

- ✅ **High-level DSL (text files with `.tdb` extension)**
  - Scripts that define tensors and execute operations:

    ```txt
    DEFINE a AS TENSOR [3] VALUES [1, 0, 0]
    DEFINE b AS TENSOR [3] VALUES [0, 1, 0]
    DEFINE c AS TENSOR [3] VALUES [1, 1, 0]

    LET sum_ab  = ADD a b
    LET corr_ac = CORRELATE a WITH c
    LET sim_ac  = SIMILARITY a WITH c
    LET dist_ac = DISTANCE a TO c
    LET a_half  = SCALE a BY 0.5
    LET c_norm  = NORMALIZE c

    SHOW ALL
    ```

- ✅ **CLI**
  - Run a script from a file:
    ```bash
    cargo run -- example.tdb
    ```
  - Or use an interactive REPL:
    ```bash
    cargo run
    ```

- ✅ **Core operations**
  - Element-wise:
    - `ADD a b`
    - `SUBTRACT a b`
    - `MULTIPLY a b`
    - `DIVIDE a b` (with division-by-zero detection)
  - Unary / scalar:
    - `SCALE x BY <number>`
    - `NORMALIZE x` (L2 norm = 1 for rank-1 tensors)
  - Vector / ML-style:
    - `CORRELATE a WITH b` → dot product (rank-1)
    - `SIMILARITY a WITH b` → cosine similarity (rank-1)
    - `DISTANCE a TO b` → L2 distance (rank-1)

- ✅ **Tests**
  - Unit tests for core numeric ops.
  - Integration tests for the DSL and engine behavior in `tests/`.

---

## DSL syntax (V1)

The DSL allows you to define tensors, apply operations, and inspect results using a readable, English-like syntax.

---

## Tensor declaration

You can define tensors in two modes:

### Normal tensor (relaxed)

```txt
DEFINE a AS TENSOR [3] VALUES [1, 2, 3]
````

### Strict tensor

```txt
DEFINE a AS STRICT TENSOR [3] VALUES [1, 2, 3]
```

### Meaning

* `TENSOR`: relaxed behavior for element-wise operations.
* `STRICT TENSOR`: enforces shape equality for element-wise operations.

If **either** operand is `STRICT`, operations run in strict mode.

---

## Supported Operations

All operations follow this pattern:

```txt
LET <result_name> = OPERATION ...
```

---

## Arithmetic (element-wise)

| Operation | DSL                    |
| --------- | ---------------------- |
| Add       | `LET x = ADD a b`      |
| Subtract  | `LET x = SUBTRACT a b` |
| Multiply  | `LET x = MULTIPLY a b` |
| Divide    | `LET x = DIVIDE a b`   |

---

### Shape rules (element-wise)

When using `ADD`, `SUBTRACT`, `MULTIPLY`, `DIVIDE`:

### STRICT + anything

If **any tensor is STRICT**:

* Shapes must match exactly.
* Otherwise, the operation fails with:

```
Shape mismatch: [a, b, ...] vs [x, y, ...]
```

---

### NORMAL + NORMAL (relaxed mode)

If **both tensors are normal (`TENSOR`)**:

1. **Scalar broadcasting**

If one tensor is scalar (`[]`), it is broadcast to the shape of the other tensor.

2. **Rank-1 vectors with different lengths**

If both tensors are vectors but have different lengths:

* The operation is applied index-by-index.
* Missing positions use a neutral value:

| Operation | Neutral |
| --------- | ------- |
| ADD / SUB | `0.0`   |
| MUL / DIV | `1.0`   |

Example:

```txt
DEFINE a AS TENSOR [3] VALUES [1, 2, 3]
DEFINE b AS TENSOR [5] VALUES [10, 20, 30, 40, 50]

LET c = ADD a b
SHOW c
```

Result:

```
[11, 22, 33, 40, 50]
```

---

## Unary / scalar operations

| Operation      | DSL                      |
| -------------- | ------------------------ |
| Scale          | `LET x = SCALE a BY 0.5` |
| Normalize (L2) | `LET x = NORMALIZE a`    |

> `NORMALIZE` works only for rank-1 tensors.

---

## Vector relations (rank-1, always strict)

These operations always require equal shape:

| Operation         | DSL                           |
| ----------------- | ----------------------------- |
| Dot product       | `LET x = CORRELATE a WITH b`  |
| Cosine similarity | `LET x = SIMILARITY a WITH b` |
| L2 distance       | `LET x = DISTANCE a TO b`     |

---

## Output commands

| Command          | DSL        |
| ---------------- | ---------- |
| Show one tensor  | `SHOW a`   |
| Show all tensors | `SHOW ALL` |

---

## Example

```txt
DEFINE a AS STRICT TENSOR [3] VALUES [1, 2, 3]
DEFINE b AS TENSOR [5] VALUES [10, 20, 30, 40, 50]

LET fail = ADD a b      # strict operation -> error
LET ok   = ADD b b      # relaxed mode

DEFINE u AS TENSOR [3] VALUES [1, 0, 0]
DEFINE w AS TENSOR [3] VALUES [1, 1, 0]

LET corr = CORRELATE u WITH w
LET sim  = SIMILARITY u WITH w
LET dist = DISTANCE u TO w
LET w_n  = NORMALIZE w

SHOW ok
SHOW corr
SHOW sim
SHOW dist
SHOW w_n
SHOW ALL
```

---

## Development

### Requirements

* Rust (stable)
* Cargo

### Run tests

```bash
cargo test
```

---

## Roadmap ideas

* `RESHAPE`, `FLATTEN`, `TRANSPOSE`, `MATMUL`
* Tensor persistence:

  ```txt
  SAVE x TO "file.bin"
  LOAD y FROM "file.bin"
  ```
* Pipelines (composable DSL blocks)
* VS Code extension for `.tdb` (syntax + icon)
* Pluggable tensor backends (future)

---

## Disclaimer

This is not a production database.

It is a playground to explore:

* Rust for data systems
* DSL design
* Tensor semantics
* Vector similarity engines