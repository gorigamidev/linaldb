# VectorDB

VectorDB is an experimental, in-memory tensor database and DSL engine written in Rust. It supports high-performance tensor operations and structured dataset management with a SQL-like query language.

## Features

- **Tensor Operations**: 
  - Create vectors and matrices.
  - Perform arithmetic (`ADD`, `SUB`, `MUL`, `DIV`).
  - Compute `SIMILARITY`, `DISTANCE`, `CORRELATE`.
  - Linear algebra: `MATMUL`, `TRANSPOSE`, `RESHAPE`, `FLATTEN`.
- **Structured Datasets**:
  - Define schemas with types (`INT`, `FLOAT`, `STRING`, `BOOL`).
  - Insert row data.
- **Query Language**:
  - `FROM ... FILTER ... SELECT ... ORDER BY ... LIMIT` syntax for datasets.
- **Modular Architecture**:
  - Clean separation between Core, Engine, and DSL layers.

## Installation

Ensure you have Rust installed. Clone the repository and build:

```bash
git clone https://github.com/yourusername/vector-db-rs.git
cd vector-db-rs
cargo build --release
```

## Usage

You can run scripts using the DSL.

### Running a script

Create a `.tdb` file (e.g., `example.tdb`):

```sql
DEFINE v1 AS TENSOR [3] VALUES [1.0, 2.0, 3.0]
DATASET users COLUMNS (name: STRING, age: INT)
INSERT INTO users VALUES ("Alice", 30)
SHOW ALL
```

Run it:

```bash
cargo run example.tdb
```

You can also run in interactive mode (REPL) by running `cargo run` without arguments.

## DSL Reference

### Tensors

```sql
DEFINE <name> AS TENSOR [<dims>] VALUES [<values>]
VECTOR <name> = [<values>]
MATRIX <name> = [[<row1>], [<row2>]]

LET <name> = ADD <t1> <t2>
LET <name> = MATMUL <m1> <m2>
LET <name> = TRANSPOSE <m>
LET <name> = SIMILARITY <t1> WITH <t2>
SHOW <name>
SHOW SHAPE <name>
```

### Datasets

**Creation:**
```sql
DATASET <name> COLUMNS (<col>: <TYPE>, ...)
```
Types: `INT`, `FLOAT`, `STRING`, `BOOL`.

**Insertion:**
```sql
INSERT INTO <name> VALUES (<val1>, <val2>, ...)
```

**Querying:**
```sql
DATASET <new_name> FROM <source> 
    [FILTER <col> <op> <val>]
    [SELECT <col1>, <col2>]
    [ORDER BY <col> [DESC]]
    [LIMIT <n>]
```

## Architecture

The project is organized into four main modules:

- **`core`**: Fundamental data types (Tensor, Dataset, Tuple, Value) and storage.
- **`engine`**: Execution logic (TensorDb, Operations).
- **`dsl`**: Parser and command handlers.
- **`utils`**: Shared utilities.

## Testing

Run the full test suite:

```bash
cargo test
```

## License

MIT