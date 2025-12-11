# VectorDB

**VectorDB** is an experimental, in-memory tensor database and DSL engine written in Rust. It combines high-performance tensor operations with structured dataset management and a SQL-like query language.

## 🚀 Features

-   **Tensor Operations**: Vectors, Matrices, Arithmetic, Dot Product, Similarity, Distance.
-   **Structured Datasets**: Define schemas, insert data, and query with `FILTER`, `SELECT`, `ORDER BY`.
-   **Server Mode**: Built-in HTTP server returning **TOON (Token-Oriented Object Notation)** responses.
-   **TOON Integration**: optimized output format for LLM consumption.
-   **REPL & CLI**: Interactive shell and script execution.

## 📦 Installation

Ensure you have Rust installed.

```bash
git clone https://github.com/yourusername/vector-db-rs.git
cd vector-db-rs
cargo build --release
```

## 🛠 Usage

### 1. Interactive REPL
Run the database in interactive mode:
```bash
cargo run
```

### 2. Execute a Script
Run a `.vdb` script file:
```bash
cargo run -- run example.vdb
```

### 3. Server Mode
Start the HTTP server (default port 8080):
```bash
cargo run -- server --port 8080
```

Send commands via HTTP (responds in `text/toon`):
```bash
curl -X POST -H "Content-Type: application/json" -d '{"command": "SHOW ALL"}' http://localhost:8080/execute
```

## 📘 Documentation

-   [DSL Reference](docs/DSL_REFERENCE.md) - Full guide to the Query Language.
-   [TOON Format](TOON_FORMAT.md) - Specification of the response format.
-   [New Features & Architecture](docs/NewFeatures.md) - detailed design docs.

## 🏗 Architecture

-   **`core`**: Base types (`Tensor`, `Dataset`, `Value`).
-   **`engine`**: Database logic and operations.
-   **`dsl`**: Parser and execution handlers.
-   **`server`**: Axum-based HTTP server with TOON encoding.

## 🧪 Testing

Run the test suite:
```bash
cargo test
```
