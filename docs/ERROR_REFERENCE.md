# LINAL Error Reference

This document provides detailed information about the errors you might encounter while using the LINAL engine and how to resolve them.

---

## 1. Engine Errors (`EngineError`)

Engine errors occur during the internal execution of algebraic or data operations.

| Error | Description | Resolution |
|-------|-------------|------------|
| `ShapeMismatch` | Attempted an operation on tensors with incompatible shapes. | Verify dimensions (e.g., Matrix A: 2x3, Matrix B: 3x5 for MATMUL). |
| `ResourceLimitExceeded` | The query exceeded the 100MB memory limit or other resource constraints. | Reduce tensor size or break complex queries into smaller steps. |
| `DatasetNotFound` | Referred to a dataset that does not exist in the active database. | Check your spelling or run `SHOW ALL DATASETS`. |
| `TensorNotFound` | Referred to a tensor ID or variable that is not in the store. | Verify the variable name or check if the tensor was deleted. |
| `DatabaseAlreadyExists` | Attempted to create a database with a name that is already taken. | Use a different name or drop the existing database first. |

---

## 2. DSL Errors (`DslError`)

DSL errors occur during the parsing or initial routing of your script commands.

### Syntax Errors

Happens when the command doesn't match LINAL's expected grammar.

- **Example**: `GET * FROM users` (Should be `SELECT`)
- **Fix**: Refer to [DSL_REFERENCE.md](DSL_REFERENCE.md) for correct syntax.

### Unbalanced Parentheses

Triggered when a multi-line command (like `DATASET` or `INSERT`) is not closed correctly.

- **Fix**: Ensure every `(` has a corresponding `)`.

---

## 3. Storage Errors (`StoreError`)

Errors related to Parquet/JSON persistence or disk access.

- **`SerializationError`**: Failed to convert data to disk format. Often happens with unsupported nested types.
- **`IOError`**: Permissions issue or disk full when saving to `./data`.

---

## 4. Common Troubleshooting

### "My command does nothing"

LINAL script requires a NEWLINE or semicolon-equivalent completion. If you are in the REPL and see no output, verify your parentheses are balanced.

### "Dangling Reference" Warning

If `SHOW <dataset>` displays a warning, it means one of your columns points to a `TensorId` that was manually removed from the store.

- **Fix**: Re-attach the data using `ATTACH <tensor> TO <dataset>.<column>`.

### "Backend Fallback"

LINAL automatically falls back to scalar execution if SIMD is not supported or the tensor layout is too complex. This is transparent but might be slower for massive datasets.
