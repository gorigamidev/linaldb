# Phase 1 — Dataset Delivery & Server Exposure (Foundational)

## 1. Dataset Package Structure

* [x] Standardize dataset folder layout:

  ```text
  dataset/
  ├── data.parquet
  ├── schema.json        # logical + physical schema (replaces *.metadata.json)
  ├── stats.json         # dataset statistics (replaces *.meta.json)
  ├── lineage.json       # derivation & dependency graph
  ├── manifest.json      # delivery contract & entrypoint
  ```
  
* [x] Ensure **all files are self-contained** and portable (no hidden state)

---

## 2. Parquet as the Persistence & Exchange Core

* [x] Treat `data.parquet` as the **authoritative data layer**
* [x] Support:

  * vectors (1D)
  * matrices (2D)
  * tensors (N-D via nested / repeated columns or encoded blocks)
* [x] Encode tensor layout explicitly in `schema.json` (shape, rank, order)

---

## 3. Schema & Stats Normalization

* [x] `schema.json`

  * logical types (vector, matrix, tensor)
  * physical mapping to Parquet columns
  * semantic roles (feature, embedding, label, index)
* [x] `stats.json`

  * row counts
  * tensor shapes
  * min/max, sparsity, nullability
  * optional distribution summaries

---

## 4. Lineage (NEW)

* [x] Introduce `lineage.json`
* [x] Capture:

  * source datasets (IDs + hashes)
  * transformations applied (DSL expressions)
  * execution context (engine, version)
* [x] Represent lineage as a **DAG**, not prose
* [x] No execution here — **pure description**

---

## 5. Manifest (NEW)

* [x] Introduce `manifest.json` as the **delivery contract**
* [x] Include:

  * dataset name, version, hash
  * entrypoints (default view, tensor index)
  * supported export formats (json, toon, parquet)
  * compatibility info
* [x] Manifest is the **only file clients must read first**

---

## 6. Delivery DSL (Export Layer)

* [x] Define a **minimal declarative DSL** for delivery

  * no math
  * no execution
* [x] Example concepts:

  * `deliver`
  * `select`
  * `shape`
  * `project`
* [x] DSL compiles to:

  * JSON views
  * TOON graph representations
  * filtered Parquet subsets
* [x] DSL output **never mutates the dataset**

---

## 7. Server Layer (NEW)

* [x] Add a lightweight **Dataset Server**
* [x] Responsibilities:

  * serve `manifest.json` as entrypoint
  * expose schema, stats, lineage via HTTP
  * stream Parquet or deliver DSL projections
* [x] No business logic:

  * no recomputation
  * no mutation
* [x] Server acts as a **read-only delivery gateway**

---

## 8. Compatibility & Guardrails

* [x] Backward compatible with existing metadata files
* [x] Deterministic hashes across all artifacts
* [x] Offline-first: dataset usable without server
* [x] Server is optional, not required

---

## Phase Outcome

✔ Linal delivers **datasets, not files**
✔ Parquet is sufficient for complexity
✔ JSON/TOON are *views*, not storage
✔ Lineage + manifest make sharing safe and explicit
✔ Server layer enables controlled distribution

---

> [!NOTE]
> Phase 1 is now **COMPLETED**. The engine successfully generates standardized packages and exposes them via the built-in Delivery Server at `/delivery`.
