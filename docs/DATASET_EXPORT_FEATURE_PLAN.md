# Phase 1 — Dataset Delivery & Server Exposure (Foundational)

## 1. Dataset Package Structure

* [ ] Standardize dataset folder layout:

  ```text
  dataset/
  ├── data.parquet
  ├── schema.json        # logical + physical schema (replaces *.metadata.json)
  ├── stats.json         # dataset statistics (replaces *.meta.json)
  ├── lineage.json       # NEW: derivation & dependency graph
  ├── manifest.json      # NEW: delivery contract & entrypoint
  ```
  
* [ ] Ensure **all files are self-contained** and portable (no hidden state)

---

## 2. Parquet as the Persistence & Exchange Core

* [ ] Treat `data.parquet` as the **authoritative data layer**
* [ ] Support:

  * vectors (1D)
  * matrices (2D)
  * tensors (N-D via nested / repeated columns or encoded blocks)
* [ ] Encode tensor layout explicitly in `schema.json` (shape, rank, order)

---

## 3. Schema & Stats Normalization

* [ ] `schema.json`

  * logical types (vector, matrix, tensor)
  * physical mapping to Parquet columns
  * semantic roles (feature, embedding, label, index)
* [ ] `stats.json`

  * row counts
  * tensor shapes
  * min/max, sparsity, nullability
  * optional distribution summaries

---

## 4. Lineage (NEW)

* [ ] Introduce `lineage.json`
* [ ] Capture:

  * source datasets (IDs + hashes)
  * transformations applied (DSL expressions)
  * execution context (engine, version)
* [ ] Represent lineage as a **DAG**, not prose
* [ ] No execution here — **pure description**

---

## 5. Manifest (NEW)

* [ ] Introduce `manifest.json` as the **delivery contract**
* [ ] Include:

  * dataset name, version, hash
  * entrypoints (default view, tensor index)
  * supported export formats (json, toon, parquet)
  * compatibility info
* [ ] Manifest is the **only file clients must read first**

---

## 6. Delivery DSL (Export Layer)

* [ ] Define a **minimal declarative DSL** for delivery

  * no math
  * no execution
* [ ] Example concepts:

  * `deliver`
  * `select`
  * `shape`
  * `project`
* [ ] DSL compiles to:

  * JSON views
  * TOON graph representations
  * filtered Parquet subsets
* [ ] DSL output **never mutates the dataset**

---

## 7. Server Layer (NEW)

* [ ] Add a lightweight **Dataset Server**
* [ ] Responsibilities:

  * serve `manifest.json` as entrypoint
  * expose schema, stats, lineage via HTTP
  * stream Parquet or deliver DSL projections
* [ ] No business logic:

  * no recomputation
  * no mutation
* [ ] Server acts as a **read-only delivery gateway**

---

## 8. Compatibility & Guardrails

* [ ] Backward compatible with existing metadata files
* [ ] Deterministic hashes across all artifacts
* [ ] Offline-first: dataset usable without server
* [ ] Server is optional, not required

---

## Phase Outcome

✔ Linal delivers **datasets, not files**
✔ Parquet is sufficient for complexity
✔ JSON/TOON are *views*, not storage
✔ Lineage + manifest make sharing safe and explicit
✔ Server layer enables controlled distribution
