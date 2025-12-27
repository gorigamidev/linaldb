# Mini Implementation Plan: Tensor-First Datasets in LINAL

## Goal

Enable **datasets as first-class runtime objects** that:

- Do not own data
- **Reference existing tensors** in the TensorStore
- Can be **incrementally built** from loose tensors or loaded data
- Preserve **performance, lazy evaluation, and SIMD execution**
- Align fully with the existing ARCHITECTURE.md

> Core principle:  
> **Tensors remain primary. Datasets are structured views over shared tensor memory.**

---

## Non-Goals (Important)

- ❌ No rewrite of TensorStore
- ❌ No breaking changes to the DSL semantics
- ❌ No forced “everything is a dataset”
- ❌ No eager materialization
- ❌ No new execution engine

---

## Phase 1 — Formalize Dataset as a Runtime Concept (No Behavior Change)

### 1. Introduce a Dataset Core Abstraction

Add a new core module:

```

src/core/dataset/
├── dataset.rs
├── schema.rs
└── registry.rs

````

**Dataset responsibilities:**

- Hold a `Schema`
- Hold references (`TensorId`) to existing tensors
- Never own or copy tensor data

```text
Dataset
 ├── name
 ├── schema
 └── columns: { column_name -> TensorId }
````

> At this stage, datasets are inert containers (no execution logic).

---

### 2. Add a Dataset Registry to the Runtime

- Similar to how tensors are tracked
- Datasets live in the same runtime scope as DSL variables
- DatasetRegistry only stores metadata + tensor references

This enables:

- Stable dataset identity
- Symbol resolution from the DSL
- Future dataset reuse

---

## Phase 2 — Minimal DSL Integration (Zero Breaking Changes)

### 3. Add a Dataset Constructor Expression

Introduce **one new DSL entry point**:

```lnl
let ds = dataset("my_dataset")
```

Runtime behavior:

- Creates an empty Dataset
- Registers it in the DatasetRegistry
- Does not touch TensorStore

---

### 4. Allow Adding Existing Tensors as Columns

Add **one method-style operation**:

```lnl
ds.add_column("embedding", emb)
```

Runtime resolution:

1. `emb` resolves to an existing `TensorId`
2. Dataset stores the `TensorId`
3. Schema is updated
4. No data is copied

> This works for:
>
> - vectors
> - matrices
> - higher-order tensors

---

## Phase 3 — Symbol Resolution & Tensor Reuse

### 5. Enable Column Access via Dot Notation

Allow:

```lnl
ds.embedding.mean()
```

Resolution path:

1. `ds.embedding` resolves to `TensorId`
2. Planner receives a normal tensor operation
3. Backend executes unchanged

> From the execution engine’s perspective, **nothing new happened**.

---

### 6. Allow Reverse Integration (Dataset → Tensor → Dataset)

Support patterns like:

```lnl
let norms = ds.embedding.norm()
ds.add_column("norm", norms)
```

This ensures:

- Free experimentation
- No rigid pipeline constraints
- Natural scientific workflows

---

## Phase 4 — Align Parquet with the Architecture (No Runtime Coupling)

### 7. Reposition Parquet as a Storage Backend

Change the mental (and code) model:

- ❌ Dataset == Parquet
- ✅ Dataset → (optional) Parquet materialization

Parquet writing flow:

```text
Dataset
 → resolve TensorIds
 → read from TensorStore
 → materialize once
 → write parquet
```

Dataset remains alive after export.

---

## Phase 5 — Robustness & Safety (Incremental)

### 8. Add Schema Validation (Non-Blocking)

- Validate tensor shapes when adding columns
- Enforce column compatibility (lengths, partitions)
- Fail early, but never auto-copy or coerce

---

### 9. Dataset Integrity Checks

- Prevent dangling TensorIds
- Detect incompatible column lengths
- Allow explicit user overrides later (future work)

---

## Phase 6 — Testing Strategy (No New Infra)

Add focused tests:

- Dataset referencing existing tensors
- Dataset round-trip with parquet
- Mixed workflows (loose tensors ↔ datasets)
- Zero-copy guarantees (assert same TensorId)

---

## Final Outcome

After this plan:

- Loose tensors remain first-class citizens
- Datasets become composable, lightweight, and natural
- Math and data share the same execution core
- Performance characteristics are preserved
- ARCHITECTURE.md is fully honored in code

> LINAL becomes a **tensor-native analytical engine** where
> **data engineering and scientific computing converge naturally**.

---

## Guiding Rule (For All Changes)

> If a change introduces copying, eager materialization, or duplicated storage — it is the wrong change.
