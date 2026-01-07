# LINALDB — Scientific Dataset Ingestion & Delivery Roadmap

## Objective

Enable LINALDB to **consume real-world scientific tensor datasets**, normalize them
into the LINAL Core Tensor Model, and **deliver reproducible, portable dataset
packages** suitable for analytical and scientific workflows.

This roadmap defines **how LINAL ingests complex formats** (HDF5, tensors,
chunked stores) and **how users interact with them** through a clear DSL.

---

## Summary

Scientific and ML datasets are commonly distributed as:

- HDF5-based containers
- Tensor snapshots
- Chunked, cloud-native stores

These formats encode **matrices, tensors, and multi-axis metadata** that cannot be
faithfully represented as flat tables.

LINALDB will:

- ingest these formats via **connectors**
- normalize them into a **stable tensor graph**
- optionally persist them as **LINALDB dataset packages**
- expose them through a **format-agnostic DSL**

LINALDB will **not**:

- act as a format-specific runtime
- query source files live
- embed domain-specific semantics

---

## Supported Scientific Input Scope (Initial)

- HDF5-based (generic)
  - `.h5ad`
  - `.h5`
  - `.nc`
- Tensor snapshots
  - `.npz`
  - `.pt`
- Chunked / cloud-native
  - Zarr

This scope reflects the **most common tensor carriers** in scientific and ML
workflows.

---

## User Interaction Model (DSL)

LINALDB introduces **two explicit ingestion intents**.

### 1. Exploratory Usage (Ephemeral)

```linaldb
USE DATASET FROM "path/to/file.h5ad"
```

- Dataset exists only in the current session
- No persistence on disk
- Ideal for exploration and inspection
- No registry entry created

---

### 2. Persistent Import (Durable)

```linaldb
IMPORT DATASET FROM "path/to/file.h5ad"
```

- Dataset is normalized and persisted
- Dataset package is created on disk
- Metadata, lineage, and statistics stored
- Dataset is registered and reusable

---

### Design Rules

- DSL never references format-specific concepts
- All formats are treated uniformly
- Persistence is always an explicit user decision

---

## Connector Strategy

Connectors are responsible for **translation only**.

Each connector:

- understands one storage family
- extracts tensors, axes, and annotations
- preserves sparse / dense / chunked encoding
- emits lineage metadata

Connectors do **not**:

- perform transformations
- define semantics
- influence execution behavior

---

## Normalized Dataset Delivery

Persistent imports produce a **LINALDB Dataset Package**:

```text
dataset/
├── data.parquet
├── schema.json
├── stats.json
├── lineage.json
└── manifest.json
```

This package is:

- self-describing
- reproducible
- independent of the source format
- the only artifact the server layer operates on

---

## Server Layer Alignment

- Server mode operates **only on dataset packages**
- Scientific source files are never accessed directly
- Datasets are immutable once registered
- Multiple concurrent sessions may read safely

---

## Phased Implementation Plan

### Phase 1 — Direction & Core Contracts

**Goal:** Establish non-negotiable foundations.

- [ ] Define Tensor / Axis / Annotation core structures
- [ ] Define connector responsibility boundaries
- [ ] Define lineage and provenance model
- [ ] Document normalization rules

---

### Phase 2 — Connector Framework

**Goal:** Enable structured ingestion.

- [ ] Define Connector interface / trait
- [ ] Implement inspection vs read separation
- [ ] Implement generic HDF5 connector
- [ ] Add deterministic normalization tests

---

### Phase 3 — Scientific Format Support

**Goal:** Cover common scientific inputs.

- [ ] `.h5ad` connector
- [ ] `.nc` connector
- [ ] `.npz` connector
- [ ] `.pt` connector
- [ ] Zarr connector (read-only)

---

### Phase 4 — DSL Integration

**Goal:** Make ingestion explicit and safe.

- [ ] Implement `USE DATASET FROM`
- [ ] Implement `IMPORT DATASET FROM`
- [ ] Wire DSL → connector selection
- [ ] Enforce explicit persistence semantics

---

### Phase 5 — Dataset Packaging & Server Wiring

**Goal:** Deliver portable analytical assets.

- [ ] Materialize dataset packages
- [ ] Persist schema, stats, lineage, manifest
- [ ] Register datasets in server mode
- [ ] Validate reload determinism

---

## Non-Goals (Explicit)

- Live querying of source scientific formats
- Python or ML runtime integration
- Training or inference workflows
- Format-specific DSL features

---

## Outcome

At the end of this roadmap, LINAL will:

- ingest real scientific tensor datasets
- normalize them into a stable semantic core
- offer explicit ephemeral vs persistent workflows
- deliver portable, inspectable dataset packages
- remain format-agnostic and future-proof
