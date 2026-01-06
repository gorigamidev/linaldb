# LINAL Roadmap

> LINAL is a tensor-first analytical engine that unifies data engineering and scientific workflows through semantic transformations.

This roadmap is structured by **layers**, not features.
Each phase builds on top of the existing codebase and preserves backward compatibility unless explicitly stated.

---

## 🧱 Phase 0 — Core Stabilization (FOUNDATION)

**Goal:** Freeze and solidify the semantic and execution core.

### Core Semantics

- [x] Freeze tensor identity model (no breaking changes without RFC)
- [x] Explicitly document immutable core concepts
- [x] Clarify what is considered "semantic core" vs "extensions"
- [x] Add semantic invariants (documented guarantees)

### DSL

- [x] Freeze DSL grammar and keywords (current syntax)
- [x] Validate DSL → AST mapping with golden tests
- [x] Ensure all existing DSL scripts remain valid
- [x] Mark deprecated syntax explicitly (if any)

### Execution Engine

- [x] Stabilize execution pipeline (parse → plan → execute)
- [x] Add execution-level tests (semantic, not only unit)
- [x] Ensure deterministic execution for identical inputs
- [x] Validate memory safety and ownership boundaries

### Documentation

- [x] Mark obsolete planning documents as archived
- [x] Update README to reflect current (not future) capabilities
- [x] Define "Core vs Non-Core" in ARCHITECTURE.md

---

## 🟢 Phase 1 — Embedded / SQLite-like Mode (LOCAL FIRST) [DONE]

**Goal:** Make LINAL usable as a local analytical engine.

### CLI

- [x] Minimal CLI entrypoint (`linal`)
- [x] Run DSL scripts from file
- [x] Execute inline DSL commands (`linal exec`)
- [x] Interactive REPL (basic polish implemented)

### Sessions

- [x] Define session lifecycle (RESET SESSION)
- [x] In-memory datasets scoped to session
- [x] Explicit session reset semantics
- [x] Ensure no global mutable state leaks (via reset_session)

### Basic I/O

- [x] Import CSV → dataset (with inference)
- [x] Export dataset → CSV
- [x] Define schema inference rules (robust Arrow-based inference)
- [x] Handle large files (via Arrow batching)

### Retrocompatibility

- [x] Ensure existing code paths still work without CLI
- [x] No changes required to existing DSL scripts

---

## 🟡 Phase 2 — Dataset Persistence & Lifecycle [DONE]

**Goal:** Turn datasets into durable, reusable analytical assets.

### Dataset Registry

- [x] Introduce dataset metadata registry
- [x] Unique dataset identity (name + version or hash)
- [x] Track dataset origin (bind / derive / attach)
- [x] Persist metadata separately from data (`.metadata.json`)
- [x] Implement Metadata Management DSL (`SET`, `SHOW`, `LIST`)

### Storage

- [x] Parquet as first-class storage format
- [x] Save dataset snapshots to disk
- [x] Reload datasets into new sessions
- [x] Support schema evolution (non-breaking foundation implemented)

### Lineage

- [x] Persist transformation lineage (integrated into Parquet)
- [x] Make lineage queryable (SHOW LINEAGE)
- [x] Ensure lineage survives restarts
- [x] Validate lineage consistency on reload

### Retrocompatibility

- [x] In-memory-only mode still supported
- [x] Existing workflows remain valid without persistence

---

## 🟠 Phase 3 — Server Mode & Parallel Execution

**Goal:** Enable long-running, concurrent analytical workloads.

### Server Mode

- [ ] Optional daemon/server mode
- [ ] Explicit separation: Embedded vs Server
- [ ] Server-managed sessions
- [ ] Graceful startup / shutdown

### Execution

- [ ] Job abstraction (submit / run / query status)
- [ ] Parallel execution using Rust concurrency primitives
- [ ] Resource isolation per job
- [ ] Deterministic results regardless of execution order

### Shared State

- [ ] Shared dataset registry across sessions
- [ ] Safe concurrent reads
- [ ] Explicit write semantics (no implicit mutation)

### Retrocompatibility

- [ ] Embedded mode remains default
- [ ] No server requirement for local usage

---

## 🔵 Phase 4 — Advanced Tensor & Analytical Capabilities (OPTIONAL)

**Goal:** Extend mathematical expressiveness without bloating the core.

### Tensor Operations

- [ ] Higher-order tensor support
- [ ] Optimized linear algebra primitives
- [ ] Explicit shape and dimension validation
- [ ] Lazy vs eager evaluation strategies

### Analytical Extensions

- [ ] Feature engineering primitives
- [ ] Statistical transformations
- [ ] Optional integration points (not dependencies)

### Guardrails

- [ ] No math feature enters core without clear use-case
- [ ] All advanced features live behind extension boundaries

---

## 🔴 Explicit Non-Goals (FOR NOW)

- [ ] Distributed cluster execution
- [ ] Full SQL compatibility
- [ ] ML model training framework
- [ ] Visualization/UI tools
- [ ] Automatic cloud integration

These may be revisited only with real user demand.

---

## 📌 Guiding Principles

- Core first, layers later
- Retrocompatibility by default
- No feature without semantic justification
- One mental model: tensors with identity
- SQLite-level simplicity at the edges, not in the core

---
