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

## 🟢 Phase 1 — Embedded / SQLite-like Mode (LOCAL FIRST)

**Goal:** Make LINAL usable as a local analytical engine.

### CLI
- [ ] Minimal CLI entrypoint (`linal`)
- [ ] Run DSL scripts from file
- [ ] Execute inline DSL commands
- [ ] Interactive REPL (basic, no UX polish required)

### Sessions
- [ ] Define session lifecycle (start / execute / end)
- [ ] In-memory datasets scoped to session
- [ ] Explicit session reset semantics
- [ ] Ensure no global mutable state leaks

### Basic I/O
- [ ] Import CSV → tensor/dataset
- [ ] Export tensor/dataset → CSV
- [ ] Define schema inference rules
- [ ] Handle large files incrementally (streaming where possible)

### Retrocompatibility
- [ ] Ensure existing code paths still work without CLI
- [ ] No changes required to existing DSL scripts

---

## 🟡 Phase 2 — Dataset Persistence & Lifecycle

**Goal:** Turn datasets into durable, reusable analytical assets.

### Dataset Registry
- [ ] Introduce dataset metadata registry
- [ ] Unique dataset identity (name + version or hash)
- [ ] Track dataset origin (bind / derive / attach)
- [ ] Persist metadata separately from data

### Storage
- [ ] Parquet as first-class storage format
- [ ] Save dataset snapshots to disk
- [ ] Reload datasets into new sessions
- [ ] Support schema evolution (non-breaking only)

### Lineage
- [ ] Persist transformation lineage
- [ ] Make lineage queryable
- [ ] Ensure lineage survives restarts
- [ ] Validate lineage consistency on reload

### Retrocompatibility
- [ ] In-memory-only mode still supported
- [ ] Existing workflows remain valid without persistence

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
