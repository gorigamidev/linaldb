# LINAL – Formalization Roadmap (Phased Plan)

## Phase 0 – Baseline Stabilization (Checkpoint: "Nothing breaks")

### Phase 0 Objective

Ensure the current engine, DSL, and storage model are stable before formalization.

### Phase 0 Checkpoints

- [x] Review all exisiting dependencies (Cargo.toml) and remove any that are not needed
- [x] All existing tests pass
- [x] Examples still execute unchanged
- [x] Dataset → Tensor separation remains intact
- [x] Benchmarks show no regression

### Phase 0 Exit Criteria

- Zero behavior changes
- No public API breakage
- Clean baseline for further phases

---

## Phase 1 – Tensor Identity (Checkpoint: "Tensors are addressable")

### Phase 1 Objective

Introduce a formal and stable identity for tensors.

### Phase 1 Tasks

- [x] Define `TensorId` (UUID / hash-based)
- [x] Define `TensorMetadata` struct (Provenance).
- [x] Replace `u64` IDs with globally unique `Uuid`.
- [x] Implement SHA-256 content-based hashing for Tensors.
- [x] **Lazy Hashing Implementation**: Defer SHA-256 computation until needed (Recovered >99% performance regression).
- [x] Explicitly disable mutable access to Tensor data (Enforce Immutability).
- [x] Update `src/core/storage.rs` to persist/load new metadata.

### Phase 1 Checkpoints

- [x] Every tensor has a stable identity
- [x] Engine can resolve a tensor by ID
- [x] No tensor is embedded in a dataset

### Phase 1 Exit Criteria

- Tensors are first-class, identifiable entities
- No changes required in existing DSL expressions

---

## Phase 2 – Dataset as Reference Graph (Checkpoint: "Datasets are views")

### Phase 2 Objective

Formalize datasets as explicit reference graphs over tensors.

### Phase 2 Tasks

- [x] Define `DatasetReference`
- [x] Define `DatasetGraph`
- [x] Update dataset metadata to declare tensor roles
- [x] Support multiple datasets pointing to the same tensor
- [x] Keep datasets lightweight and rebuildable

### Phase 2 Checkpoints

- [x] Dataset metadata expresses references explicitly
- [x] Engine resolves datasets via tensor references
- [x] No tensor duplication occurs

### Phase 2 Exit Criteria

- Dataset = semantic view, not storage
- Reference graph is explicit and inspectable

---

## Phase 3 – Execution Context & Lineage (Checkpoint: "Everything is traceable")

### Phase 3 Objective

Track how tensors and datasets are derived.

### Phase 3 Tasks

- [ ] Introduce `ExecutionContext`
- [ ] Assign execution IDs
- [ ] Capture inputs, operations, and outputs
- [ ] Attach lineage metadata to derived tensors
- [ ] Enable inspection of derivation history

### Phase 3 Checkpoints

- [ ] Derived tensors know their origin
- [ ] Engine can emit lineage info
- [ ] No orchestration or scheduling introduced

### Phase 3 Exit Criteria

- Minimal but reliable traceability
- Reproducible execution paths

---

## Phase 4 – DSL Semantic Expansion (Checkpoint: "Intent is explicit")

### Phase 4 Objective

Make the DSL declarative without increasing complexity.

### Phase 4 Tasks

- [ ] Add `use` semantic
- [ ] Add `bind` semantic
- [ ] Add `attach` semantic
- [ ] Add `derive` semantic
- [ ] Map semantics into logical plan

### Phase 4 Checkpoints

- [ ] Existing DSL remains valid
- [ ] New semantics are optional
- [ ] Logical planner understands relationships

### Phase 4 Exit Criteria

- DSL expresses intent and relationships
- No new mathematical operations added

---

## Phase 5 – Internal Consistency & Validation (Checkpoint: "The model is solid")

### Phase 5 Objective

Validate that all formalized concepts work together coherently.

### Phase 5 Tasks

- [ ] Cross-validate tensor identity with dataset graphs
- [ ] Validate lineage consistency
- [ ] Add introspection APIs
- [ ] Improve error messages and diagnostics

### Phase 5 Checkpoints

- [ ] Engine can explain what happened and why
- [ ] Clear errors for invalid references
- [ ] Strong internal invariants

### Phase 5 Exit Criteria

- System is internally coherent
- Users can reason about their data and math

---

## Phase 6 – Usability Hardening (Checkpoint: "People can actually use it")

### Phase 6 Objective

Prepare Linal for real users and external integrations.

### Phase 6 Tasks

- [ ] Improve CLI feedback (optional)
- [ ] Add inspection commands (list tensors, datasets, lineage)
- [ ] Document mental model clearly
- [ ] Add end-to-end examples

### Phase 6 Checkpoints

- [ ] New users understand core concepts
- [ ] Scientists can explore results
- [ ] Data engineers can integrate pipelines

### Phase 6 Exit Criteria

- Linal is usable, not just powerful

---

## Phase 7 – Ecosystem Readiness (Checkpoint: "Open but controlled")

### Phase 7 Objective

Prepare for future extensions without implementing them.

### Phase 7 Tasks

- [ ] Define extension points
- [ ] Define external tensor consumers (e.g. vector index)
- [ ] Keep ownership rules strict
- [ ] Document integration contracts

### Phase 7 Checkpoints

- [ ] External tools can plug in conceptually
- [ ] Core remains small and stable
- [ ] No premature feature creep

### Phase 7 Exit Criteria

- Linal is extensible by design
- Future work is unblocked

---

## Final Outcome

At the end of these phases, Linal becomes:

- A mathematically rigorous engine
- A usable data + tensor system
- A traceable and reproducible platform
- A solid foundation for ML, RAG, and scientific workflows

Without becoming a framework or a monolith.

---
