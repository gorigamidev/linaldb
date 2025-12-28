# LINAL – Formalization Roadmap (Phased Plan)

## Phase 0 – Baseline Stabilization (Checkpoint: "Nothing breaks")

### Objective

Ensure the current engine, DSL, and storage model are stable before formalization.

### Checkpoints

- [ ] Review all exisiting dependencies (Cargo.toml) and remove any that are not needed
- [ ] All existing tests pass
- [ ] Examples still execute unchanged
- [ ] Dataset → Tensor separation remains intact
- [ ] Benchmarks show no regression

### Exit Criteria

- Zero behavior changes
- No public API breakage
- Clean baseline for further phases

---

## Phase 1 – Tensor Identity (Checkpoint: "Tensors are addressable")

### Objective

Introduce a formal and stable identity for tensors.

### Tasks

- [ ] Define `TensorId` (UUID / hash-based)
- [ ] Create `TensorMetadata` struct
- [ ] Persist tensor metadata alongside physical storage
- [ ] Ensure tensors are immutable by default
- [ ] Allow lookup by `TensorId`

### Checkpoints

- [ ] Every tensor has a stable identity
- [ ] Engine can resolve a tensor by ID
- [ ] No tensor is embedded in a dataset

### Exit Criteria

- Tensors are first-class, identifiable entities
- No changes required in existing DSL expressions

---

## Phase 2 – Dataset as Reference Graph (Checkpoint: "Datasets are views")

### Objective

Formalize datasets as explicit reference graphs over tensors.

### Tasks

- [ ] Define `DatasetReference`
- [ ] Define `DatasetGraph`
- [ ] Update dataset metadata to declare tensor roles
- [ ] Support multiple datasets pointing to the same tensor
- [ ] Keep datasets lightweight and rebuildable

### Checkpoints

- [ ] Dataset metadata expresses references explicitly
- [ ] Engine resolves datasets via tensor references
- [ ] No tensor duplication occurs

### Exit Criteria

- Dataset = semantic view, not storage
- Reference graph is explicit and inspectable

---

## Phase 3 – Execution Context & Lineage (Checkpoint: "Everything is traceable")

### Objective

Track how tensors and datasets are derived.

### Tasks

- [ ] Introduce `ExecutionContext`
- [ ] Assign execution IDs
- [ ] Capture inputs, operations, and outputs
- [ ] Attach lineage metadata to derived tensors
- [ ] Enable inspection of derivation history

### Checkpoints

- [ ] Derived tensors know their origin
- [ ] Engine can emit lineage info
- [ ] No orchestration or scheduling introduced

### Exit Criteria

- Minimal but reliable traceability
- Reproducible execution paths

---

## Phase 4 – DSL Semantic Expansion (Checkpoint: "Intent is explicit")

### Objective

Make the DSL declarative without increasing complexity.

### Tasks

- [ ] Add `use` semantic
- [ ] Add `bind` semantic
- [ ] Add `attach` semantic
- [ ] Add `derive` semantic
- [ ] Map semantics into logical plan

### Checkpoints

- [ ] Existing DSL remains valid
- [ ] New semantics are optional
- [ ] Logical planner understands relationships

### Exit Criteria

- DSL expresses intent and relationships
- No new mathematical operations added

---

## Phase 5 – Internal Consistency & Validation (Checkpoint: "The model is solid")

### Objective

Validate that all formalized concepts work together coherently.

### Tasks

- [ ] Cross-validate tensor identity with dataset graphs
- [ ] Validate lineage consistency
- [ ] Add introspection APIs
- [ ] Improve error messages and diagnostics

### Checkpoints

- [ ] Engine can explain what happened and why
- [ ] Clear errors for invalid references
- [ ] Strong internal invariants

### Exit Criteria

- System is internally coherent
- Users can reason about their data and math

---

## Phase 6 – Usability Hardening (Checkpoint: "People can actually use it")

### Objective

Prepare Linal for real users and external integrations.

### Tasks

- [ ] Improve CLI feedback (optional)
- [ ] Add inspection commands (list tensors, datasets, lineage)
- [ ] Document mental model clearly
- [ ] Add end-to-end examples

### Checkpoints

- [ ] New users understand core concepts
- [ ] Scientists can explore results
- [ ] Data engineers can integrate pipelines

### Exit Criteria

- Linal is usable, not just powerful

---

## Phase 7 – Ecosystem Readiness (Checkpoint: "Open but controlled")

### Objective

Prepare for future extensions without implementing them.

### Tasks

- [ ] Define extension points
- [ ] Define external tensor consumers (e.g. vector index)
- [ ] Keep ownership rules strict
- [ ] Document integration contracts

### Checkpoints

- [ ] External tools can plug in conceptually
- [ ] Core remains small and stable
- [ ] No premature feature creep

### Exit Criteria

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
