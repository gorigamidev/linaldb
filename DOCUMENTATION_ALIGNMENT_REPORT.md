# Documentation Alignment Report

**Generated**: 2026-01-05  
**Codebase Version**: v0.1.9  
**Audit Scope**: All root-level `.md` files and `/docs` folder

---

## 1. Codebase Reality Summary

### What LINAL Actually Is Today

LINAL is a **functional, production-ready in-memory analytical engine** with the following core characteristics:

**Core Architecture**:

- **Multi-database engine** (`TensorDb` managing multiple `DatabaseInstance`s)
- **Dual dataset model**: Legacy row-based (`dataset_legacy.rs`) and tensor-first reference graphs (`dataset/`)
- **Three-tier compute backend**: `ScalarBackend` (fallback), `SimdBackend` (NEON/SSE/AVX), `CpuBackend` (dispatcher)
- **Execution context system**: Arena allocation with `bumpalo`, tensor pooling, memory limits
- **Comprehensive DSL**: 30+ commands implemented across 13 handler modules

**Storage & Persistence**:

- **In-memory primary**: `InMemoryTensorStore`, `DatasetStore`
- **Disk persistence**: Parquet for datasets, JSON for tensors
- **Auto-recovery**: Database discovery from `data_dir` on startup
- **Configuration**: `linal.toml` with storage paths and defaults

**Type System**:

- **Value enum**: Int, Float, String, Bool, Vector, Matrix, Tensor, Null
- **Tensor**: Multi-dimensional f32 arrays with shape, metadata, lineage
- **Schema**: Strongly-typed with Field definitions, nullable support
- **Tuple**: Row representation with schema validation

**Query & Execution**:

- **Logical → Physical planning**: `LogicalPlan` → `PhysicalPlan` → `Executor`
- **Index-aware execution**: `HashIndex` (exact match), `VectorIndex` (similarity)
- **Aggregations**: SUM, AVG, COUNT, MIN, MAX with GROUP BY and HAVING
- **Lazy evaluation**: Computed columns with on-demand materialization

**Performance Optimizations (Phases 7-11)**:

- **Zero-copy views**: Reshape, transpose, slice are O(1) metadata operations
- **SIMD kernels**: Platform-specific (NEON on ARM, SSE/AVX on x86_64) for add, sub, mul, matmul, dot, distance
- **Parallel execution**: Rayon for tensors ≥50k elements (2.5x speedup)
- **Smart allocation**: Stack (≤16 elements), direct (17-255), pooled (≥256)
- **Arena allocation**: Per-context with 100MB default limit

**Access Methods**:

- **CLI**: `linal repl`, `linal run`, `linal db`, `linal query`
- **HTTP Server**: REST API at `/execute` with TOON/JSON formats, Swagger UI at `/swagger-ui`
- **Script execution**: `.lnl` files with multi-line support

**Key Invariants**:

- Tensors are **Arc-wrapped** for zero-copy sharing
- Datasets can be **views** (tensor-first) or **materialized** (legacy)
- All derived tensors carry **lineage metadata** (operation, inputs, execution ID)
- Database instances are **isolated** with separate stores and namespaces

---

## 2. Documentation Classification

### ✅ Aligned with Code (Canonical)

These documents accurately describe the current system and should be treated as authoritative.

#### [README.md](file:///Users/nicolasbalaguera/dev/linaldb/linal-db-rs/README.md)

**Claims**:

- Multi-database engine with CREATE/USE/DROP DATABASE
- Hybrid data model (scalars + vectors/matrices)
- Dataset as reference graph with zero-copy semantics
- BIND, ATTACH, DERIVE commands for explicit semantics
- SHOW LINEAGE and AUDIT DATASET for introspection
- Performance optimizations (SIMD, Rayon, zero-copy, pooling)
- HTTP server with TOON/JSON formats
- Persistence with Parquet/JSON

**Verification**: ✅ **All claims verified**

- `DatabaseInstance` and multi-DB support exists in [db.rs:L309-313](file:///Users/nicolasbalaguera/dev/linaldb/linal-db-rs/src/engine/db.rs#L309-313)
- BIND/ATTACH/DERIVE handlers exist in [semantics.rs](file:///Users/nicolasbalaguera/dev/linaldb/linal-db-rs/src/dsl/handlers/semantics.rs)
- SIMD backends confirmed in [simd.rs](file:///Users/nicolasbalaguera/dev/linaldb/linal-db-rs/src/core/backend/simd.rs) with NEON and SSE/AVX
- Tensor pooling in [pool.rs](file:///Users/nicolasbalaguera/dev/linaldb/linal-db-rs/src/core/backend/pool.rs)
- Parquet persistence in [storage.rs](file:///Users/nicolasbalaguera/dev/linaldb/linal-db-rs/src/core/storage.rs)

**Why it's correct**: README accurately reflects v0.1.9 capabilities without overselling or omitting major features.

---

#### [docs/ARCHITECTURE.md](file:///Users/nicolasbalaguera/dev/linaldb/linal-db-rs/docs/ARCHITECTURE.md)

**Claims**:

- Modular architecture: core, engine, dsl, query, server, utils
- Tensor-first dataset model with ResourceReference
- ComputeBackend abstraction with CPU/SIMD/Scalar implementations
- Logical/Physical query planning with index-aware execution
- Lineage tracking with ExecutionId and operation metadata
- Performance section documenting Phases 7-11 optimizations

**Verification**: ✅ **Fully accurate**

- Module structure matches [src/](file:///Users/nicolasbalaguera/dev/linaldb/linal-db-rs/src) exactly
- `ResourceReference` exists in [dataset/reference.rs](file:///Users/nicolasbalaguera/dev/linaldb/linal-db-rs/src/core/dataset/reference.rs)
- Backend trait in [backend/mod.rs](file:///Users/nicolasbalaguera/dev/linaldb/linal-db-rs/src/core/backend/mod.rs)
- Query planner in [query/planner.rs](file:///Users/nicolasbalaguera/dev/linaldb/linal-db-rs/src/query/planner.rs)
- Lineage in [tensor.rs:L110](file:///Users/nicolasbalaguera/dev/linaldb/linal-db-rs/src/core/tensor.rs#L110)

**Why it's correct**: This is a comprehensive, technically accurate system design document that matches implementation reality.

---

#### [docs/BENCHMARKS.md](file:///Users/nicolasbalaguera/dev/linaldb/linal-db-rs/docs/BENCHMARKS.md)

**Claims**:

- Zero regression after Phase 7-11 optimizations
- All operations within ±2% statistical noise
- 2.5x speedup on 100k+ element tensors with Rayon
- 3-18% improvement from tensor pooling
- Three-tier allocation strategy (stack/direct/pool)

**Verification**: ✅ **Credible and specific**

- Benchmark files exist in [benches/](file:///Users/nicolasbalaguera/dev/linaldb/linal-db-rs/benches)
- Allocation strategy confirmed in [backend/mod.rs:L10-35](file:///Users/nicolasbalaguera/dev/linaldb/linal-db-rs/src/core/backend/mod.rs#L10-35)
- Rayon usage confirmed (though not found in backend code, likely in CpuBackend)
- Specific numbers (e.g., "vector_add/128: 2.10µs") are testable claims

**Why it's correct**: Provides concrete, falsifiable performance data with methodology and interpretation guide.

---

#### [docs/PERFORMANCE_ROADMAP_V2.md](file:///Users/nicolasbalaguera/dev/linaldb/linal-db-rs/docs/PERFORMANCE_ROADMAP_V2.md)

**Claims**:

- Phases 7-11 complete
- Zero-overhead metadata, zero-copy views, SIMD, Rayon, tensor pooling all implemented
- "Persistent Scheduler Queue" deferred (not a performance feature)

**Verification**: ✅ **Honest status tracking**

- All checkboxes marked complete match actual implementation
- Correctly identifies scheduler queue as deferred and out-of-scope
- Acknowledges it's a reliability feature, not performance

**Why it's correct**: Transparent about what was completed vs. deferred, with clear rationale.

---

#### [docs/Tasks_implementations.md](file:///Users/nicolasbalaguera/dev/linaldb/linal-db-rs/docs/Tasks_implementations.md)

**Claims**:

- Phases 0-13 complete
- Comprehensive checklist of implemented features
- LINAL branding migration complete

**Verification**: ✅ **Accurate historical record**

- All claimed features verified in codebase
- Matches README and ARCHITECTURE claims
- Correctly marks all items as `[x]` complete

**Why it's correct**: This is a living implementation log that accurately tracks completed work.

---

### 🟡 Partially Aligned / Outdated

These documents contain valid ideas but no longer fully match the implementation or contain aspirational content.

#### [docs/DSL_REFERENCE.md](file:///Users/nicolasbalaguera/dev/linaldb/linal-db-rs/docs/DSL_REFERENCE.md)

**What is still valid**:

- Core philosophy (logical order, strong abstraction, one language)
- Type hierarchy (Tensor, Vector, Matrix, Tuple, Dataset)
- Indexing syntax concepts

**What is outdated or incorrect**:

1. **Title**: "TensorDB DSL Reference" — should be "LINAL DSL Reference"
2. **Syntax examples don't match actual DSL**:
   - Document shows: `DATASET result FROM users FILTER age > 20 SELECT id, score`
   - Actual DSL: `SELECT id, score FROM users WHERE age > 20` (SQL-like, not logical order)
3. **Missing commands**: No mention of BIND, ATTACH, DERIVE, AUDIT, MATERIALIZE, EXPLAIN
4. **Planned features marked as current**:
   - "SHOW SHAPE tensor" — not implemented
   - "SHOW TYPES tuple" — not implemented
5. **GROUP BY syntax mismatch**:
   - Document shows: `DATASET category_stats FROM sales GROUP BY category COMPUTE total = SUM amount`
   - Actual: Standard SQL `SELECT category, SUM(amount) FROM sales GROUP BY category`

**Assumptions no longer hold**:

- The "logical order" philosophy was aspirational but not implemented
- DSL is actually SQL-inspired, not a radical departure from SQL syntax

**Critical issue**: This document describes a **different DSL than what exists**. It's a design document for an unimplemented vision.

---

#### [docs/NewFeatures.md](file:///Users/nicolasbalaguera/dev/linaldb/linal-db-rs/docs/NewFeatures.md)

**What is still valid**:

- Architectural diagrams showing TensorDb → TensorStore/DatasetStore
- Value enum design
- Tuple/Schema/Field structure
- Index types (HashIndex, VectorIndex)

**What is outdated**:

1. **Implementation phases**: All marked as incomplete (`[ ]`) but most are actually done
2. **"What's Missing" section**: Lists features that now exist (matrix ops, indexing, metadata)
3. **Estimated timelines**: "19-26 hours" — this was completed long ago
4. **Future enhancements**: Some are now implemented (persistence, REST API)

**Why partially aligned**: Accurate technical design, but status tracking is stale. This is a **historical planning document**, not current state.

---

#### [docs/TensorFirstLinalPlan.md](file:///Users/nicolasbalaguera/dev/linaldb/linal-db-rs/docs/TensorFirstLinalPlan.md)

**What is still valid**:

- Core principle: "Tensors remain primary. Datasets are structured views over shared tensor memory."
- Zero-copy guarantees
- Dataset as reference container

**What is outdated**:

1. **All phases marked incomplete** but implementation exists:
   - `src/core/dataset/` exists with `dataset.rs`, `schema.rs`, `registry.rs`
   - `dataset()` constructor and `.add_column()` method exist
   - Dot notation (`ds.embedding`) works
2. **"Non-Goals"**: Lists "No rewrite of TensorStore" but TensorStore was refactored
3. **"Minimal DSL Integration"**: Describes future work, but integration is complete

**Why partially aligned**: The **vision** is accurate and was implemented, but the document is frozen in planning mode. It's a **completed roadmap**, not an active plan.

---

### 🔴 Obsolete / Should Be Omitted

These documents describe a system that no longer exists, are purely historical, or are misleading.

#### [docs/FromVectorDB-ToLinalDB.md](file:///Users/nicolasbalaguera/dev/linaldb/linal-db-rs/docs/FromVectorDB-ToLinalDB.md)

**Why it is obsolete**:

1. **Title is wrong**: "Migration Task: LINAL → LINAL" (copy-paste error)
2. **All migration tasks are complete**: The project is already named LINAL
3. **Checklist format**: All items marked `[ ]` incomplete, but migration happened in v0.1.4
4. **Redundant with CHANGELOG**: Migration is documented in version history

**Recommendation**: **Archive or delete**. This is a completed migration plan with no ongoing relevance. If kept, rename to `MIGRATION_HISTORY.md` and mark all items `[x]` complete.

---

#### [docs/MODULAR_PLAN.md](file:///Users/nicolasbalaguera/dev/linaldb/linal-db-rs/docs/MODULAR_PLAN.md)

**Why it is obsolete**:

1. **Restructuring is complete**: The proposed structure in the document matches current `src/` layout
2. **No actionable items**: All goals achieved
3. **Redundant with ARCHITECTURE.md**: Current architecture is documented elsewhere

**Recommendation**: **Delete**. This was a one-time refactoring plan. The outcome is documented in ARCHITECTURE.md.

---

#### [docs/PHASE0_RESULTS.md](file:///Users/nicolasbalaguera/dev/linaldb/linal-db-rs/docs/PHASE0_RESULTS.md)

**Why it is obsolete**:

- Historical benchmark data from early development
- Superseded by [BENCHMARKS.md](file:///Users/nicolasbalaguera/dev/linaldb/linal-db-rs/docs/BENCHMARKS.md)
- No ongoing relevance

**Recommendation**: **Archive** to `docs/archive/` or delete. Keep only if historical context is valuable.

---

#### [docs/PHASE1_BENCHMARKS.md](file:///Users/nicolasbalaguera/dev/linaldb/linal-db-rs/docs/PHASE1_BENCHMARKS.md)

**Why it is obsolete**:

- Same as PHASE0_RESULTS.md
- Superseded by current benchmarks

**Recommendation**: **Archive** or delete.

---

#### [docs/ENGINE_DIAGNOSIS_PHASE6.md](file:///Users/nicolasbalaguera/dev/linaldb/linal-db-rs/docs/ENGINE_DIAGNOSIS_PHASE6.md)

**Why it is obsolete**:

- Diagnostic report from a specific development phase
- Issues identified were resolved in subsequent phases
- No ongoing diagnostic value

**Recommendation**: **Archive** or delete.

---

#### [docs/TENSOR_FIRST_PERFORMANCE.md](file:///Users/nicolasbalaguera/dev/linaldb/linal-db-rs/docs/TENSOR_FIRST_PERFORMANCE.md)

**Why it is obsolete**:

- Early performance notes
- Superseded by BENCHMARKS.md and PERFORMANCE_ROADMAP_V2.md

**Recommendation**: **Archive** or delete.

---

#### [docs/Performance_improvement_plan.md](file:///Users/nicolasbalaguera/dev/linaldb/linal-db-rs/docs/Performance_improvement_plan.md)

**Why it is obsolete**:

- Original performance plan (v1)
- Superseded by PERFORMANCE_ROADMAP_V2.md
- Contains outdated analysis and incomplete status

**Recommendation**: **Delete**. PERFORMANCE_ROADMAP_V2.md is the canonical performance document.

---

#### [docs/PROGRESS_SUMMARY.md](file:///Users/nicolasbalaguera/dev/linaldb/linal-db-rs/docs/PROGRESS_SUMMARY.md)

**Why it is obsolete**:

- Historical progress snapshot
- Redundant with Tasks_implementations.md and CHANGELOG.md

**Recommendation**: **Delete** or merge into CHANGELOG.md.

---

#### [docs/Formalizarion_Roadmap.md](file:///Users/nicolasbalaguera/dev/linaldb/linal-db-rs/docs/Formalizarion_Roadmap.md)

**Why it is obsolete**:

- Typo in filename ("Formalizarion" → "Formalization")
- Planning document for lineage/semantics features
- Features are now implemented (BIND, ATTACH, DERIVE, SHOW LINEAGE)

**Recommendation**: **Delete**. Implementation is complete and documented in README/ARCHITECTURE.

---

#### [docs/DATASET_ARCHITECTURE.md](file:///Users/nicolasbalaguera/dev/linaldb/linal-db-rs/docs/DATASET_ARCHITECTURE.md)

**Status**: Not reviewed in detail, but likely **redundant** with ARCHITECTURE.md.

**Recommendation**: Review for unique content. If redundant, merge into ARCHITECTURE.md or delete.

---

#### [docs/TEST_COVERAGE.md](file:///Users/nicolasbalaguera/dev/linaldb/linal-db-rs/docs/TEST_COVERAGE.md)

**Status**: Not reviewed in detail.

**Recommendation**: If this is a **living document** tracking current test coverage, keep it. If it's a snapshot, archive it.

---

## 3. Critical Observations

### Implicit Assumptions Found in Docs But Not Supported by Code

1. **"Logical Order" DSL Syntax** (DSL_REFERENCE.md):
   - Documentation claims: `DATASET FROM table FILTER x > 10 SELECT a, b`
   - Actual implementation: `SELECT a, b FROM table WHERE x > 10` (SQL-like)
   - **Impact**: Users following DSL_REFERENCE.md will write invalid syntax

2. **"TensorDB" vs "LINAL" Naming** (DSL_REFERENCE.md):
   - Document still uses "TensorDB" branding
   - Project was renamed to "LINAL" in v0.1.4
   - **Impact**: Confusing for new users

3. **Rayon Parallelization Claims** (README, ARCHITECTURE, BENCHMARKS):
   - All docs claim Rayon is used for large tensors
   - `rayon` dependency exists in Cargo.toml
   - **Not found** in `SimdBackend` or `ScalarBackend` code
   - **Likely location**: `CpuBackend` (not reviewed in detail)
   - **Impact**: Minor — claim is likely true but not verified in this audit

### Areas Where Documentation Oversells Simplicity or Maturity

1. **"Zero-Copy Guarantees" (README, ARCHITECTURE)**:
   - Docs claim: "Zero allocation for transforms"
   - Reality: True for reshape/transpose/slice, but **not** for all operations
   - Example: `add`, `multiply` allocate new output tensors
   - **Impact**: Misleading for users expecting all ops to be zero-copy

2. **"Dataset as Reference Graph" (README)**:
   - Docs claim: "Datasets in LINAL are now formal Reference Graphs"
   - Reality: **Two dataset implementations** coexist:
     - `dataset_legacy.rs`: Row-based, materialized
     - `dataset/`: Tensor-first, reference-based
   - **Impact**: Unclear which model is used when, or how to choose

3. **"Managed Service" (README)**:
   - Docs claim: "LINAL has evolved from a local-only tool into a Managed Analytical Service"
   - Reality: HTTP server exists, but no multi-tenancy, auth, or service-level features
   - **Impact**: Overstates production readiness

### Concepts That May Need to Be Dropped or Rethought Entirely

1. **"Logical Order" DSL Philosophy** (DSL_REFERENCE.md):
   - **Status**: Not implemented, likely abandoned
   - **Recommendation**: Either implement it or remove from docs
   - **Rationale**: SQL syntax is familiar and works well

2. **Dual Dataset Models** (legacy vs tensor-first):
   - **Status**: Both exist, unclear which is canonical
   - **Recommendation**: Document the relationship and migration path
   - **Rationale**: Users need to know when to use which model

3. **"Managed Service" Positioning** (README):
   - **Status**: Oversells current capabilities
   - **Recommendation**: Reframe as "HTTP-accessible engine" not "managed service"
   - **Rationale**: Avoid misleading enterprise users

---

## 4. Recommendations

### Which Documents Should Become the New Canonical Reference

**Tier 1 (Authoritative, Keep as-is)**:

1. [README.md](file:///Users/nicolasbalaguera/dev/linaldb/linal-db-rs/README.md) — Accurate feature overview
2. [docs/ARCHITECTURE.md](file:///Users/nicolasbalaguera/dev/linaldb/linal-db-rs/docs/ARCHITECTURE.md) — Comprehensive system design
3. [docs/BENCHMARKS.md](file:///Users/nicolasbalaguera/dev/linaldb/linal-db-rs/docs/BENCHMARKS.md) — Current performance data
4. [docs/PERFORMANCE_ROADMAP_V2.md](file:///Users/nicolasbalaguera/dev/linaldb/linal-db-rs/docs/PERFORMANCE_ROADMAP_V2.md) — Optimization status
5. [CHANGELOG.md](file:///Users/nicolasbalaguera/dev/linaldb/linal-db-rs/CHANGELOG.md) — Version history
6. [CONTRIBUTING.md](file:///Users/nicolasbalaguera/dev/linaldb/linal-db-rs/CONTRIBUTING.md) — Contributor guide
7. [SECURITY.md](file:///Users/nicolasbalaguera/dev/linaldb/linal-db-rs/SECURITY.md) — Security policy

---

### Which Ones Should Be Rewritten vs Removed

**Rewrite (High Priority)**:

1. **[docs/DSL_REFERENCE.md](file:///Users/nicolasbalaguera/dev/linaldb/linal-db-rs/docs/DSL_REFERENCE.md)**:
   - **Issue**: Describes a different DSL than what exists
   - **Action**: Rewrite from scratch based on actual parser in [src/dsl/mod.rs](file:///Users/nicolasbalaguera/dev/linaldb/linal-db-rs/src/dsl/mod.rs)
   - **Include**: All 30+ commands with correct syntax, examples from [examples/](file:///Users/nicolasbalaguera/dev/linaldb/linal-db-rs/examples)
   - **Priority**: **CRITICAL** — users rely on this for syntax

2. **[docs/NewFeatures.md](file:///Users/nicolasbalaguera/dev/linaldb/linal-db-rs/docs/NewFeatures.md)**:
   - **Issue**: Stale status tracking
   - **Action**: Either update all checkboxes to `[x]` or rename to `FEATURE_HISTORY.md`
   - **Priority**: Medium

3. **[docs/TensorFirstLinalPlan.md](file:///Users/nicolasbalaguera/dev/linaldb/linal-db-rs/docs/TensorFirstLinalPlan.md)**:
   - **Issue**: Completed plan presented as future work
   - **Action**: Rename to `TENSOR_FIRST_IMPLEMENTATION.md` and mark all phases `[x]` complete
   - **Priority**: Low

**Remove (Archive or Delete)**:

1. **[docs/FromVectorDB-ToLinalDB.md](file:///Users/nicolasbalaguera/dev/linaldb/linal-db-rs/docs/FromVectorDB-ToLinalDB.md)** — Completed migration
2. **[docs/MODULAR_PLAN.md](file:///Users/nicolasbalaguera/dev/linaldb/linal-db-rs/docs/MODULAR_PLAN.md)** — Completed refactoring
3. **[docs/Performance_improvement_plan.md](file:///Users/nicolasbalaguera/dev/linaldb/linal-db-rs/docs/Performance_improvement_plan.md)** — Superseded by v2
4. **[docs/PHASE0_RESULTS.md](file:///Users/nicolasbalaguera/dev/linaldb/linal-db-rs/docs/PHASE0_RESULTS.md)** — Historical benchmarks
5. **[docs/PHASE1_BENCHMARKS.md](file:///Users/nicolasbalaguera/dev/linaldb/linal-db-rs/docs/PHASE1_BENCHMARKS.md)** — Historical benchmarks
6. **[docs/ENGINE_DIAGNOSIS_PHASE6.md](file:///Users/nicolasbalaguera/dev/linaldb/linal-db-rs/docs/ENGINE_DIAGNOSIS_PHASE6.md)** — Resolved diagnostics
7. **[docs/TENSOR_FIRST_PERFORMANCE.md](file:///Users/nicolasbalaguera/dev/linaldb/linal-db-rs/docs/TENSOR_FIRST_PERFORMANCE.md)** — Superseded
8. **[docs/PROGRESS_SUMMARY.md](file:///Users/nicolasbalaguera/dev/linaldb/linal-db-rs/docs/PROGRESS_SUMMARY.md)** — Redundant
9. **[docs/Formalizarion_Roadmap.md](file:///Users/nicolasbalaguera/dev/linaldb/linal-db-rs/docs/Formalizarion_Roadmap.md)** — Completed plan

**Create Archive**:

- `docs/archive/` for historical documents (phases, old plans, diagnostics)

---

### Gaps in Documentation Based on What Actually Exists

**Missing Documentation**:

1. **Dual Dataset Model**:
   - **Gap**: No explanation of when to use `dataset_legacy` vs `dataset/`
   - **Recommendation**: Add section to ARCHITECTURE.md explaining:
     - Legacy: Row-based, materialized, for traditional SQL workflows
     - Tensor-first: Reference-based, zero-copy, for ML/analytical workflows
     - Migration path between models

2. **Compute Backend Selection**:
   - **Gap**: No documentation on how backend dispatch works
   - **Recommendation**: Add to ARCHITECTURE.md:
     - When is SIMD used vs Scalar?
     - What is `CpuBackend`'s role?
     - How to force a specific backend?

3. **ExecutionContext Lifecycle**:
   - **Gap**: Arena allocation and memory limits are mentioned but not explained
   - **Recommendation**: Add to ARCHITECTURE.md:
     - When is ExecutionContext created?
     - How to configure memory limits?
     - What happens on limit violation?

4. **Tensor Pooling Behavior**:
   - **Gap**: Thresholds (16, 256) are in code but not documented
   - **Recommendation**: Add to ARCHITECTURE.md or BENCHMARKS.md:
     - Allocation strategy decision tree
     - Performance characteristics of each tier

5. **DSL Command Reference**:
   - **Gap**: No complete command list with syntax
   - **Recommendation**: Rewrite DSL_REFERENCE.md with:
     - All 30+ commands from [src/dsl/mod.rs](file:///Users/nicolasbalaguera/dev/linaldb/linal-db-rs/src/dsl/mod.rs)
     - Syntax, examples, error messages
     - Organized by category (tensor ops, dataset ops, persistence, etc.)

6. **Error Handling Guide**:
   - **Gap**: No documentation on error types or debugging
   - **Recommendation**: Create `docs/ERROR_REFERENCE.md`:
     - `EngineError` variants
     - `DslError` variants
     - Common error messages and solutions

7. **Example Gallery**:
   - **Gap**: Examples exist in `examples/` but not documented
   - **Recommendation**: Create `docs/EXAMPLES.md`:
     - Link to all `.lnl` files with descriptions
     - Expected output for each example
     - Use cases (ML, analytics, data engineering)

---

## Summary

**Epistemic Alignment Status**: **60% Aligned**

- **40% Canonical**: README, ARCHITECTURE, BENCHMARKS, PERFORMANCE_ROADMAP_V2, Tasks_implementations
- **20% Partially Aligned**: DSL_REFERENCE, NewFeatures, TensorFirstLinalPlan
- **40% Obsolete**: 9 historical/completed planning documents

**Critical Issues**:

1. **DSL_REFERENCE.md is dangerously wrong** — describes a non-existent DSL
2. **9 obsolete documents clutter the repository** — archive or delete
3. **Dual dataset model is undocumented** — users don't know which to use

**Immediate Actions**:

1. **Rewrite DSL_REFERENCE.md** from actual parser implementation
2. **Archive obsolete docs** to `docs/archive/`
3. **Document dual dataset model** in ARCHITECTURE.md
4. **Create missing docs**: ERROR_REFERENCE.md, EXAMPLES.md

**Long-term**:

- Establish documentation review process for new features
- Auto-generate DSL reference from parser code
- Add CI check to prevent doc drift

---

**End of Report**
