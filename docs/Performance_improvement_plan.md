# Performance & Resource Management Plan

This document defines a **review and implementation plan** to improve performance,
memory management, and resource utilization in LINAL, leveraging Rust’s strengths
(deterministic memory, zero-cost abstractions, safe concurrency).

The goal is to make LINAL **predictable, fast, and production-usable** across
CLI, embedded, and server deployments.

---

## 1. Goals

- Achieve **low-latency, predictable execution** (no GC pauses).
- Use **memory efficiently** (RAM first, controlled growth).
- Enable **safe concurrency** and parallel execution.
- Prepare the engine for **CPU SIMD and optional GPU acceleration**.
- Prevent resource exhaustion when running as a long-lived server.

---

## 2. Guiding Principles

1. **No Garbage Collector**
   - Memory must be released deterministically.
   - Prefer scoped lifetimes over background cleanup.

2. **Execution is Scoped**
   - Every query runs inside an explicit execution context.
   - All temporary allocations must belong to that scope.

3. **Zero-Copy by Default**
   - Avoid copying tensors and datasets unless strictly necessary.
   - Favor views, references, and shared ownership (`Arc`).

4. **In-Memory First, Disk as Backup**
   - RAM for hot data and indexes.
   - Disk (Parquet / JSON) for persistence and recovery.

5. **Measure Before Optimizing**
   - No optimization without benchmarks and metrics.

---

## 3. Execution Context & Memory Lifecycle

### 3.1 Introduce `ExecutionContext`

Create a per-query execution context responsible for all temporary state:

```rust
pub struct ExecutionContext<'a> {
    pub arena: Bump,
    pub temp_tensors: Vec<TensorId>,
    pub temp_datasets: Vec<DatasetId>,
    pub engine: &'a TensorDb,
}
````

#### Responsibilities

- Own all temporary allocations.
- Track intermediate tensors and datasets.
- Ensure **automatic cleanup on drop**.

When execution finishes:

```rust
drop(ctx); // deterministic cleanup
```

---

### 3.2 Arena Allocation (Critical)

Use arena/bump allocation for:

- Tuples
- Intermediate values
- Expression evaluation results
- Query execution buffers

Recommended crates:

- `bumpalo`
- or a custom simple arena

### *Benefits*

- O(1) allocation
- No fragmentation
- Instant cleanup at end of query
- No GC-like behavior

---

## 4. Tensor & Dataset Memory Strategy

### 4.1 Tensor Ownership

- Tensor data stored as:

  ```rust
  Arc<Vec<f32>>
  ```

- Tensor operations should:

  - Share data when possible
  - Create views/slices instead of copies
  - Clone only metadata, not data

### 4.2 Dataset Storage

- Keep datasets immutable during query execution.
- Use copy-on-write semantics for updates.
- Prefer columnar layouts for batch processing (future step).

---

## 5. Query Execution Performance

### 5.1 Vectorized / Batch Execution

Move from tuple-at-a-time execution to batch-based execution:

- Process rows in chunks (e.g. 1024 rows).
- Apply filters, projections, and aggregations per batch.

### *Benefits 2*

- Better CPU cache utilization
- Lower function call overhead
- SIMD-friendly execution

---

### 5.2 Index-Aware Execution

Ensure planner always considers:

- HashIndex for equality predicates
- VectorIndex for similarity search

Avoid full scans when an index is available.

---

## 6. SIMD & CPU Optimization

### 6.1 SIMD Kernels

Critical operations to SIMD-optimize:

- Dot product
- Cosine similarity
- L2 distance
- Matrix multiplication

Strategy:

- Use feature detection at compile-time.
- Provide scalar fallback.

Example:

```rust
#[cfg(target_feature = "avx2")]
fn dot_simd(...) { ... }
```

---

### 6.2 Cache-Friendly Layout

- Use contiguous memory for vectors and matrices.
- Avoid nested allocations.
- Minimize pointer chasing in hot loops.

---

## 7. GPU Acceleration (Optional, Future)

GPU is **optional**, not default.

Suitable workloads:

- Large batch vector similarity
- Matrix-heavy analytics
- ML-related computations

Design principle:

- Abstract compute backend:

  ```rust
  trait ComputeBackend {
      fn matmul(...);
      fn cosine(...);
  }
  ```

Backends:

- CPU (default)
- GPU (CUDA / Metal / wgpu)

---

## 8. Server Resource Control

When running as an HTTP server:

### 8.1 Memory Limits

- Maximum arena size per request.
- Reject or abort queries exceeding limits.

### 8.2 Concurrency Limits

- Limit number of concurrent queries.
- Backpressure instead of overload.

### 8.3 Timeouts

- Enforce query execution timeout.
- Kill execution context safely.

---

## 9. Index Lifecycle Management

Indexes require **explicit lifecycle handling**:

- Invalidate on DELETE / UPDATE.
- Lazy rebuild when needed.
- Track reference counts per dataset.

This is **not GC**, but lifecycle management.

---

## 10. Benchmarks & Observability

### 10.1 Benchmarks

Use `criterion` for:

- Query execution latency
- Vector similarity throughput
- Aggregation performance

Benchmarks must run in CI.

---

### 10.2 Metrics (Server Mode)

Track:

- Query duration
- Memory usage per query
- Active execution contexts
- Index hit/miss ratio

---

## 11. Phased Implementation Plan

> [!IMPORTANT]
> Each phase must be implemented incrementally with backward compatibility.
> New features should be **additive**, not replacements, until proven stable.

### Phase 0 – Measurement & Baseline (COMPLETED)

**Goal**: Establish performance baselines before any optimization.

**Tasks**:

1. **Benchmark Suite**
   - Add `criterion` benchmarks for:
     - Tensor operations (dot product, matmul, element-wise ops)
     - Query execution (SELECT, FILTER, GROUP BY, aggregations)
     - Index operations (hash lookup, vector similarity search)
     - Dataset operations (INSERT, UPDATE, DELETE)
   - Create `benches/` directory with modular benchmark files
   - Run benchmarks in CI to detect regressions

2. **Memory Profiling**
   - Add memory tracking utilities
   - Measure peak memory usage per operation type
   - Profile allocation patterns (frequency, size distribution)
   - Use `heaptrack` or `valgrind` for detailed analysis

3. **Performance Tests**
   - Create regression test suite
   - Define acceptable performance thresholds
   - Test with realistic workloads (1K, 10K, 100K rows)

4. **Documentation**
   - Document current performance characteristics
   - Create `docs/BENCHMARKS.md` with baseline results
   - Establish performance goals for each phase

**Success Criteria**:

- Reproducible benchmark results
- Documented baseline performance
- CI integration for regression detection

---

### Phase 1 – Non-Breaking Foundations (COMPLETED)

**Goal**: Add new infrastructure without replacing existing code.

**Risk Level**: LOW (purely additive)

#### 1.1 ExecutionContext (Optional Wrapper)

Add `ExecutionContext` as an **optional** parameter, keeping existing APIs working:

```rust
// New struct in src/engine/context.rs
pub struct ExecutionContext<'a> {
    pub arena: Bump,
    pub temp_tensors: Vec<TensorId>,
    pub temp_datasets: Vec<DatasetId>,
    pub engine: &'a TensorDb,
}

impl<'a> ExecutionContext<'a> {
    pub fn new(engine: &'a TensorDb) -> Self {
        Self {
            arena: Bump::new(),
            temp_tensors: Vec::new(),
            temp_datasets: Vec::new(),
            engine,
        }
    }
}

impl Drop for ExecutionContext<'_> {
    fn drop(&mut self) {
        // Clean up temporary resources
        for tid in &self.temp_tensors {
            let _ = self.engine.remove_tensor(*tid);
        }
        for did in &self.temp_datasets {
            let _ = self.engine.remove_dataset(*did);
        }
    }
}
```

**Migration Strategy**:

- Add new methods: `execute_with_context()` alongside existing `execute()`
- Existing code continues to work unchanged
- New code can opt-in to using context
- Gradually migrate hot paths to use context

#### 1.2 Arena Allocation (Isolated)

Add arena allocation for **new** temporary allocations only:

```rust
// Use arena for expression evaluation temporaries
impl ExecutionContext<'_> {
    pub fn alloc_temp<T>(&self, value: T) -> &T {
        self.arena.alloc(value)
    }
    
    pub fn alloc_slice<T: Copy>(&self, slice: &[T]) -> &[T] {
        self.arena.alloc_slice_copy(slice)
    }
}
```

**Migration Strategy**:

- Start with expression evaluation (small scope)
- Don't touch existing tensor/dataset allocations yet
- Measure memory reduction in benchmarks

#### 1.3 Tensor Arc Migration (Parallel Implementation)

Add `Arc`-based storage **alongside** existing `Vec`:

```rust
// In src/core/tensor.rs
pub struct Tensor {
    pub shape: Shape,
    // Keep existing field for compatibility
    pub data: Vec<f32>,
    // Add new field for zero-copy sharing
    pub shared_data: Option<Arc<Vec<f32>>>,
}

impl Tensor {
    // New constructor for shared tensors
    pub fn from_shared(shape: Shape, data: Arc<Vec<f32>>) -> Self {
        Self {
            shape,
            data: Vec::new(), // Empty, using shared_data instead
            shared_data: Some(data),
        }
    }
    
    // Accessor that works with both
    pub fn data(&self) -> &[f32] {
        self.shared_data.as_ref()
            .map(|arc| arc.as_slice())
            .unwrap_or(&self.data)
    }
    
    // Create a view/reference without copying
    pub fn share(&self) -> Self {
        let shared = self.shared_data.clone()
            .unwrap_or_else(|| Arc::new(self.data.clone()));
        Self::from_shared(self.shape.clone(), shared)
    }
}
```

**Migration Strategy**:

- Existing code uses `data` field (no changes needed)
- New code uses `shared_data` for zero-copy
- Gradually migrate operations to use `share()` instead of `clone()`
- Remove `data` field in Phase 2 after full migration

**Success Criteria**:

- All existing tests pass
- Benchmarks show no regression
- New context-based code paths tested
- Memory profiling shows reduced allocations

---

### Phase 2 – Tensor Zero-Copy Migration (COMPLETED)

**Goal**: Fully migrate to Arc-based tensor sharing.

**Risk Level**: MEDIUM (requires careful migration)

#### 2.1 Complete Tensor Migration

- Remove `data` field, keep only `shared_data`
- Update all tensor operations to use Arc
- Implement copy-on-write for mutations:

```rust
impl Tensor {
    pub fn make_mut(&mut self) -> &mut Vec<f32> {
        Arc::make_mut(&mut self.shared_data)
    }
}
```

#### 2.2 Tensor Views and Slicing

Add zero-copy views:

```rust
pub struct TensorView<'a> {
    data: &'a [f32],
    shape: Shape,
    stride: Vec<usize>,
}
```

#### 2.3 Operation Optimization

Update kernels to avoid unnecessary copies:

- Binary ops: share when possible, copy only when needed
- Transpose: create view instead of copying
- Reshape: metadata-only operation

**Migration Strategy**:

- Use feature flag: `--features zero-copy`
- Run full test suite with flag enabled
- Fix any breaking changes incrementally
- Make default after stabilization

**Success Criteria**:

- 50%+ reduction in tensor memory allocations
- No performance regression in benchmarks
- All tests pass with zero-copy enabled

---

### Phase 3 – Query Execution Improvements

**Goal**: Add batch execution and SIMD optimizations.

**Risk Level**: MEDIUM (new execution paths)

#### 3.1 Batch Execution Mode

Add batch executor **alongside** existing tuple-at-a-time:

```rust
pub enum ExecutionMode {
    TupleAtATime,  // Existing
    Batched(usize), // New: batch size
}

impl Executor {
    pub fn execute_batched(&self, plan: &PhysicalPlan, batch_size: usize) 
        -> Result<Dataset> {
        // New batched implementation
    }
}
```

**Migration Strategy**:

- Keep existing `execute()` method unchanged
- Add `execute_batched()` as new method
- Use heuristics to choose execution mode
- Make ExecutionContext choose mode automatically

#### 3.2 SIMD Kernels

Add SIMD implementations with fallbacks:

```rust
// In src/engine/kernels.rs
#[cfg(target_feature = "avx2")]
mod simd {
    pub fn dot_product_simd(a: &[f32], b: &[f32]) -> f32 {
        // AVX2 implementation
    }
}

pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_feature = "avx2")]
    return simd::dot_product_simd(a, b);
    
    #[cfg(not(target_feature = "avx2"))]
    return dot_product_scalar(a, b);
}
```

**Migration Strategy**:

- Implement SIMD for hot paths first (dot, cosine, matmul)
- Always provide scalar fallback
- Use runtime feature detection where needed
- Benchmark SIMD vs scalar to verify speedup

#### 3.3 Mandatory ExecutionContext

Make ExecutionContext required for all query execution:

```rust
// Old API (deprecated)
#[deprecated(note = "Use execute_with_context instead")]
pub fn execute(&self, query: &str) -> Result<Dataset> {
    let ctx = ExecutionContext::new(self);
    self.execute_with_context(&ctx, query)
}

// New API (required)
pub fn execute_with_context(&self, ctx: &ExecutionContext, query: &str) 
    -> Result<Dataset> {
    // Implementation
}
```

**Success Criteria**:

- 2-5x speedup for batch-friendly queries
- SIMD kernels show measurable improvement
- ExecutionContext cleanup prevents memory leaks

---

### Phase 4 – Server Hardening

**Goal**: Add resource controls for production server deployment.

**Risk Level**: LOW (mostly additive middleware)

#### 4.1 Memory Limits

Add per-request memory limits:

```rust
pub struct ResourceLimits {
    pub max_arena_size: usize,
    pub max_result_rows: usize,
    pub max_temp_tensors: usize,
}

impl ExecutionContext<'_> {
    pub fn with_limits(engine: &TensorDb, limits: ResourceLimits) -> Self {
        // Create context with limits
    }
    
    fn check_limits(&self) -> Result<()> {
        if self.arena.allocated_bytes() > self.limits.max_arena_size {
            return Err(EngineError::MemoryLimitExceeded);
        }
        Ok(())
    }
}
```

#### 4.2 Concurrency Control

Add request queue with backpressure:

```rust
// In src/server/mod.rs
pub struct RequestQueue {
    max_concurrent: usize,
    semaphore: Arc<Semaphore>,
}
```

#### 4.3 Query Timeouts

Add timeout enforcement:

```rust
pub async fn execute_with_timeout(
    ctx: &ExecutionContext,
    query: &str,
    timeout: Duration,
) -> Result<Dataset> {
    tokio::time::timeout(timeout, execute_async(ctx, query))
        .await
        .map_err(|_| EngineError::QueryTimeout)?
}
```

**Migration Strategy**:

- Add as server middleware (doesn't affect engine core)
- Make limits configurable via `linal.toml`
- Start with generous limits, tune based on production metrics

**Success Criteria**:

- Server handles sustained load without OOM
- Graceful degradation under overload
- Configurable resource limits work correctly

---

### Phase 5 – Advanced Optimizations (Future)

**Goal**: Advanced features for specialized workloads.

**Risk Level**: HIGH (significant architectural changes)

#### 5.1 Cost-Based Optimizer

- Collect statistics during execution
- Build cost model for operations
- Choose optimal execution strategy

#### 5.2 Columnar Execution Engine

- Columnar memory layout for datasets
- Vectorized execution over columns
- Better cache utilization

#### 5.3 GPU Backend (Optional)

- Abstract compute backend trait
- CPU implementation (default)
- GPU implementation (CUDA/Metal/wgpu)
- Automatic workload routing

**Migration Strategy**:

- These are long-term goals
- Require significant design work
- Should be separate RFCs/design docs
- Only pursue after Phase 1-4 complete and stable

---

## 12. Safety Measures & Migration Strategies

> [!CAUTION]
> Never replace working code without a proven alternative.
> All changes must be reversible until fully validated.

### 12.1 Feature Flags

Use Cargo features to toggle new implementations:

```toml
# Cargo.toml
[features]
default = []
zero-copy = []
batched-execution = []
simd = []
experimental = ["zero-copy", "batched-execution", "simd"]
```

**Usage**:

```rust
#[cfg(feature = "zero-copy")]
fn optimized_path() { ... }

#[cfg(not(feature = "zero-copy"))]
fn legacy_path() { ... }
```

### 12.2 Parallel Implementations

Keep old and new code paths running in parallel:

```rust
pub struct QueryExecutor {
    legacy_executor: LegacyExecutor,
    batched_executor: Option<BatchedExecutor>,
}

impl QueryExecutor {
    pub fn execute(&self, plan: &PhysicalPlan) -> Result<Dataset> {
        if let Some(batched) = &self.batched_executor {
            // Try new path
            match batched.execute(plan) {
                Ok(result) => return Ok(result),
                Err(e) => {
                    warn!("Batched execution failed: {}, falling back", e);
                }
            }
        }
        // Fallback to legacy
        self.legacy_executor.execute(plan)
    }
}
```

### 12.3 Comprehensive Testing Strategy

#### Unit Tests

- Test new implementations in isolation
- Maintain 100% test coverage for new code
- Use property-based testing for correctness

#### Integration Tests

- Test new and old paths produce identical results
- Use snapshot testing for query results
- Test edge cases and error conditions

#### Performance Tests

- Benchmark before and after each change
- Set performance regression thresholds (e.g., no more than 5% slower)
- Test with realistic workloads

#### Compatibility Tests

```rust
#[test]
fn test_legacy_vs_new_execution() {
    let query = "SELECT * FROM users WHERE age > 25";
    let legacy_result = legacy_execute(query);
    let new_result = new_execute(query);
    assert_eq!(legacy_result, new_result);
}
```

### 12.4 Incremental Migration Checklist

For each phase:

- [ ] Create feature branch
- [ ] Implement new code alongside existing code
- [ ] Add comprehensive tests
- [ ] Run benchmarks and compare with baseline
- [ ] Enable feature flag in CI
- [ ] Run full test suite with new code
- [ ] Fix any issues found
- [ ] Merge to main with feature flag OFF by default
- [ ] Monitor in production (if applicable)
- [ ] Enable feature flag by default after validation
- [ ] Remove old code in next phase

### 12.5 Rollback Procedures

**If a phase causes issues**:

1. **Immediate**: Disable feature flag
2. **Short-term**: Revert merge if needed
3. **Investigation**: Identify root cause
4. **Fix**: Address issues in new branch
5. **Retry**: Re-enable after fixes validated

**Rollback triggers**:

- Performance regression > 10%
- Memory usage increase > 20%
- Any correctness issues
- Test failures in CI

### 12.6 Monitoring & Validation

Add observability for new code paths:

```rust
// Track execution mode usage
metrics::counter!("query.execution.mode", 1, "mode" => "batched");

// Track performance
let start = Instant::now();
let result = execute(query);
metrics::histogram!("query.duration", start.elapsed());

// Track memory
metrics::gauge!("arena.allocated_bytes", ctx.arena.allocated_bytes());
```

### 12.7 Documentation Requirements

For each phase:

- Update ARCHITECTURE.md with new components
- Document migration path in CHANGELOG.md
- Add code examples for new APIs
- Update benchmarks documentation
- Create migration guide for users (if API changes)

### 12.8 Code Review Guidelines

All performance changes must:

- Include benchmark results in PR description
- Show memory impact (before/after)
- Explain why the change is safe
- Demonstrate backward compatibility
- Include rollback plan

---

## 13. Success Criteria

### Overall Project Goals

- **No memory leaks**: Valgrind/heaptrack shows zero leaks after 1M operations
- **Stable latency**: P99 latency variance < 10% under sustained load
- **Predictable RAM usage**: Memory growth < 5% over 24-hour server run
- **Competitive performance**: Within 2x of DuckDB/DataFusion for comparable workloads

### Phase-Specific Metrics

#### Phase 0: Baseline

- ✅ Reproducible benchmarks with < 5% variance
- ✅ Documented baseline for all critical operations
- ✅ CI integration with regression detection

#### Phase 1: Foundations

- ✅ ExecutionContext cleanup prevents resource leaks
- ✅ Arena allocation reduces allocation count by 30%+
- ✅ Zero performance regression in existing code paths
- ✅ All existing tests pass

#### Phase 2: Zero-Copy

- ✅ 50%+ reduction in tensor memory allocations
- ✅ Copy-on-write prevents unnecessary data duplication
- ✅ Tensor operations 20%+ faster due to reduced copying

#### Phase 3: Execution

- ✅ Batched execution 2-5x faster for large datasets (>10K rows)
- ✅ SIMD kernels 2-4x faster for vector operations
- ✅ Query latency P50 improved by 30%+

#### Phase 4: Server

- ✅ Server handles 1000+ concurrent requests without OOM
- ✅ Graceful degradation under overload (no crashes)
- ✅ Configurable limits prevent resource exhaustion

### Validation Tests

```rust
// Memory leak test
#[test]
fn no_memory_leaks_after_million_queries() {
    let start_mem = current_memory_usage();
    for _ in 0..1_000_000 {
        execute_query("SELECT * FROM test");
    }
    let end_mem = current_memory_usage();
    assert!(end_mem - start_mem < MB(10)); // < 10MB growth
}

// Latency stability test
#[test]
fn stable_latency_under_load() {
    let latencies: Vec<Duration> = (0..10_000)
        .map(|_| measure_query_latency())
        .collect();
    let p99 = percentile(&latencies, 0.99);
    let p50 = percentile(&latencies, 0.50);
    assert!(p99 < p50 * 3); // P99 < 3x P50
}
```

---

## 13. Non-Goals

- No tracing garbage collector.
- No distributed execution (yet).
- No premature GPU dependency.

---

This plan intentionally favors **predictability and correctness first**, then
throughput and scale.

Rust’s strengths are fully leveraged when **lifetimes and scopes are explicit**.
