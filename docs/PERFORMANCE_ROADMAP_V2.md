# LINAL Performance Roadmap (v2): Scaling & Optimization

This document reconciles the original performance plan with the current state of LINAL (v0.1.9). It addresses the significant regressions observed in Phase 6 and outlines the remaining high-impact tasks.

## 1. Regression Analysis (Phase 6 Post-Mortem)

Latest benchmarks show **34% to 75% regressions** in core vector/matrix operations.

### Identified Bottlenecks

- **Metadata Allocation**: `TensorMetadata::new` (and `Utc::now()`) is called in every math kernel. For 128-element vectors, the syscall overhead dwarfs the math.
- **SIMD Zero-Filling**: `SimdBackend` uses `vec![0.0; len]` which zero-fills memory before immediately overwriting it with SIMD results.
- **Excessive Boxing**: Trait objects and Arc clones add small but cumulative overhead in tight loops.
- **Non-Optimized Standard Paths**: `matmul` and `cosine_similarity` are still using scalar fallbacks.

---

## 2. Revised Checklist (Remaining Tasks)

### Phase 7: The "Zero-Overhead" Push

**Goal**: Recovery of baseline performance and optimization of internal kernels.

- [x] **Bypass Metadata Syscalls**: Reuse metadata templates or use a non-syscall timestamp for intermediate tensors.
- [x] **Uninitialized Allocation**: Optimize `alloc_output` to use `Vec::with_capacity` plus unsafe `set_len` or similar (safely wrapped) to avoid zero-filling.
- [x] **Kernel Specialization**: Remove `elementwise_binary_op` abstraction for hot paths. Implement dedicated loops for same-shape tensors.
- [x] **Fast-Path Benchmarking**: Target `< 1µs` for `vector_add/128`. (Achieved ~10% improvement, ~2µs).

### Phase 8: True Zero-Copy Views

**Goal**: Eliminate data copies for structural transformations.

- [x] **Metadata-Only Reshape**: Update `reshape` to never clone the data Arc, only the Metadata with a new shape.
- [x] **Metadata-Only Transpose**: Implement `TensorView` with strides so transpose is a O(1) metadata swap.
- [x] **Metadata-Only Slice**: Implement slicing as a view over the same `Arc<Vec<f32>>`.

### Phase 9: Batch & Parallel Execution

**Goal**: Vertical scaling for large datasets.

- [x] **Dataset Batching**: Process datasets in 1024-row chunks instead of row-by-row.
- [x] **Multi-Threaded Kernels**: Use `rayon` for large tensor operations (e.g., > 1M elements).
- [x] **SIMD expansion**: Implement SIMD for `matmul` (tiled approach) and `divide`.

### Phase 10: Server & Resource Governance

**Goal**: Production stability.

- [x] **Arena-Backed Tensors**: Allow `ExecutionContext` to allocate tensor data in the `Bump` arena for ephemeral results.
- [x] **Memory Limit Enforcement**: Kill queries that exceed per-context arena limits.
- [ ] **Persistent Scheduler Queue**: Move tasks to a disk-backed queue.
  - **Status**: Deferred - Not a performance optimization
  - **Reason**: Requires disk backend (sled/rocksdb), task serialization, server changes
  - **Scope**: This is a **reliability feature**, not a performance feature
  - **Future**: Implement when server-side task scheduling/persistence is prioritized

### Phase 11: Allocation Optimization - **COMPLETE**

**Goal**: Reduce heap allocation overhead.

- [x] **Tensor Pooling**: Reuse `Vec<f32>` allocations for common sizes.
  - **Status**: Complete with size threshold optimization
  - **Wins**: 18% faster for medium tensors (4096 elements)
  - **Fix**: Size threshold (256) eliminates small tensor overhead
- [x] **Arena Integration**: Infrastructure in place via `ExecutionContext::alloc_tensor_data()`
- [x] **Stack Tensors**: `SmallVec<[f32; 16]>` for tensors ≤16 elements (zero heap allocation)

**Result**: 3-18% improvement across operations, zero regression

---

## 3. Current Implementation Status vs Original Plan

| Feature | Original Phase | Status | Notes |
| :--- | :--- | :--- | :--- |
| `ExecutionContext` | Phase 1 | **Done** | Basic implementation exists. |
| `Arc` Tensors | Phase 2 | **Done** | Core is shared, but views still copy data. |
| Zero-Copy Views | Phase 2 | **Done** | Reshape/Transpose/Slice are metadata-only. |
| SIMD Kernels | Phase 3 | **Done** | Add/Sub/Mul/Matmul with SIMD. |
| Batch Execution | Phase 3 | **Done** | Dataset batching implemented. |
| Rayon Parallelization | Phase 3 | **Done** | Multi-threaded kernels for large tensors. |
| Asset Timeouts | Phase 4 | **Done** | Implemented in Server. |
| Resource Limits | Phase 4 | **Done** | Memory limits + arena allocation. |
| Tensor Pooling | Phase 11 | **Complete** | Size threshold optimization, zero regression. |

---
