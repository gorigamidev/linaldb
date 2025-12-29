use crate::core::dataset_legacy::DatasetId;
use crate::core::tensor::{ExecutionId, TensorId};
use bumpalo::Bump;

/// Execution context for a single query/operation.
/// Manages temporary allocations and ensures automatic cleanup.
pub struct ExecutionContext {
    /// Unique ID for this execution
    execution_id: ExecutionId,
    /// Arena allocator for temporary values
    arena: Bump,
    /// Temporary tensors to clean up on drop (lazily initialized)
    temp_tensors: Option<Vec<TensorId>>,
    /// Temporary datasets to clean up on drop (lazily initialized)
    temp_datasets: Option<Vec<DatasetId>>,
}

impl ExecutionContext {
    /// Create a new execution context
    pub fn new() -> Self {
        Self {
            execution_id: ExecutionId::new(),
            arena: Bump::new(),
            temp_tensors: None,
            temp_datasets: None,
        }
    }

    /// Create with specific arena capacity
    pub fn with_capacity(bytes: usize) -> Self {
        Self {
            execution_id: ExecutionId::new(),
            arena: Bump::with_capacity(bytes),
            temp_tensors: None,
            temp_datasets: None,
        }
    }

    pub fn execution_id(&self) -> ExecutionId {
        self.execution_id
    }

    /// Reset the context for reuse without reallocating
    pub fn reset(&mut self) {
        self.arena.reset();

        if let Some(v) = &mut self.temp_tensors {
            v.clear();
        }
        if let Some(v) = &mut self.temp_datasets {
            v.clear();
        }
    }

    /// Allocate a temporary value in the arena
    #[inline(always)]
    pub fn alloc_temp<T>(&self, value: T) -> &T {
        self.arena.alloc(value)
    }

    /// Allocate a temporary slice in the arena
    #[inline(always)]
    pub fn alloc_slice<T: Copy>(&self, slice: &[T]) -> &[T] {
        self.arena.alloc_slice_copy(slice)
    }

    /// Allocate a temporary vec in the arena
    pub fn alloc_vec<T>(&self, vec: Vec<T>) -> &[T] {
        self.arena.alloc_slice_fill_iter(vec.into_iter())
    }

    /// Track a temporary tensor for cleanup
    pub fn track_tensor(&mut self, id: TensorId) {
        self.temp_tensors
            .get_or_insert_with(|| Vec::with_capacity(8))
            .push(id);
    }

    /// Track a temporary dataset for cleanup
    pub fn track_dataset(&mut self, id: DatasetId) {
        self.temp_datasets
            .get_or_insert_with(|| Vec::with_capacity(8))
            .push(id);
    }

    /// Get tracked tensors (for cleanup)
    /// Note: Will be used in Phase 2 for proper resource cleanup
    #[allow(dead_code)]
    pub(crate) fn temp_tensors(&self) -> &[TensorId] {
        self.temp_tensors.as_deref().unwrap_or(&[])
    }

    /// Get tracked datasets (for cleanup)
    /// Note: Will be used in Phase 2 for proper resource cleanup
    #[allow(dead_code)]
    pub(crate) fn temp_datasets(&self) -> &[DatasetId] {
        self.temp_datasets.as_deref().unwrap_or(&[])
    }

    /// Clear tracked resources
    pub(crate) fn clear_tracked(&mut self) {
        if let Some(v) = &mut self.temp_tensors {
            v.clear();
        }
        if let Some(v) = &mut self.temp_datasets {
            v.clear();
        }
    }

    /// Get arena statistics
    pub fn arena_stats(&self) -> ArenaStats {
        ArenaStats {
            allocated_bytes: self.arena.allocated_bytes(),
        }
    }
}

impl Default for ExecutionContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about arena allocation
#[derive(Debug, Clone, Copy)]
pub struct ArenaStats {
    pub allocated_bytes: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_creation() {
        let ctx = ExecutionContext::new();
        assert_eq!(ctx.arena_stats().allocated_bytes, 0);
    }

    #[test]
    fn test_arena_allocation() {
        let ctx = ExecutionContext::new();

        let val = ctx.alloc_temp(42);
        assert_eq!(*val, 42);

        let slice = ctx.alloc_slice(&[1, 2, 3]);
        assert_eq!(slice, &[1, 2, 3]);
    }

    #[test]
    fn test_tracking() {
        let mut ctx = ExecutionContext::new();

        ctx.track_tensor(TensorId::new());
        ctx.track_dataset(DatasetId(2));

        assert_eq!(ctx.temp_tensors().len(), 1);
        assert_eq!(ctx.temp_datasets().len(), 1);

        ctx.clear_tracked();
        assert_eq!(ctx.temp_tensors().len(), 0);
        assert_eq!(ctx.temp_datasets().len(), 0);
    }
}
