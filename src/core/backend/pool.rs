use std::collections::HashMap;

/// Tensor allocation pool for reusing Vec<f32> allocations
/// Reduces heap allocation overhead for common tensor sizes
pub struct TensorPool {
    /// Pools of available vectors, keyed by capacity
    pools: HashMap<usize, Vec<Vec<f32>>>,
    /// Maximum number of vectors to pool per size
    max_per_size: usize,
    /// Common sizes to pool (powers of 2 and common sizes)
    pooled_sizes: Vec<usize>,
}

impl TensorPool {
    /// Create a new tensor pool with default configuration
    pub fn new() -> Self {
        Self {
            pools: HashMap::new(),
            max_per_size: 8,
            pooled_sizes: vec![128, 256, 512, 1024, 2048, 4096, 8192],
        }
    }

    /// Create a pool with custom configuration
    pub fn with_config(max_per_size: usize, pooled_sizes: Vec<usize>) -> Self {
        Self {
            pools: HashMap::new(),
            max_per_size,
            pooled_sizes,
        }
    }

    /// Acquire a vector from the pool or allocate a new one
    pub fn acquire(&mut self, size: usize) -> Vec<f32> {
        // Find the smallest pooled size that fits
        let pool_size = self.pooled_sizes.iter().find(|&&s| s >= size).copied();

        if let Some(pool_size) = pool_size {
            // Try to get from pool
            if let Some(pool) = self.pools.get_mut(&pool_size) {
                if let Some(mut vec) = pool.pop() {
                    // Reuse pooled vector
                    vec.clear();
                    vec.resize(size, 0.0);
                    return vec;
                }
            }
            // Pool empty, allocate with pooled capacity
            Vec::with_capacity(pool_size)
        } else {
            // Size too large for pooling, allocate exact size
            Vec::with_capacity(size)
        }
    }

    /// Release a vector back to the pool
    pub fn release(&mut self, mut vec: Vec<f32>) {
        let capacity = vec.capacity();

        // Only pool if capacity matches a pooled size
        if !self.pooled_sizes.contains(&capacity) {
            return; // Drop the vector
        }

        // Get or create pool for this size
        let pool = self.pools.entry(capacity).or_default();

        // Only pool if we haven't hit the limit
        if pool.len() < self.max_per_size {
            vec.clear(); // Clear data but keep capacity
            pool.push(vec);
        }
        // Otherwise drop the vector
    }

    /// Clear all pooled vectors
    pub fn clear(&mut self) {
        self.pools.clear();
    }

    /// Get statistics about the pool
    pub fn stats(&self) -> PoolStats {
        let mut total_pooled = 0;
        let mut total_capacity = 0;

        for (&size, pool) in &self.pools {
            total_pooled += pool.len();
            total_capacity += size * pool.len();
        }

        PoolStats {
            total_pooled,
            total_capacity_bytes: total_capacity * std::mem::size_of::<f32>(),
            pools_count: self.pools.len(),
        }
    }
}

impl Default for TensorPool {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about tensor pool usage
#[derive(Debug, Clone, Copy)]
pub struct PoolStats {
    pub total_pooled: usize,
    pub total_capacity_bytes: usize,
    pub pools_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_acquire_release() {
        let mut pool = TensorPool::new();

        // Acquire a vector
        let vec = pool.acquire(100);
        assert!(vec.capacity() >= 100);

        // Release it back
        pool.release(vec);

        // Acquire again - should reuse
        let vec2 = pool.acquire(100);
        assert!(vec2.capacity() >= 100);
    }

    #[test]
    fn test_pool_size_matching() {
        let mut pool = TensorPool::new();

        // Request 100 elements, should get 128 capacity
        let vec = pool.acquire(100);
        assert_eq!(vec.capacity(), 128);

        pool.release(vec);

        // Request 128 elements, should reuse the 128 capacity vec
        let vec2 = pool.acquire(128);
        assert_eq!(vec2.capacity(), 128);
    }

    #[test]
    fn test_pool_limit() {
        let mut pool = TensorPool::with_config(2, vec![128]);

        // Fill the pool
        let v1 = pool.acquire(128);
        let v2 = pool.acquire(128);
        pool.release(v1);
        pool.release(v2);

        let stats = pool.stats();
        assert_eq!(stats.total_pooled, 2);

        // Release one more - should be dropped (not pooled)
        let v3 = pool.acquire(128);
        pool.release(v3);

        let stats = pool.stats();
        assert_eq!(stats.total_pooled, 2); // Still 2, not 3
    }

    #[test]
    fn test_large_allocation() {
        let mut pool = TensorPool::new();

        // Request very large size (not in pooled sizes)
        let vec = pool.acquire(100_000);
        assert!(vec.capacity() >= 100_000);

        // Release - should not be pooled
        pool.release(vec);

        let stats = pool.stats();
        assert_eq!(stats.total_pooled, 0);
    }

    #[test]
    fn test_clear() {
        let mut pool = TensorPool::new();

        let v1 = pool.acquire(128);
        let v2 = pool.acquire(256);
        pool.release(v1);
        pool.release(v2);

        assert_eq!(pool.stats().total_pooled, 2);

        pool.clear();
        assert_eq!(pool.stats().total_pooled, 0);
    }
}
