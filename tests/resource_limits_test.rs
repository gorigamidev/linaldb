use linal::engine::context::{ExecutionContext, ResourceError, DEFAULT_MEMORY_LIMIT};

#[test]
fn test_memory_limit_enforcement() {
    // Create context with 1KB limit
    let ctx = ExecutionContext::with_memory_limit(1024);

    assert_eq!(ctx.memory_limit(), Some(1024));
    assert_eq!(ctx.memory_usage(), 0);

    // Small allocation should succeed
    let result = ctx.alloc_tensor_data(10); // 10 * 4 bytes = 40 bytes
    assert!(result.is_ok());

    // Large allocation should fail
    let result = ctx.alloc_tensor_data(1000); // 1000 * 4 bytes = 4000 bytes
    assert!(result.is_err());

    if let Err(ResourceError::MemoryLimitExceeded {
        requested,
        limit,
        current,
    }) = result
    {
        assert_eq!(limit, 1024);
        assert!(requested > limit - current);
    } else {
        panic!("Expected MemoryLimitExceeded error");
    }
}

#[test]
fn test_no_memory_limit() {
    // Context without limit should allow any allocation
    let ctx = ExecutionContext::new();

    assert_eq!(ctx.memory_limit(), None);

    // Large allocation should succeed
    let result = ctx.alloc_tensor_data(10_000);
    assert!(result.is_ok());
}

#[test]
fn test_default_memory_limit() {
    let ctx = ExecutionContext::with_memory_limit(DEFAULT_MEMORY_LIMIT);

    assert_eq!(ctx.memory_limit(), Some(DEFAULT_MEMORY_LIMIT));

    // Should be able to allocate reasonable amounts
    let result = ctx.alloc_tensor_data(1_000_000); // ~4MB
    assert!(result.is_ok());

    // Should fail for very large allocations
    let result = ctx.alloc_tensor_data(100_000_000); // ~400MB
    assert!(result.is_err());
}

#[test]
fn test_memory_tracking() {
    let ctx = ExecutionContext::with_memory_limit(10_000);

    let initial = ctx.memory_usage();

    // Allocate some data
    ctx.alloc_tensor_data(100).unwrap();

    let after_alloc = ctx.memory_usage();
    assert!(after_alloc > initial);

    // Allocate more
    ctx.alloc_tensor_data(200).unwrap();

    let after_second = ctx.memory_usage();
    assert!(after_second > after_alloc);
}

#[test]
fn test_check_allocation() {
    let ctx = ExecutionContext::with_memory_limit(1000);

    // Check small allocation
    assert!(ctx.check_allocation(100).is_ok());

    // Check large allocation
    assert!(ctx.check_allocation(2000).is_err());

    // After actual allocation, available space decreases
    ctx.alloc_tensor_data(50).unwrap();

    // Now a smaller allocation might fail
    let bytes_used = ctx.memory_usage();
    let remaining = 1000 - bytes_used;

    // Allocation within remaining should succeed
    assert!(ctx.check_allocation(remaining / 2).is_ok());

    // Allocation exceeding remaining should fail
    assert!(ctx.check_allocation(remaining + 100).is_err());
}

#[test]
fn test_arena_reset_with_limit() {
    let mut ctx = ExecutionContext::with_memory_limit(5000);

    // Allocate some data
    ctx.alloc_tensor_data(100).unwrap();
    let usage_before = ctx.memory_usage();
    assert!(usage_before > 0);

    // Reset clears tracking but arena memory is still allocated
    // (Bump::reset() resets the allocation pointer but doesn't free memory)
    ctx.reset();

    // Memory usage may not be zero because Bump keeps allocated chunks
    // But we should be able to allocate again
    assert!(ctx.alloc_tensor_data(100).is_ok());

    // And the limit should still be enforced
    assert_eq!(ctx.memory_limit(), Some(5000));
}

#[test]
fn test_resource_error_display() {
    let error = ResourceError::MemoryLimitExceeded {
        requested: 1000,
        limit: 500,
        current: 200,
    };

    let error_string = format!("{}", error);
    assert!(error_string.contains("Memory limit exceeded"));
    assert!(error_string.contains("1000"));
    assert!(error_string.contains("500"));
    assert!(error_string.contains("200"));
}
