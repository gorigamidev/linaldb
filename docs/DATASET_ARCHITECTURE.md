# Dataset Architecture Explanation

## Two Dataset Implementations

### 1. `dataset_legacy.rs` - **Currently Active**

**Location**: `src/core/dataset_legacy.rs`

**Structure**:

```rust
pub struct Dataset {
    pub id: DatasetId,
    pub schema: Arc<Schema>,
    pub rows: Vec<Tuple>,  // ← Stores actual row data
    pub metadata: DatasetMetadata,
    pub indices: HashMap<String, Box<dyn Index>>,
    pub lazy_expressions: HashMap<String, Expr>,
}
```

**Characteristics**:

- **Row-based storage**: Stores actual `Vec<Tuple>` data
- **Used by**: DSL (`src/dsl/mod.rs`), Engine
- **Operations**: filter, map, select, join, etc.
- **Memory**: Copies data for transformations
- **Status**: **Active** - this is what the engine uses

### 2. `dataset/dataset.rs` - **Future Zero-Copy Design**

**Location**: `src/core/dataset/dataset.rs`

**Structure**:

```rust
pub struct Dataset {
    pub name: String,
    pub schema: DatasetSchema,
    pub columns: HashMap<String, ResourceReference>,  // ← References, not copies
}
```

**Characteristics**:

- **Column-based storage**: References to tensor data
- **Zero-copy**: Uses `ResourceReference` (views over existing data)
- **Memory efficient**: No data duplication
- **Status**: **Experimental** - not yet integrated with engine

## Why Both Exist?

**Migration Strategy**:

1. `dataset_legacy.rs` is the **current implementation**
2. `dataset/dataset.rs` is the **target architecture**
3. Migration is **in progress** but not complete

**Benefits of New Design**:

- Zero-copy transformations
- Column-oriented (better for analytics)
- Integrates with tensor system
- More memory efficient

**Challenges**:

- Requires DSL changes
- Engine integration work
- Backward compatibility

## Recommendation

**Keep both for now**:

- `dataset_legacy.rs`: Production use
- `dataset/dataset.rs`: Future development

**Future Work** (Phase 12+):

- Complete migration to zero-copy dataset
- Deprecate dataset_legacy
- Update DSL to use new design
