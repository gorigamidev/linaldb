/// Operaciones binarias de alto nivel
#[derive(Debug, Clone)]
pub enum BinaryOp {
    /// a + b (element-wise)
    Add,
    /// a - b (element-wise)
    Subtract,
    /// a * b (element-wise)
    Multiply,
    /// a / b (element-wise)
    Divide,
    /// CORRELATE a WITH b  -> dot(a, b) (rank-1)
    Correlate,
    /// SIMILARITY a WITH b -> cosine_similarity(a, b) (rank-1)
    Similarity,
    /// DISTANCE a TO b -> distancia L2 (rank-1)
    Distance,
}

/// Operaciones unarias
#[derive(Debug, Clone)]
pub enum UnaryOp {
    /// SCALE a BY s
    Scale(f32),
    /// NORMALIZE a
    Normalize,
    /// TRANSPOSE a (matrix transpose)
    Transpose,
    /// FLATTEN a (flatten to 1D)
    Flatten,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorKind {
    /// Comportamiento por defecto (permite operaciones relajadas)
    Normal,
    /// Comportamiento estricto (shapes deben coincidir para element-wise)
    Strict,
}
