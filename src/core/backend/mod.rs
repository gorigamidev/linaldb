use crate::core::tensor::{Tensor, TensorId};
use crate::engine::context::ExecutionContext;

pub trait ComputeBackend: std::fmt::Debug + Send + Sync {
    fn name(&self) -> &str;

    /// Allocate an output buffer for a tensor.
    fn alloc_output(&self, _ctx: &mut ExecutionContext, len: usize) -> Vec<f32> {
        vec![0.0; len]
    }

    // Binary operations
    fn add(
        &self,
        ctx: &mut ExecutionContext,
        a: &Tensor,
        b: &Tensor,
        new_id: TensorId,
    ) -> Result<Tensor, String>;
    fn sub(
        &self,
        ctx: &mut ExecutionContext,
        a: &Tensor,
        b: &Tensor,
        new_id: TensorId,
    ) -> Result<Tensor, String>;
    fn multiply(
        &self,
        ctx: &mut ExecutionContext,
        a: &Tensor,
        b: &Tensor,
        new_id: TensorId,
    ) -> Result<Tensor, String>;
    fn divide(
        &self,
        ctx: &mut ExecutionContext,
        a: &Tensor,
        b: &Tensor,
        new_id: TensorId,
    ) -> Result<Tensor, String>;

    // Matrix operations
    fn matmul(
        &self,
        ctx: &mut ExecutionContext,
        a: &Tensor,
        b: &Tensor,
        new_id: TensorId,
    ) -> Result<Tensor, String>;

    // Reductions / Metrics
    fn dot(&self, ctx: &mut ExecutionContext, a: &Tensor, b: &Tensor) -> Result<f32, String>;
    fn cosine_similarity(
        &self,
        ctx: &mut ExecutionContext,
        a: &Tensor,
        b: &Tensor,
    ) -> Result<f32, String>;
    fn distance(&self, ctx: &mut ExecutionContext, a: &Tensor, b: &Tensor) -> Result<f32, String>;

    // Unary operations
    fn scale(
        &self,
        ctx: &mut ExecutionContext,
        a: &Tensor,
        factor: f32,
        new_id: TensorId,
    ) -> Result<Tensor, String>;
    fn normalize(
        &self,
        ctx: &mut ExecutionContext,
        a: &Tensor,
        new_id: TensorId,
    ) -> Result<Tensor, String>;
    fn transpose(
        &self,
        ctx: &mut ExecutionContext,
        a: &Tensor,
        new_id: TensorId,
    ) -> Result<Tensor, String>;
    fn flatten(
        &self,
        ctx: &mut ExecutionContext,
        a: &Tensor,
        new_id: TensorId,
    ) -> Result<Tensor, String>;

    // Layout operations
    fn reshape(
        &self,
        ctx: &mut ExecutionContext,
        a: &Tensor,
        new_shape: crate::core::tensor::Shape,
        new_id: TensorId,
    ) -> Result<Tensor, String>;
    fn stack(
        &self,
        ctx: &mut ExecutionContext,
        tensors: &[&Tensor],
        axis: usize,
        new_id: TensorId,
    ) -> Result<Tensor, String>;
}

pub mod cpu;
pub mod scalar;
pub mod simd;

pub use cpu::CpuBackend;
pub use scalar::ScalarBackend;
pub use simd::SimdBackend;
