use super::ComputeBackend;
use crate::core::tensor::{Tensor, TensorId};
use crate::engine::context::ExecutionContext;
use crate::engine::kernels;

#[derive(Debug, Default)]
pub struct ScalarBackend;

impl ScalarBackend {
    pub fn new() -> Self {
        Self
    }
}

impl ComputeBackend for ScalarBackend {
    fn name(&self) -> &str {
        "Scalar (Reference)"
    }

    fn add(
        &self,
        _ctx: &mut ExecutionContext,
        a: &Tensor,
        b: &Tensor,
        new_id: TensorId,
    ) -> Result<Tensor, String> {
        kernels::add_with_timestamp(a, b, new_id, _ctx.created_at)
    }

    fn sub(
        &self,
        _ctx: &mut ExecutionContext,
        a: &Tensor,
        b: &Tensor,
        new_id: TensorId,
    ) -> Result<Tensor, String> {
        kernels::sub_with_timestamp(a, b, new_id, _ctx.created_at)
    }

    fn multiply(
        &self,
        _ctx: &mut ExecutionContext,
        a: &Tensor,
        b: &Tensor,
        new_id: TensorId,
    ) -> Result<Tensor, String> {
        kernels::multiply_with_timestamp(a, b, new_id, _ctx.created_at)
    }

    fn divide(
        &self,
        _ctx: &mut ExecutionContext,
        a: &Tensor,
        b: &Tensor,
        new_id: TensorId,
    ) -> Result<Tensor, String> {
        kernels::divide_with_timestamp(a, b, new_id, _ctx.created_at)
    }

    fn matmul(
        &self,
        _ctx: &mut ExecutionContext,
        a: &Tensor,
        b: &Tensor,
        new_id: TensorId,
    ) -> Result<Tensor, String> {
        kernels::matmul_with_timestamp(a, b, new_id, _ctx.created_at)
    }

    fn dot(&self, _ctx: &mut ExecutionContext, a: &Tensor, b: &Tensor) -> Result<f32, String> {
        kernels::dot_1d(a, b)
    }

    fn cosine_similarity(
        &self,
        _ctx: &mut ExecutionContext,
        a: &Tensor,
        b: &Tensor,
    ) -> Result<f32, String> {
        kernels::cosine_similarity_1d(a, b)
    }

    fn distance(&self, _ctx: &mut ExecutionContext, a: &Tensor, b: &Tensor) -> Result<f32, String> {
        kernels::distance_1d(a, b)
    }

    fn scale(
        &self,
        _ctx: &mut ExecutionContext,
        a: &Tensor,
        factor: f32,
        new_id: TensorId,
    ) -> Result<Tensor, String> {
        kernels::scalar_mul_with_timestamp(a, factor, new_id, _ctx.created_at)
    }

    fn normalize(
        &self,
        _ctx: &mut ExecutionContext,
        a: &Tensor,
        new_id: TensorId,
    ) -> Result<Tensor, String> {
        kernels::normalize_1d_with_timestamp(a, new_id, _ctx.created_at)
    }

    fn transpose(
        &self,
        _ctx: &mut ExecutionContext,
        a: &Tensor,
        new_id: TensorId,
    ) -> Result<Tensor, String> {
        kernels::transpose_with_timestamp(a, new_id, _ctx.created_at)
    }

    fn flatten(
        &self,
        _ctx: &mut ExecutionContext,
        a: &Tensor,
        new_id: TensorId,
    ) -> Result<Tensor, String> {
        kernels::flatten_with_timestamp(a, new_id, _ctx.created_at)
    }

    fn reshape(
        &self,
        _ctx: &mut ExecutionContext,
        a: &Tensor,
        new_shape: crate::core::tensor::Shape,
        new_id: TensorId,
    ) -> Result<Tensor, String> {
        kernels::reshape_with_timestamp(a, new_shape, new_id, _ctx.created_at)
    }

    fn stack(
        &self,
        _ctx: &mut ExecutionContext,
        tensors: &[&Tensor],
        axis: usize,
        new_id: TensorId,
    ) -> Result<Tensor, String> {
        kernels::stack_with_timestamp(tensors, axis, new_id, _ctx.created_at)
    }
}
