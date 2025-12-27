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
        kernels::add(a, b, new_id)
    }

    fn sub(
        &self,
        _ctx: &mut ExecutionContext,
        a: &Tensor,
        b: &Tensor,
        new_id: TensorId,
    ) -> Result<Tensor, String> {
        kernels::sub(a, b, new_id)
    }

    fn multiply(
        &self,
        _ctx: &mut ExecutionContext,
        a: &Tensor,
        b: &Tensor,
        new_id: TensorId,
    ) -> Result<Tensor, String> {
        kernels::multiply(a, b, new_id)
    }

    fn divide(
        &self,
        _ctx: &mut ExecutionContext,
        a: &Tensor,
        b: &Tensor,
        new_id: TensorId,
    ) -> Result<Tensor, String> {
        kernels::divide(a, b, new_id)
    }

    fn matmul(
        &self,
        _ctx: &mut ExecutionContext,
        a: &Tensor,
        b: &Tensor,
        new_id: TensorId,
    ) -> Result<Tensor, String> {
        kernels::matmul(a, b, new_id)
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
        kernels::scalar_mul(a, factor, new_id)
    }

    fn normalize(
        &self,
        _ctx: &mut ExecutionContext,
        a: &Tensor,
        new_id: TensorId,
    ) -> Result<Tensor, String> {
        kernels::normalize_1d(a, new_id)
    }

    fn transpose(
        &self,
        _ctx: &mut ExecutionContext,
        a: &Tensor,
        new_id: TensorId,
    ) -> Result<Tensor, String> {
        kernels::transpose(a, new_id)
    }

    fn flatten(
        &self,
        _ctx: &mut ExecutionContext,
        a: &Tensor,
        new_id: TensorId,
    ) -> Result<Tensor, String> {
        kernels::flatten(a, new_id)
    }

    fn reshape(
        &self,
        _ctx: &mut ExecutionContext,
        a: &Tensor,
        new_shape: crate::core::tensor::Shape,
        new_id: TensorId,
    ) -> Result<Tensor, String> {
        kernels::reshape(a, new_shape, new_id)
    }

    fn stack(
        &self,
        _ctx: &mut ExecutionContext,
        tensors: &[&Tensor],
        axis: usize,
        new_id: TensorId,
    ) -> Result<Tensor, String> {
        kernels::stack(tensors, axis, new_id)
    }
}
