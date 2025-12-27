use super::{ComputeBackend, ScalarBackend};
use crate::core::tensor::{Tensor, TensorId};
use crate::engine::context::ExecutionContext;

#[derive(Debug, Default)]
pub struct SimdBackend {
    scalar: ScalarBackend, // Fallback for unsupported ops or small tensors
}

impl SimdBackend {
    pub fn new() -> Self {
        Self {
            scalar: ScalarBackend::new(),
        }
    }

    #[inline(always)]
    #[allow(dead_code)]
    fn is_aligned(ptr: *const f32, alignment: usize) -> bool {
        (ptr as usize) % alignment == 0
    }

    #[inline(always)]
    fn add_simd(&self, a: &[f32], b: &[f32], out: &mut [f32]) {
        let n = a.len();
        let mut i = 0;

        #[cfg(target_arch = "aarch64")]
        {
            use std::arch::aarch64::*;
            // NEON vld1q_f32 works for both aligned and unaligned
            while i + 4 <= n {
                unsafe {
                    let va = vld1q_f32(a.as_ptr().add(i));
                    let vb = vld1q_f32(b.as_ptr().add(i));
                    let vres = vaddq_f32(va, vb);
                    vst1q_f32(out.as_mut_ptr().add(i), vres);
                }
                i += 4;
            }
        }

        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::*;
            let a_ptr = a.as_ptr();
            let b_ptr = b.as_ptr();
            let out_ptr = out.as_mut_ptr();

            if Self::is_aligned(a_ptr, 16)
                && Self::is_aligned(b_ptr, 16)
                && Self::is_aligned(out_ptr, 16)
            {
                while i + 4 <= n {
                    unsafe {
                        let va = _mm_load_ps(a_ptr.add(i));
                        let vb = _mm_load_ps(b_ptr.add(i));
                        let vres = _mm_add_ps(va, vb);
                        _mm_store_ps(out_ptr.add(i), vres);
                    }
                    i += 4;
                }
            } else {
                while i + 4 <= n {
                    unsafe {
                        let va = _mm_loadu_ps(a_ptr.add(i));
                        let vb = _mm_loadu_ps(b_ptr.add(i));
                        let vres = _mm_add_ps(va, vb);
                        _mm_storeu_ps(out_ptr.add(i), vres);
                    }
                    i += 4;
                }
            }
        }

        // Remaining elements
        while i < n {
            out[i] = a[i] + b[i];
            i += 1;
        }
    }

    #[inline(always)]
    fn sub_simd(&self, a: &[f32], b: &[f32], out: &mut [f32]) {
        let n = a.len();
        let mut i = 0;

        #[cfg(target_arch = "aarch64")]
        {
            use std::arch::aarch64::*;
            while i + 4 <= n {
                unsafe {
                    let va = vld1q_f32(a.as_ptr().add(i));
                    let vb = vld1q_f32(b.as_ptr().add(i));
                    let vres = vsubq_f32(va, vb);
                    vst1q_f32(out.as_mut_ptr().add(i), vres);
                }
                i += 4;
            }
        }

        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::*;
            let a_ptr = a.as_ptr();
            let b_ptr = b.as_ptr();
            let out_ptr = out.as_mut_ptr();

            if Self::is_aligned(a_ptr, 16)
                && Self::is_aligned(b_ptr, 16)
                && Self::is_aligned(out_ptr, 16)
            {
                while i + 4 <= n {
                    unsafe {
                        let va = _mm_load_ps(a_ptr.add(i));
                        let vb = _mm_load_ps(b_ptr.add(i));
                        let vres = _mm_sub_ps(va, vb);
                        _mm_store_ps(out_ptr.add(i), vres);
                    }
                    i += 4;
                }
            } else {
                while i + 4 <= n {
                    unsafe {
                        let va = _mm_loadu_ps(a_ptr.add(i));
                        let vb = _mm_loadu_ps(b_ptr.add(i));
                        let vres = _mm_sub_ps(va, vb);
                        _mm_storeu_ps(out_ptr.add(i), vres);
                    }
                    i += 4;
                }
            }
        }

        while i < n {
            out[i] = a[i] - b[i];
            i += 1;
        }
    }

    #[inline(always)]
    fn mul_simd(&self, a: &[f32], b: &[f32], out: &mut [f32]) {
        let n = a.len();
        let mut i = 0;

        #[cfg(target_arch = "aarch64")]
        {
            use std::arch::aarch64::*;
            while i + 4 <= n {
                unsafe {
                    let va = vld1q_f32(a.as_ptr().add(i));
                    let vb = vld1q_f32(b.as_ptr().add(i));
                    let vres = vmulq_f32(va, vb);
                    vst1q_f32(out.as_mut_ptr().add(i), vres);
                }
                i += 4;
            }
        }

        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::*;
            let a_ptr = a.as_ptr();
            let b_ptr = b.as_ptr();
            let out_ptr = out.as_mut_ptr();

            if Self::is_aligned(a_ptr, 16)
                && Self::is_aligned(b_ptr, 16)
                && Self::is_aligned(out_ptr, 16)
            {
                while i + 4 <= n {
                    unsafe {
                        let va = _mm_load_ps(a_ptr.add(i));
                        let vb = _mm_load_ps(b_ptr.add(i));
                        let vres = _mm_mul_ps(va, vb);
                        _mm_store_ps(out_ptr.add(i), vres);
                    }
                    i += 4;
                }
            } else {
                while i + 4 <= n {
                    unsafe {
                        let va = _mm_loadu_ps(a_ptr.add(i));
                        let vb = _mm_loadu_ps(b_ptr.add(i));
                        let vres = _mm_mul_ps(va, vb);
                        _mm_storeu_ps(out_ptr.add(i), vres);
                    }
                    i += 4;
                }
            }
        }

        while i < n {
            out[i] = a[i] * b[i];
            i += 1;
        }
    }

    #[inline(always)]
    fn dot_simd(&self, a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let mut i = 0;
        #[allow(unused_assignments)]
        let mut sum = 0.0;

        #[cfg(target_arch = "aarch64")]
        {
            use std::arch::aarch64::*;
            unsafe {
                let mut vsum = vdupq_n_f32(0.0);
                while i + 4 <= n {
                    let va = vld1q_f32(a.as_ptr().add(i));
                    let vb = vld1q_f32(b.as_ptr().add(i));
                    vsum = vfmaq_f32(vsum, va, vb);
                    i += 4;
                }
                sum = vaddvq_f32(vsum);
            }
        }

        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::*;
            let a_ptr = a.as_ptr();
            let b_ptr = b.as_ptr();

            unsafe {
                let mut vsum = _mm_setzero_ps();
                if Self::is_aligned(a_ptr, 16) && Self::is_aligned(b_ptr, 16) {
                    while i + 4 <= n {
                        let va = _mm_load_ps(a_ptr.add(i));
                        let vb = _mm_load_ps(b_ptr.add(i));
                        vsum = _mm_add_ps(vsum, _mm_mul_ps(va, vb));
                        i += 4;
                    }
                } else {
                    while i + 4 <= n {
                        let va = _mm_loadu_ps(a_ptr.add(i));
                        let vb = _mm_loadu_ps(b_ptr.add(i));
                        vsum = _mm_add_ps(vsum, _mm_mul_ps(va, vb));
                        i += 4;
                    }
                }
                // Horizontal add
                let mut tmp = [0.0f32; 4];
                _mm_storeu_ps(tmp.as_mut_ptr(), vsum);
                sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];
            }
        }

        while i < n {
            sum += a[i] * b[i];
            i += 1;
        }
        sum
    }

    #[inline(always)]
    fn distance_simd(&self, a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let mut i = 0;
        #[allow(unused_assignments)]
        let mut sum = 0.0;

        #[cfg(target_arch = "aarch64")]
        {
            use std::arch::aarch64::*;
            unsafe {
                let mut vsum = vdupq_n_f32(0.0);
                while i + 4 <= n {
                    let va = vld1q_f32(a.as_ptr().add(i));
                    let vb = vld1q_f32(b.as_ptr().add(i));
                    let vdiff = vsubq_f32(va, vb);
                    vsum = vfmaq_f32(vsum, vdiff, vdiff);
                    i += 4;
                }
                sum = vaddvq_f32(vsum);
            }
        }

        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::*;
            let a_ptr = a.as_ptr();
            let b_ptr = b.as_ptr();

            unsafe {
                let mut vsum = _mm_setzero_ps();
                if Self::is_aligned(a_ptr, 16) && Self::is_aligned(b_ptr, 16) {
                    while i + 4 <= n {
                        let va = _mm_load_ps(a_ptr.add(i));
                        let vb = _mm_load_ps(b_ptr.add(i));
                        let vdiff = _mm_sub_ps(va, vb);
                        vsum = _mm_add_ps(vsum, _mm_mul_ps(vdiff, vdiff));
                        i += 4;
                    }
                } else {
                    while i + 4 <= n {
                        let va = _mm_loadu_ps(a_ptr.add(i));
                        let vb = _mm_loadu_ps(b_ptr.add(i));
                        let vdiff = _mm_sub_ps(va, vb);
                        vsum = _mm_add_ps(vsum, _mm_mul_ps(vdiff, vdiff));
                        i += 4;
                    }
                }
                let mut tmp = [0.0f32; 4];
                _mm_storeu_ps(tmp.as_mut_ptr(), vsum);
                sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];
            }
        }

        while i < n {
            let diff = a[i] - b[i];
            sum += diff * diff;
            i += 1;
        }
        sum.sqrt()
    }
}

impl ComputeBackend for SimdBackend {
    fn name(&self) -> &str {
        #[cfg(target_arch = "aarch64")]
        return "SIMD (NEON Optimized)";
        #[cfg(target_arch = "x86_64")]
        return "SIMD (SSE/AVX Optimized)";
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        return "SIMD (Portable Fallback)";
    }

    /// Allocate an output buffer for a tensor.
    fn alloc_output(&self, _ctx: &mut ExecutionContext, len: usize) -> Vec<f32> {
        vec![0.0; len]
    }

    fn add(
        &self,
        ctx: &mut ExecutionContext,
        a: &Tensor,
        b: &Tensor,
        new_id: TensorId,
    ) -> Result<Tensor, String> {
        if a.shape != b.shape {
            return Err("Shape mismatch".into());
        }
        let len = a.len();
        let mut data = self.alloc_output(ctx, len);
        self.add_simd(a.data_ref(), b.data_ref(), &mut data);
        Tensor::new(new_id, a.shape.clone(), data).map_err(|e| e.to_string())
    }

    fn sub(
        &self,
        ctx: &mut ExecutionContext,
        a: &Tensor,
        b: &Tensor,
        new_id: TensorId,
    ) -> Result<Tensor, String> {
        if a.shape != b.shape {
            return Err("Shape mismatch".into());
        }
        let len = a.len();
        let mut data = self.alloc_output(ctx, len);
        self.sub_simd(a.data_ref(), b.data_ref(), &mut data);
        Tensor::new(new_id, a.shape.clone(), data).map_err(|e| e.to_string())
    }

    fn multiply(
        &self,
        ctx: &mut ExecutionContext,
        a: &Tensor,
        b: &Tensor,
        new_id: TensorId,
    ) -> Result<Tensor, String> {
        if a.shape != b.shape {
            return Err("Shape mismatch".into());
        }
        let len = a.len();
        let mut data = self.alloc_output(ctx, len);
        self.mul_simd(a.data_ref(), b.data_ref(), &mut data);
        Tensor::new(new_id, a.shape.clone(), data).map_err(|e| e.to_string())
    }

    fn divide(
        &self,
        ctx: &mut ExecutionContext,
        a: &Tensor,
        b: &Tensor,
        new_id: TensorId,
    ) -> Result<Tensor, String> {
        self.scalar.divide(ctx, a, b, new_id)
    }

    fn matmul(
        &self,
        ctx: &mut ExecutionContext,
        a: &Tensor,
        b: &Tensor,
        new_id: TensorId,
    ) -> Result<Tensor, String> {
        self.scalar.matmul(ctx, a, b, new_id)
    }

    fn dot(&self, _ctx: &mut ExecutionContext, a: &Tensor, b: &Tensor) -> Result<f32, String> {
        if a.len() != b.len() {
            return Err("Length mismatch".into());
        }
        Ok(self.dot_simd(a.data_ref(), b.data_ref()))
    }

    fn cosine_similarity(
        &self,
        ctx: &mut ExecutionContext,
        a: &Tensor,
        b: &Tensor,
    ) -> Result<f32, String> {
        self.scalar.cosine_similarity(ctx, a, b)
    }

    fn distance(&self, _ctx: &mut ExecutionContext, a: &Tensor, b: &Tensor) -> Result<f32, String> {
        if a.len() != b.len() {
            return Err("Length mismatch".into());
        }
        Ok(self.distance_simd(a.data_ref(), b.data_ref()))
    }

    fn scale(
        &self,
        ctx: &mut ExecutionContext,
        a: &Tensor,
        factor: f32,
        new_id: TensorId,
    ) -> Result<Tensor, String> {
        self.scalar.scale(ctx, a, factor, new_id)
    }

    fn normalize(
        &self,
        ctx: &mut ExecutionContext,
        a: &Tensor,
        new_id: TensorId,
    ) -> Result<Tensor, String> {
        self.scalar.normalize(ctx, a, new_id)
    }

    fn transpose(
        &self,
        ctx: &mut ExecutionContext,
        a: &Tensor,
        new_id: TensorId,
    ) -> Result<Tensor, String> {
        self.scalar.transpose(ctx, a, new_id)
    }

    fn flatten(
        &self,
        ctx: &mut ExecutionContext,
        a: &Tensor,
        new_id: TensorId,
    ) -> Result<Tensor, String> {
        self.scalar.flatten(ctx, a, new_id)
    }

    fn reshape(
        &self,
        ctx: &mut ExecutionContext,
        a: &Tensor,
        new_shape: crate::core::tensor::Shape,
        new_id: TensorId,
    ) -> Result<Tensor, String> {
        self.scalar.reshape(ctx, a, new_shape, new_id)
    }

    fn stack(
        &self,
        ctx: &mut ExecutionContext,
        tensors: &[&Tensor],
        axis: usize,
        new_id: TensorId,
    ) -> Result<Tensor, String> {
        self.scalar.stack(ctx, tensors, axis, new_id)
    }
}
