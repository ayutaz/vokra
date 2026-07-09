//! Graph-level per-op evaluation for the Vulkan backend (M3-02-T24).
//!
//! This is the [`Backend::eval_op`](vokra_core::Backend::eval_op) surface the
//! graph evaluator ([`vokra_core::run_graph`]) drives on the GPU. Symmetric
//! with the Metal / CUDA `eval_metal_op` / `eval_cuda_op`.
//!
//! Foundation slice: NO SPIR-V kernel is wired yet, so **every op is an
//! explicit [`VokraError::UnsupportedOp`]** — never a silent CPU fallback
//! (FR-EX-08). As T14〜T22 SPIR-V shaders and T25 buffer helpers land, this
//! dispatcher gains one arm per op (Gemm / Gemv / Softmax / SoftmaxCausal /
//! LayerNorm / Gelu / Conv1D / Add / Mul / Relu / Sigmoid / Tanh / Transpose /
//! Gather), and `VulkanBackend::supports` (backend.rs) flips to `true` for
//! the same set. The op coverage of `supports()` and `eval_op()` MUST stay in
//! lock-step (M3-02-T35 gate) — the catch-all keeps the contract honest even
//! when `eval_op` is called directly, bypassing the engine's coverage
//! precheck.

use vokra_core::{OpKind, Result, Tensor, VokraError};

/// Dispatch a single op on the Vulkan backend.
///
/// Foundation slice: the match has no covering arm; every op falls through
/// to the explicit-error catch-all. Later tickets extend this in place.
pub(crate) fn eval_vulkan_op(op: &OpKind, _inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
    // Kept as a single arm so T24〜T29 can extend it in place without
    // reshaping the function. Clippy would prefer a `let other = op;` today
    // (single-arm match), but that would force a churn-heavy diff the moment
    // the first real kernel arm lands. Allow the single-arm match to keep
    // this a pure diff-shape target.
    #[allow(clippy::match_single_binding)]
    match op {
        // Every arm added by T24〜T29 will look like:
        //
        //   OpKind::MatMul => eval_matmul(_inputs),
        //   OpKind::Softmax => eval_softmax(_inputs),
        //   ...
        //
        // and delegate to a helper in this file. Kept catch-all-only today so
        // there is a single source of truth for the "no kernel wired yet"
        // state.
        other => Err(VokraError::UnsupportedOp(format!(
            "vulkan backend has no graph kernel for {other:?} (no silent CPU fallback, \
             FR-EX-08). Foundation slice: SPIR-V kernels land with M3-02-T14 onwards."
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn any_op_is_explicit_unsupported() {
        // We do not need a real Vulkan device to test the dispatcher: it never
        // touches a GPU in the foundation slice; every op is an explicit
        // `UnsupportedOp` immediately.
        let a = Tensor::zeros_f32(vec![2, 2]);
        let err = eval_vulkan_op(&OpKind::MatMul, &[&a, &a]).unwrap_err();
        assert!(matches!(err, VokraError::UnsupportedOp(_)));
        let err2 = eval_vulkan_op(&OpKind::Add, &[&a, &a]).unwrap_err();
        assert!(matches!(err2, VokraError::UnsupportedOp(_)));
        let err3 = eval_vulkan_op(&OpKind::Softmax, &[&a]).unwrap_err();
        assert!(matches!(err3, VokraError::UnsupportedOp(_)));
    }
}
