//! Graph-level per-op evaluation for the Vulkan backend (M3-02-T24).
//!
//! This is the [`Backend::eval_op`](vokra_core::Backend::eval_op) surface the
//! graph evaluator ([`vokra_core::run_graph`]) drives on the GPU. Symmetric
//! with the Metal / CUDA `eval_metal_op` / `eval_cuda_op`.
//!
//! # Coverage today (M3-02-T14 partial / T24 / T26)
//!
//! - [`OpKind::Copy`] — identity element-wise copy; routes into
//!   [`crate::smoke_dispatch_copy_f32`] (hand-crafted `copy_f32` SPIR-V,
//!   2 SSBOs).
//! - [`OpKind::Add`] — element-wise addition; routes into
//!   [`crate::smoke_dispatch_add_f32`] (hand-crafted `add_f32` SPIR-V, 3
//!   SSBOs).
//! - Every other op → explicit [`VokraError::UnsupportedOp`], never a silent
//!   CPU fallback (FR-EX-08). Coverage widens as T14〜T22 `glslc`-produced
//!   SPIR-V shaders (GEMM / GEMV / softmax / softmax_causal / layer_norm /
//!   gelu / conv1d / mul / relu / sigmoid / tanh / transpose / gather) land.
//!
//! # Lock-step gate (M3-02-T35)
//!
//! [`crate::backend::VulkanBackend::supports`] and this dispatcher MUST cover
//! the same op set: an op supported here without `supports() == true` slips
//! past the [`run_graph`] coverage precheck; an op advertised by `supports()`
//! without an arm here surfaces as `UnsupportedOp` mid-dispatch. The
//! catch-all arm keeps the contract honest even when `eval_op` is called
//! directly (bypassing the engine's coverage precheck).

use vokra_core::{OpKind, Result, Tensor, VokraError};

use crate::context::{smoke_dispatch_add_f32_impl, smoke_dispatch_copy_f32_impl};

/// Dispatch a single op on the Vulkan backend.
///
/// Op coverage matches [`crate::backend::VulkanBackend::supports`] in
/// lock-step (M3-02-T35). See the module docs for the current op set.
///
/// The `Copy` / `Add` arms route into the hand-crafted SPIR-V dispatch
/// helpers in [`crate::context`]; on non-Vulkan hosts those return
/// [`VokraError::BackendUnavailable`], which bubbles up unchanged — the
/// FR-EX-08 contract explicitly forbids a silent CPU fallback here.
/// (This whole module is only compiled on Vulkan-target platforms via the
/// `mod eval;` cfg-gate in `lib.rs`, so a `BackendUnavailable` here means
/// the loader / ICD is missing, not that the feature is off.)
pub(crate) fn eval_vulkan_op(op: &OpKind, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
    match op {
        OpKind::Copy => eval_copy(inputs),
        OpKind::Add => eval_add(inputs),
        other => Err(VokraError::UnsupportedOp(format!(
            "vulkan backend has no graph kernel for {other:?} (no silent CPU fallback, \
             FR-EX-08). Foundation slice: SPIR-V kernels for GEMM / GEMV / softmax / \
             layer_norm / gelu / conv1d / … land with M3-02-T14 onwards."
        ))),
    }
}

/// `OpKind::Copy` → identity element-wise copy through the hand-crafted
/// `copy_f32` SPIR-V kernel. Single input, single output; output shape ==
/// input shape.
fn eval_copy(inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
    let x = take1(inputs, "Copy")?;
    let host = x.as_f32()?;
    let out = smoke_dispatch_copy_f32_impl(host)?;
    Ok(vec![Tensor::host_f32(x.shape.clone(), out)?])
}

/// `OpKind::Add` → element-wise sum through the hand-crafted `add_f32`
/// SPIR-V kernel. Two inputs of matching shape, single output of that same
/// shape.
fn eval_add(inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
    let (a, b) = take2(inputs, "Add")?;
    if a.shape != b.shape {
        return Err(VokraError::InvalidArgument(format!(
            "Add: input shapes must match; got {:?} and {:?}",
            a.shape, b.shape
        )));
    }
    let out = smoke_dispatch_add_f32_impl(a.as_f32()?, b.as_f32()?)?;
    Ok(vec![Tensor::host_f32(a.shape.clone(), out)?])
}

// ---- input-arity helpers (mirror the CPU / Metal backends' `eval.rs`) ----

fn take1<'t>(inputs: &[&'t Tensor], op_name: &str) -> Result<&'t Tensor> {
    if inputs.len() != 1 {
        return Err(VokraError::InvalidArgument(format!(
            "{op_name}: expected 1 input, got {}",
            inputs.len()
        )));
    }
    Ok(inputs[0])
}

fn take2<'t>(inputs: &[&'t Tensor], op_name: &str) -> Result<(&'t Tensor, &'t Tensor)> {
    if inputs.len() != 2 {
        return Err(VokraError::InvalidArgument(format!(
            "{op_name}: expected 2 inputs, got {}",
            inputs.len()
        )));
    }
    Ok((inputs[0], inputs[1]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uncovered_ops_are_explicit_unsupported() {
        // Ops the dispatcher doesn't cover surface as an explicit
        // `UnsupportedOp` — never a silent CPU fallback.
        let a = Tensor::zeros_f32(vec![2, 2]);
        let err = eval_vulkan_op(&OpKind::MatMul, &[&a, &a]).unwrap_err();
        assert!(
            matches!(err, VokraError::UnsupportedOp(_)),
            "MatMul must be UnsupportedOp today, got {err:?}"
        );
        let err = eval_vulkan_op(&OpKind::Softmax, &[&a]).unwrap_err();
        assert!(
            matches!(err, VokraError::UnsupportedOp(_)),
            "Softmax must be UnsupportedOp today, got {err:?}"
        );
        let err = eval_vulkan_op(&OpKind::Mul, &[&a, &a]).unwrap_err();
        assert!(
            matches!(err, VokraError::UnsupportedOp(_)),
            "Mul must be UnsupportedOp today, got {err:?}"
        );
    }

    #[test]
    fn copy_rejects_bad_arity() {
        // Arity check runs before any GPU dispatch, so it fires on all
        // hosts (Vulkan or not).
        let a = Tensor::zeros_f32(vec![2, 2]);
        let err = eval_vulkan_op(&OpKind::Copy, &[&a, &a]).unwrap_err();
        assert!(
            matches!(err, VokraError::InvalidArgument(_)),
            "Copy with 2 inputs must be InvalidArgument, got {err:?}"
        );
    }

    #[test]
    fn add_rejects_shape_mismatch() {
        // Shape check runs before any GPU dispatch, so it fires on all hosts.
        let a = Tensor::zeros_f32(vec![2, 3]);
        let b = Tensor::zeros_f32(vec![3, 2]);
        let err = eval_vulkan_op(&OpKind::Add, &[&a, &b]).unwrap_err();
        assert!(
            matches!(err, VokraError::InvalidArgument(_)),
            "Add with mismatched shapes must be InvalidArgument, got {err:?}"
        );
    }

    #[test]
    fn add_rejects_bad_arity() {
        let a = Tensor::zeros_f32(vec![2, 2]);
        let err = eval_vulkan_op(&OpKind::Add, &[&a]).unwrap_err();
        assert!(
            matches!(err, VokraError::InvalidArgument(_)),
            "Add with 1 input must be InvalidArgument, got {err:?}"
        );
    }
}
