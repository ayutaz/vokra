//! Graph-level per-op evaluation for the Vulkan backend (M3-02-T24 +
//! M4-13-T09).
//!
//! This is the [`Backend::eval_op`](vokra_core::Backend::eval_op) surface the
//! graph evaluator ([`vokra_core::run_graph`]) drives on the GPU. Symmetric
//! with the Metal / CUDA `eval_metal_op` / `eval_cuda_op`.
//!
//! # Coverage (M4-13-T09 — CUDA-arm parity)
//!
//! | `OpKind`  | Vulkan kernel                                             |
//! |-----------|-----------------------------------------------------------|
//! | `Copy`    | hand-crafted `copy_f32` (always available)                |
//! | `Add`     | hand-crafted `add_f32` (always available)                 |
//! | `MatMul`  | `gemm_subgroup` / `gemm_coopmat` (probe-selected)         |
//! | `Mul`     | `elementwise` (OP = mul specialization)                   |
//! | `Softmax` | `softmax`                                                 |
//!
//! Minus `Copy`, this is exactly the CUDA graph arm's set
//! (`crates/vokra-backend-cuda/src/backend.rs` — `MatMul | Add | Mul |
//! Softmax`); `Stft` is covered by NEITHER arm (the honest gap in the
//! M4-13-T14 coverage table — front-end signal ops run in `vokra-ops`).
//!
//! The three glslc-backed arms are **blob-gated** (placeholder-then-swap):
//! until the owner commits the `.spv` (M4-13-T16) they surface the explicit
//! [`VokraError::UnsupportedOp`] from `spirv::require_blob`, and
//! [`crate::backend::VulkanBackend::supports`] reports `false` for them —
//! the M3-02-T35 lock-step invariant holds *by construction* because both
//! sides derive from [`crate::backend::graph_op_backing_shader`] +
//! [`crate::spirv::has_blob`]. Every other op → explicit
//! [`VokraError::UnsupportedOp`], never a silent CPU fallback (FR-EX-08).
//!
//! # Arbitrary-length `Copy` / `Add` (M4-13-T09 bug fix)
//!
//! The hand-crafted smoke kernels have no bounds check (their bytecode is
//! deliberately minimal) and **assert** the element count is a multiple of
//! their `LOCAL_SIZE_X` (64). The M3-02 arms passed graph tensors straight
//! through, so a `[2, 2]` tensor (4 elements) would have PANICKED on a live
//! Vulkan host — a contract violation (public APIs must return `Result`,
//! never panic across the boundary). The arms now zero-pad to the next
//! multiple of 64 before dispatch and truncate the readback; the pad lanes
//! compute `0.0` (copy) / `0.0 + 0.0` (add) into memory that is dropped, so
//! the live region is bit-identical to the unpadded dispatch.

use vokra_core::{OpKind, Result, Tensor, VokraError};

use crate::backend::{GemmPipelinePreference, VulkanBackend, graph_op_backing_shader};
use crate::context::{smoke_dispatch_add_f32_impl, smoke_dispatch_copy_f32_impl};
use crate::plan::ElementwiseOp;

/// Dispatch a single op on the Vulkan backend.
///
/// Op coverage matches [`crate::backend::VulkanBackend::supports`] in
/// lock-step (M3-02-T35): both derive from
/// [`graph_op_backing_shader`] + blob availability. See the module docs for
/// the coverage table.
pub(crate) fn eval_vulkan_op(
    backend: &VulkanBackend,
    op: &OpKind,
    inputs: &[&Tensor],
) -> Result<Vec<Tensor>> {
    match op {
        OpKind::Copy => eval_copy(inputs),
        OpKind::Add => eval_add(inputs),
        OpKind::MatMul => eval_matmul(backend, inputs),
        OpKind::Mul => eval_mul(backend, inputs),
        OpKind::Softmax => eval_softmax(backend, inputs),
        other => {
            debug_assert!(
                graph_op_backing_shader(other, crate::backend::GemmPipelineVariant::Subgroup)
                    .is_none(),
                "op with a backing shader fell into the catch-all — supports()/eval_op drifted"
            );
            Err(VokraError::UnsupportedOp(format!(
                "vulkan backend has no graph kernel for {other:?} (no silent CPU fallback, \
                 FR-EX-08). Graph-arm coverage is {{Copy, Add, MatMul, Mul, Softmax}} — the \
                 CUDA arm plus Copy; front-end signal ops (Stft / MelFilterbank / …) run in \
                 vokra-ops."
            )))
        }
    }
}

/// Zero-pads `v` to the next multiple of `multiple` (no-op when already
/// aligned). The hand-crafted smoke kernels require it; see the module docs.
fn pad_to_multiple(v: &[f32], multiple: usize) -> Vec<f32> {
    let mut padded = v.to_vec();
    padded.resize(v.len().div_ceil(multiple) * multiple, 0.0);
    padded
}

/// `OpKind::Copy` → identity element-wise copy through the hand-crafted
/// `copy_f32` SPIR-V kernel. Single input, single output; output shape ==
/// input shape. Arbitrary lengths are zero-padded to the kernel's
/// `LOCAL_SIZE_X` and truncated on readback (bit-identical live region).
fn eval_copy(inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
    let x = take1(inputs, "Copy")?;
    let host = x.as_f32()?;
    let local = crate::spirv::handcrafted_copy_f32::LOCAL_SIZE_X as usize;
    let mut out = if host.len() % local == 0 {
        smoke_dispatch_copy_f32_impl(host)?
    } else {
        smoke_dispatch_copy_f32_impl(&pad_to_multiple(host, local))?
    };
    out.truncate(host.len());
    Ok(vec![Tensor::host_f32(x.shape.clone(), out)?])
}

/// `OpKind::Add` → element-wise sum through the hand-crafted `add_f32`
/// SPIR-V kernel. Two inputs of matching shape, single output of that same
/// shape. Arbitrary lengths are zero-padded (pad lanes compute `0 + 0` into
/// dropped memory) and truncated on readback.
fn eval_add(inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
    let (a, b) = take2(inputs, "Add")?;
    if a.shape != b.shape {
        return Err(VokraError::InvalidArgument(format!(
            "Add: input shapes must match; got {:?} and {:?}",
            a.shape, b.shape
        )));
    }
    let av = a.as_f32()?;
    let bv = b.as_f32()?;
    let local = crate::spirv::handcrafted_add_f32::LOCAL_SIZE_X as usize;
    let mut out = if av.len() % local == 0 {
        smoke_dispatch_add_f32_impl(av, bv)?
    } else {
        smoke_dispatch_add_f32_impl(&pad_to_multiple(av, local), &pad_to_multiple(bv, local))?
    };
    out.truncate(av.len());
    Ok(vec![Tensor::host_f32(a.shape.clone(), out)?])
}

/// `OpKind::MatMul` → `out[m,n] = Σ_k a[m,k]·b[k,n]` through the
/// probe-selected GEMM pipeline (M4-13-T09 / T03). Both operands are rank-2
/// with agreeing inner dimensions — the exact shape contract of the CPU /
/// CUDA backends' `MatMul` arms.
fn eval_matmul(backend: &VulkanBackend, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
    let (a, b) = take2(inputs, "MatMul")?;
    let (m, k) = as_2d(a, "MatMul lhs")?;
    let (k2, n) = as_2d(b, "MatMul rhs")?;
    if k != k2 {
        return Err(VokraError::InvalidArgument(format!(
            "MatMul: lhs is {m}x{k} but rhs is {k2}x{n} (inner dimensions differ)"
        )));
    }
    let out = backend.gemm_f32(
        GemmPipelinePreference::default(),
        m,
        n,
        k,
        a.as_f32()?,
        b.as_f32()?,
    )?;
    Ok(vec![Tensor::host_f32(vec![m, n], out)?])
}

/// `OpKind::Mul` → element-wise product through the `elementwise` kernel
/// (OP = mul specialization, M4-13-T09 / T07). Same shape contract as `Add`.
fn eval_mul(backend: &VulkanBackend, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
    let (a, b) = take2(inputs, "Mul")?;
    if a.shape != b.shape {
        return Err(VokraError::InvalidArgument(format!(
            "Mul: input shapes must match; got {:?} and {:?}",
            a.shape, b.shape
        )));
    }
    let out = backend.elementwise_f32(ElementwiseOp::Mul, a.as_f32()?, b.as_f32()?)?;
    Ok(vec![Tensor::host_f32(a.shape.clone(), out)?])
}

/// `OpKind::Softmax` → numerically-stable softmax over the innermost axis
/// (M4-13-T09 / T05): `rows = product of leading axes`, `cols = last axis`
/// — the exact shape contract of the CPU / CUDA backends' `Softmax` arms.
fn eval_softmax(backend: &VulkanBackend, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
    let x = take1(inputs, "Softmax")?;
    let (rows, cols) = rows_cols(&x.shape)?;
    let out = backend.softmax_f32(rows, cols, x.as_f32()?)?;
    Ok(vec![Tensor::host_f32(x.shape.clone(), out)?])
}

// ---- input-arity / shape helpers (mirror the CPU / Metal / CUDA eval.rs) ----

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

fn as_2d(t: &Tensor, what: &str) -> Result<(usize, usize)> {
    match t.shape.as_slice() {
        [rows, cols] => Ok((*rows, *cols)),
        other => Err(VokraError::InvalidArgument(format!(
            "{what}: expected a rank-2 tensor, got shape {other:?}"
        ))),
    }
}

fn rows_cols(shape: &[usize]) -> Result<(usize, usize)> {
    match shape.split_last() {
        Some((&cols, rest)) => Ok((rest.iter().product(), cols)),
        None => Err(VokraError::InvalidArgument(
            "Softmax requires a tensor with at least one axis (got a scalar)".to_owned(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pad_to_multiple_is_zero_fill_and_noop_when_aligned() {
        let v = [1.0f32, 2.0, 3.0];
        let padded = pad_to_multiple(&v, 4);
        assert_eq!(padded, vec![1.0, 2.0, 3.0, 0.0]);
        let aligned = pad_to_multiple(&padded, 4);
        assert_eq!(aligned.len(), 4, "aligned input stays untouched");
        // Multiple workgroups' worth.
        assert_eq!(pad_to_multiple(&[1.0; 65], 64).len(), 128);
    }

    #[test]
    fn shape_helpers_mirror_the_cuda_arm_contracts() {
        // as_2d accepts rank-2 only.
        let t = Tensor::zeros_f32(vec![2, 3]);
        assert_eq!(as_2d(&t, "t").unwrap(), (2, 3));
        let t3 = Tensor::zeros_f32(vec![2, 3, 4]);
        assert!(matches!(
            as_2d(&t3, "t"),
            Err(VokraError::InvalidArgument(_))
        ));
        // rows_cols folds leading axes and rejects scalars.
        assert_eq!(rows_cols(&[2, 3, 4]).unwrap(), (6, 4));
        assert_eq!(rows_cols(&[5]).unwrap(), (1, 5));
        assert!(matches!(
            rows_cols(&[]),
            Err(VokraError::InvalidArgument(_))
        ));
    }
}
