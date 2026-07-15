//! Graph-level per-op evaluation for the WebGPU backend (M4-01-T17;
//! wasm32 + feature `webgpu` only).
//!
//! This is the [`Backend::eval_op`](vokra_core::Backend::eval_op) surface the
//! graph evaluator ([`vokra_core::run_graph`]) drives. Symmetric with the
//! Vulkan `eval_vulkan_op` (M4-13-T09).
//!
//! # Coverage — the M4-13 Vulkan graph-arm target set
//!
//! | `OpKind`  | WGSL kernel                  |
//! |-----------|------------------------------|
//! | `Copy`    | `copy_f32`                   |
//! | `Add`     | `add_f32`                    |
//! | `MatMul`  | `gemm_f32`                   |
//! | `Mul`     | `elementwise` (op = mul)     |
//! | `Softmax` | `softmax`                    |
//!
//! Identical to the Vulkan arm `{Copy, Add, MatMul, Mul, Softmax}`; minus
//! `Copy` it is the CUDA arm's set (`Copy` is the Vulkan/WebGPU
//! runtime-verification extra — the CUDA arm does NOT have it). `Stft` (and
//! every front-end signal op) is covered by NO backend graph arm — the
//! honest all-backend gap the M4-13-T14 coverage table records (front-end
//! ops run in `vokra-ops`). Unlike the Vulkan arm there is **no blob
//! gating**: WGSL sources are embedded text, so all five arms are live from
//! this commit. Every other op → explicit
//! [`VokraError::UnsupportedOp`](vokra_core::VokraError::UnsupportedOp),
//! never a silent CPU fallback (FR-EX-08).
//!
//! `OpKind::Gemv` / `LayerNorm` / `Gelu` / `Conv1D` do not exist — those
//! kernels are imperative Whisper primitives with no graph `OpKind` variant
//! (surface 2 of the M4-13-T01 two-surface distinction), reached through the
//! `Compute` seam (`vokra-models`, M4-01-T16).

use vokra_core::{OpKind, Result, Tensor, VokraError};

use crate::backend::{WebGpuBackend, graph_op_backing_shader};
use crate::plan::ElementwiseOp;

/// Dispatch a single op on the WebGPU backend. Coverage matches
/// [`crate::backend::WebGpuBackend::supports`] in lock-step (both derive
/// from [`graph_op_backing_shader`]).
pub(crate) fn eval_webgpu_op(
    backend: &WebGpuBackend,
    op: &OpKind,
    inputs: &[&Tensor],
) -> Result<Vec<Tensor>> {
    match op {
        OpKind::Copy => eval_copy(backend, inputs),
        OpKind::Add => eval_add(backend, inputs),
        OpKind::MatMul => eval_matmul(backend, inputs),
        OpKind::Mul => eval_mul(backend, inputs),
        OpKind::Softmax => eval_softmax(backend, inputs),
        other => {
            debug_assert!(
                graph_op_backing_shader(other).is_none(),
                "op with a backing shader fell into the catch-all — supports()/eval_op drifted"
            );
            Err(VokraError::UnsupportedOp(format!(
                "webgpu backend has no graph kernel for {other:?} (no silent CPU fallback, \
                 FR-EX-08). Graph-arm coverage is {{Copy, Add, MatMul, Mul, Softmax}} — the \
                 CUDA arm plus Copy, identical to the Vulkan arm; front-end signal ops (Stft / \
                 MelFilterbank / …) run in vokra-ops."
            )))
        }
    }
}

fn eval_copy(backend: &WebGpuBackend, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
    let x = take1(inputs, "Copy")?;
    let host = x.as_f32()?;
    let mut out = vec![0.0f32; host.len()];
    backend.context().copy_f32(host, &mut out)?;
    Ok(vec![Tensor::host_f32(x.shape.clone(), out)?])
}

fn eval_add(backend: &WebGpuBackend, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
    let (a, b) = take2(inputs, "Add")?;
    if a.shape != b.shape {
        return Err(VokraError::InvalidArgument(format!(
            "Add: input shapes must match; got {:?} and {:?}",
            a.shape, b.shape
        )));
    }
    let av = a.as_f32()?;
    let bv = b.as_f32()?;
    let mut out = vec![0.0f32; av.len()];
    backend
        .context()
        .elementwise_f32(ElementwiseOp::Add, av, bv, &mut out)?;
    Ok(vec![Tensor::host_f32(a.shape.clone(), out)?])
}

fn eval_matmul(backend: &WebGpuBackend, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
    let (a, b) = take2(inputs, "MatMul")?;
    let (m, k) = as_2d(a, "MatMul lhs")?;
    let (k2, n) = as_2d(b, "MatMul rhs")?;
    if k != k2 {
        return Err(VokraError::InvalidArgument(format!(
            "MatMul: lhs is {m}x{k} but rhs is {k2}x{n} (inner dimensions differ)"
        )));
    }
    let mut out = vec![0.0f32; m * n];
    backend
        .context()
        .gemm_f32(m, n, k, a.as_f32()?, b.as_f32()?, None, &mut out)?;
    Ok(vec![Tensor::host_f32(vec![m, n], out)?])
}

fn eval_mul(backend: &WebGpuBackend, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
    let (a, b) = take2(inputs, "Mul")?;
    if a.shape != b.shape {
        return Err(VokraError::InvalidArgument(format!(
            "Mul: input shapes must match; got {:?} and {:?}",
            a.shape, b.shape
        )));
    }
    let av = a.as_f32()?;
    let bv = b.as_f32()?;
    let mut out = vec![0.0f32; av.len()];
    backend
        .context()
        .elementwise_f32(ElementwiseOp::Mul, av, bv, &mut out)?;
    Ok(vec![Tensor::host_f32(a.shape.clone(), out)?])
}

fn eval_softmax(backend: &WebGpuBackend, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
    let x = take1(inputs, "Softmax")?;
    let (rows, cols) = rows_cols(&x.shape)?;
    let mut out = vec![0.0f32; rows * cols];
    backend
        .context()
        .softmax_f32(x.as_f32()?, &mut out, rows, cols)?;
    Ok(vec![Tensor::host_f32(x.shape.clone(), out)?])
}

// ---- input-arity / shape helpers (mirror the CPU / Vulkan eval.rs) ----------

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
