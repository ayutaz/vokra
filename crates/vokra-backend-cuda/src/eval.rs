//! Graph-level per-op evaluation for the CUDA backend (Phase 2 + M3-01-T06).
//!
//! This is the [`Backend::eval_op`](vokra_core::Backend::eval_op) surface the
//! graph evaluator ([`vokra_core::run_graph`]) drives on the GPU. Each op below
//! routes into the wrapped [`CudaContext`]'s kernel of the same
//! shape/semantics contract as the CPU backend's `kernels::*`, so a CUDA graph
//! run and a CPU graph run of the same graph agree within the FP32 bound
//! (NFR-QL-01, `atol = 0.01`).
//!
//! Coverage after M3-01-T06 matches the CPU backend's
//! [`CpuBackend::supports`](vokra_backend_cpu::CpuBackend) set:
//!
//! | OpKind      | CUDA kernel                            |
//! |-------------|----------------------------------------|
//! | `MatMul`    | [`CudaContext::gemm_f32`]              |
//! | `Add`       | [`CudaContext::add_f32`] (new M3-01)   |
//! | `Mul`       | [`CudaContext::mul_f32`] (new M3-01)   |
//! | `Softmax`   | [`CudaContext::softmax_f32`]           |
//!
//! Every other op is an explicit [`VokraError::UnsupportedOp`] — never a
//! silent CPU fall back (FR-EX-08). The catch-all keeps the contract honest
//! even when `eval_op` is called directly. Speech front-end ops
//! (`Stft` / `MelFilterbank` / …) and further primitives (`Gemv` /
//! `SoftmaxCausal` / `LayerNorm` / `Gelu` / `Conv1D` — which exist as
//! [`CudaContext`] kernels but not as `OpKind` graph variants) surface here as
//! `UnsupportedOp` too; extending them requires an `OpKind` extension in
//! `vokra-core::ir::graph` (out of the M3-01 scope, tracked in the ADR).

use vokra_core::{OpKind, Result, Tensor, VokraError};

use crate::context::CudaContext;

/// Evaluates one op on resolved host-resident inputs by dispatching to the GPU.
/// Mirrors the CPU/Metal backends' op-for-op so the graph paths are
/// differentially comparable.
pub(crate) fn eval_cuda_op(
    ctx: &CudaContext,
    op: &OpKind,
    inputs: &[&Tensor],
) -> Result<Vec<Tensor>> {
    match op {
        OpKind::MatMul => eval_matmul(ctx, inputs),
        OpKind::Add => eval_binary(ctx, inputs, "Add", CudaContext::add_f32),
        OpKind::Mul => eval_binary(ctx, inputs, "Mul", CudaContext::mul_f32),
        OpKind::Softmax => eval_softmax(ctx, inputs),
        other => Err(VokraError::UnsupportedOp(format!(
            "cuda backend has no graph kernel for {other:?} (no silent CPU fallback, FR-EX-08)"
        ))),
    }
}

/// `out[m,n] = Σ_k a[m,k] · b[k,n]` on the GPU via [`CudaContext::gemm_f32`]
/// (no bias). Both operands are rank-2 with agreeing inner dimensions — the
/// exact shape contract of the CPU backend's `MatMul`.
fn eval_matmul(ctx: &CudaContext, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
    let (a, b) = take2(inputs, "MatMul")?;
    let (m, k) = as_2d(a, "MatMul lhs")?;
    let (k2, n) = as_2d(b, "MatMul rhs")?;
    if k != k2 {
        return Err(VokraError::InvalidArgument(format!(
            "MatMul: lhs is {m}x{k} but rhs is {k2}x{n} (inner dimensions differ)"
        )));
    }
    let mn = m.checked_mul(n).ok_or_else(|| {
        VokraError::InvalidArgument("MatMul: output element count overflows usize".to_owned())
    })?;
    let mut out = vec![0.0f32; mn];
    ctx.gemm_f32(m, n, k, a.as_f32()?, b.as_f32()?, None, &mut out)?;
    Ok(vec![Tensor::host_f32(vec![m, n], out)?])
}

// ---- input-arity / shape helpers (mirror the CPU / Metal backends) ----

fn take1<'a>(inputs: &[&'a Tensor], op: &str) -> Result<&'a Tensor> {
    if inputs.len() != 1 {
        return Err(VokraError::InvalidArgument(format!(
            "{op} expects 1 input, got {}",
            inputs.len()
        )));
    }
    Ok(inputs[0])
}

fn take2<'a>(inputs: &[&'a Tensor], op: &str) -> Result<(&'a Tensor, &'a Tensor)> {
    if inputs.len() != 2 {
        return Err(VokraError::InvalidArgument(format!(
            "{op} expects 2 inputs, got {}",
            inputs.len()
        )));
    }
    Ok((inputs[0], inputs[1]))
}

fn as_2d(t: &Tensor, what: &str) -> Result<(usize, usize)> {
    match t.shape.as_slice() {
        [m, n] => Ok((*m, *n)),
        other => Err(VokraError::InvalidArgument(format!(
            "{what} must be rank-2, got shape {other:?}"
        ))),
    }
}

/// Splits a shape into `(rows, cols)` for a row-wise op: `cols` is the innermost
/// axis, `rows` the product of the rest. A scalar (empty shape) is rejected.
/// Mirrors the CPU backend's `rows_cols` (M3-01-T06 unified coverage).
fn rows_cols(shape: &[usize]) -> Result<(usize, usize)> {
    match shape.split_last() {
        Some((&cols, rest)) => Ok((rest.iter().product(), cols)),
        None => Err(VokraError::InvalidArgument(
            "Softmax requires a tensor with at least one axis (got a scalar)".to_owned(),
        )),
    }
}

/// Element-wise binary kernel signature: `kernel(ctx, a, b, out)` (M3-01-T06).
/// Factored out to keep `eval_binary`'s parameter list under
/// `clippy::type_complexity`.
type BinaryKernel = fn(&CudaContext, &[f32], &[f32], &mut [f32]) -> Result<()>;

/// Element-wise binary op preserving shape (M3-01-T06); both operands must be
/// identically shaped (no broadcast in the MVP, matching the CPU backend's
/// `eval_binary`). `kernel` is the `CudaContext` method that runs the op —
/// [`CudaContext::add_f32`] or [`CudaContext::mul_f32`].
fn eval_binary(
    ctx: &CudaContext,
    inputs: &[&Tensor],
    name: &str,
    kernel: BinaryKernel,
) -> Result<Vec<Tensor>> {
    let (a, b) = take2(inputs, name)?;
    if a.shape != b.shape {
        return Err(VokraError::InvalidArgument(format!(
            "{name}: operand shapes {:?} and {:?} differ (element-wise op, no broadcast)",
            a.shape, b.shape
        )));
    }
    let av = a.as_f32()?;
    let mut out = vec![0.0f32; av.len()];
    kernel(ctx, av, b.as_f32()?, &mut out)?;
    Ok(vec![Tensor::host_f32(a.shape.clone(), out)?])
}

/// Row-wise softmax over the innermost axis via [`CudaContext::softmax_f32`]
/// (M3-01-T06); the output keeps the input shape. Mirrors the CPU backend's
/// `eval_softmax` for differential parity.
fn eval_softmax(ctx: &CudaContext, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
    let x = take1(inputs, "Softmax")?;
    let (rows, cols) = rows_cols(&x.shape)?;
    let xv = x.as_f32()?;
    let mut out = vec![0.0f32; xv.len()];
    ctx.softmax_f32(xv, &mut out, rows, cols)?;
    Ok(vec![Tensor::host_f32(x.shape.clone(), out)?])
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::ir::graph::StftAttrs;

    /// The dispatcher rejects a genuinely-uncovered op with an explicit
    /// `UnsupportedOp`, never a silent CPU fall back (FR-EX-08, NFR-RL-06).
    /// Uses `Stft` — a speech front-end op the CUDA graph-executor has no
    /// kernel for (deferred to a future OpKind extension, ADR §3). Device-gated:
    /// needs a real `CudaContext`, so it runs only on the vast.ai GPU runner
    /// (skips here).
    #[test]
    fn uncovered_op_is_explicit_unsupported() {
        let Ok(ctx) = CudaContext::new() else {
            eprintln!("no CUDA device; skipping eval dispatcher test (run on vast.ai)");
            return;
        };
        let x = Tensor::zeros_f32(vec![400]);
        let err = eval_cuda_op(&ctx, &OpKind::Stft(StftAttrs::new(400, 160)), &[&x]).unwrap_err();
        assert!(matches!(err, VokraError::UnsupportedOp(_)));
    }

    /// MatMul is genuinely wired: a small case computes the expected value.
    /// Device-gated (vast.ai RTX 4090); skips on a CUDA-less host.
    #[test]
    fn matmul_is_wired_and_computes() {
        let Ok(ctx) = CudaContext::new() else {
            eprintln!("no CUDA device; skipping eval MatMul test (run on vast.ai)");
            return;
        };
        // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]].
        let a = Tensor::host_f32(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Tensor::host_f32(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]).unwrap();
        let out = eval_cuda_op(&ctx, &OpKind::MatMul, &[&a, &b]).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].shape, vec![2, 2]);
        let got = out[0].as_f32().unwrap();
        let want = [19.0f32, 22.0, 43.0, 50.0];
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() <= 1e-3, "got {got:?} want {want:?}");
        }
    }

    /// M3-01-T06: Add is element-wise and shape-preserving. Device-gated.
    #[test]
    fn add_is_wired_and_computes() {
        let Ok(ctx) = CudaContext::new() else {
            eprintln!("no CUDA device; skipping eval Add test (run on vast.ai)");
            return;
        };
        let a = Tensor::host_f32(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Tensor::host_f32(vec![2, 2], vec![10.0, 20.0, 30.0, 40.0]).unwrap();
        let out = eval_cuda_op(&ctx, &OpKind::Add, &[&a, &b]).unwrap();
        assert_eq!(out[0].as_f32().unwrap(), &[11.0, 22.0, 33.0, 44.0]);
        assert_eq!(out[0].shape, vec![2, 2]);
    }

    /// M3-01-T06: Mul is element-wise and shape-preserving. Device-gated.
    #[test]
    fn mul_is_wired_and_computes() {
        let Ok(ctx) = CudaContext::new() else {
            eprintln!("no CUDA device; skipping eval Mul test (run on vast.ai)");
            return;
        };
        let a = Tensor::host_f32(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Tensor::host_f32(vec![2, 2], vec![10.0, 20.0, 30.0, 40.0]).unwrap();
        let out = eval_cuda_op(&ctx, &OpKind::Mul, &[&a, &b]).unwrap();
        assert_eq!(out[0].as_f32().unwrap(), &[10.0, 40.0, 90.0, 160.0]);
    }

    /// M3-01-T06: Softmax rejects a scalar (mirrors the CPU backend's
    /// `rows_cols` guard). This is a pure shape check — reached before any
    /// device work — so it does not need a live CUDA device to exercise the
    /// error path, but it does need a context to call through the dispatcher.
    #[test]
    fn softmax_rejects_scalar_and_bad_arity() {
        let Ok(ctx) = CudaContext::new() else {
            eprintln!("no CUDA device; skipping eval Softmax shape test (run on vast.ai)");
            return;
        };
        let scalar = Tensor::host_f32(vec![], vec![1.0]).unwrap();
        assert!(matches!(
            eval_cuda_op(&ctx, &OpKind::Softmax, &[&scalar]).unwrap_err(),
            VokraError::InvalidArgument(_)
        ));
        let x = Tensor::host_f32(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
        assert!(matches!(
            eval_cuda_op(&ctx, &OpKind::Softmax, &[&x, &x]).unwrap_err(),
            VokraError::InvalidArgument(_)
        ));
    }

    /// M3-01-T06: binary ops reject shape mismatch and bad arity — same
    /// contract as `vokra-backend-cpu`'s `eval_binary`.
    #[test]
    fn binary_rejects_shape_mismatch_and_bad_arity() {
        let Ok(ctx) = CudaContext::new() else {
            eprintln!("no CUDA device; skipping eval Add shape test (run on vast.ai)");
            return;
        };
        let a = Tensor::host_f32(vec![2, 2], vec![0.0; 4]).unwrap();
        // same numel, different shape
        let b = Tensor::host_f32(vec![4], vec![0.0; 4]).unwrap();
        assert!(matches!(
            eval_cuda_op(&ctx, &OpKind::Add, &[&a, &b]).unwrap_err(),
            VokraError::InvalidArgument(_)
        ));
        // wrong arity
        assert!(matches!(
            eval_cuda_op(&ctx, &OpKind::Add, &[&a]).unwrap_err(),
            VokraError::InvalidArgument(_)
        ));
    }
}
