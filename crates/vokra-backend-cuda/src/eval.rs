//! Graph-level per-op evaluation for the CUDA backend (Phase 2).
//!
//! This is the [`Backend::eval_op`](vokra_core::Backend::eval_op) surface the
//! graph evaluator ([`vokra_core::run_graph`]) drives on the GPU. `MatMul`
//! routes into the wrapped [`CudaContext::gemm_f32`] — the same GPU kernel the
//! parity harness exercises (M2-03-T19) and the exact shape/semantics contract
//! of the CPU backend's `kernels::gemm_f32`. So a CUDA graph run and a CPU graph
//! run of the same MatMul graph agree within the FP32 bound (NFR-QL-01,
//! `atol = 0.01`).
//!
//! Only `MatMul` is covered in this slice (matching
//! [`CudaBackend::supports`](crate::CudaBackend)); every other op is an explicit
//! [`VokraError::UnsupportedOp`] — never a silent CPU fall back (FR-EX-08). The
//! catch-all keeps the contract honest even when `eval_op` is called directly.

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

#[cfg(test)]
mod tests {
    use super::*;

    /// The dispatcher rejects an op it has no kernel for with an explicit
    /// `UnsupportedOp`, never a silent CPU fall back. Device-gated: needs a real
    /// `CudaContext`, so it runs only on the vast.ai GPU runner (skips here).
    #[test]
    fn uncovered_op_is_explicit_unsupported() {
        let Ok(ctx) = CudaContext::new() else {
            eprintln!("no CUDA device; skipping eval dispatcher test (run on vast.ai)");
            return;
        };
        let a = Tensor::zeros_f32(vec![2, 2]);
        let err = eval_cuda_op(&ctx, &OpKind::Add, &[&a, &a]).unwrap_err();
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
}
