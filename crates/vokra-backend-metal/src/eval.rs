//! Graph-level per-op evaluation for the Metal backend (Phase 2).
//!
//! This is the [`Backend::eval_op`](vokra_core::Backend::eval_op) surface the
//! graph evaluator ([`vokra_core::run_graph`]) drives on the GPU. `MatMul`
//! routes into the wrapped [`MetalContext::gemm_f32`] — the very same GPU kernel
//! the parity harness exercises (M2-01-T18) and the exact shape/semantics
//! contract of the CPU backend's `kernels::gemm_f32`. So a Metal graph run and a
//! CPU graph run of the same MatMul graph agree within the FP32 bound
//! (NFR-QL-01, `atol = 0.01`); there is **no second kernel** — `eval_op` and the
//! imperative `MetalContext::gemm_f32` share one GPU path.
//!
//! Only `MatMul` is covered in this slice (matching
//! [`MetalBackend::supports`](crate::MetalBackend)); every other op is an
//! explicit [`VokraError::UnsupportedOp`] — never a silent CPU fallback
//! (FR-EX-08). The engine's coverage precheck already rejects uncovered ops
//! before they reach here, but the catch-all keeps the contract honest even when
//! `eval_op` is called directly (keeps `supports()` and `eval_op()` in sync).

use vokra_core::{OpKind, Result, Tensor, VokraError};

use crate::context::MetalContext;

/// Evaluates one op on resolved host-resident inputs by dispatching to the GPU
/// (see module docs). Mirrors the CPU backend's `eval_cpu_op` op-for-op so the
/// two graph paths are differentially comparable.
pub(crate) fn eval_metal_op(
    ctx: &MetalContext,
    op: &OpKind,
    inputs: &[&Tensor],
) -> Result<Vec<Tensor>> {
    match op {
        OpKind::MatMul => eval_matmul(ctx, inputs),
        other => Err(VokraError::UnsupportedOp(format!(
            "metal backend has no graph kernel for {other:?} (no silent CPU fallback, FR-EX-08)"
        ))),
    }
}

/// `out[m,n] = Σ_k a[m,k] · b[k,n]` on the GPU via [`MetalContext::gemm_f32`]
/// (no bias). Both operands are rank-2 and their inner dimensions must agree —
/// the exact shape contract of the CPU backend's `MatMul`, computed identically
/// so the produced value matches the CPU graph within FP32 tolerance.
fn eval_matmul(ctx: &MetalContext, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
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

// ---- input-arity / shape helpers (mirror the CPU backend's `eval.rs`) ----

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

    /// The op dispatcher rejects every op it has no kernel for with an explicit
    /// `UnsupportedOp` — before touching the inputs — so a direct `eval_op` call
    /// (bypassing the engine's coverage precheck) can never silently fall back.
    /// Device-gated: needs a real `MetalContext`.
    #[test]
    fn uncovered_op_is_explicit_unsupported() {
        let Ok(ctx) = MetalContext::new() else {
            eprintln!("no Metal device; skipping eval dispatcher coverage test");
            return;
        };
        // `Add` is a real op the CPU backend covers, but this slice ships only
        // the Metal GEMM — it must surface as UnsupportedOp, not run on the CPU.
        let a = Tensor::zeros_f32(vec![2, 2]);
        let err = eval_metal_op(&ctx, &OpKind::Add, &[&a, &a]).unwrap_err();
        assert!(matches!(err, VokraError::UnsupportedOp(_)));
    }

    /// MatMul is genuinely wired: a small case computes and returns the declared
    /// `[m, n]` shape (correctness vs the CPU backend is `tests/graph_metal.rs`).
    #[test]
    fn matmul_is_wired_and_shapes_output() {
        let Ok(ctx) = MetalContext::new() else {
            eprintln!("no Metal device; skipping eval MatMul wiring test");
            return;
        };
        // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]].
        let a = Tensor::host_f32(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Tensor::host_f32(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]).unwrap();
        let out = eval_metal_op(&ctx, &OpKind::MatMul, &[&a, &b]).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].shape, vec![2, 2]);
        let got = out[0].as_f32().unwrap();
        let want = [19.0f32, 22.0, 43.0, 50.0];
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() <= 1e-3, "got {got:?} want {want:?}");
        }
    }

    /// Shape / arity errors are explicit `InvalidArgument`, not a GPU fault.
    #[test]
    fn matmul_rejects_bad_shapes() {
        let Ok(ctx) = MetalContext::new() else {
            eprintln!("no Metal device; skipping eval shape-validation test");
            return;
        };
        let a = Tensor::host_f32(vec![2, 3], vec![0.0; 6]).unwrap();
        let bad_inner = Tensor::host_f32(vec![2, 2], vec![0.0; 4]).unwrap(); // rows 2 != cols 3
        assert!(matches!(
            eval_metal_op(&ctx, &OpKind::MatMul, &[&a, &bad_inner]).unwrap_err(),
            VokraError::InvalidArgument(_)
        ));
        let rank1 = Tensor::host_f32(vec![4], vec![0.0; 4]).unwrap();
        assert!(matches!(
            eval_metal_op(&ctx, &OpKind::MatMul, &[&rank1, &a]).unwrap_err(),
            VokraError::InvalidArgument(_)
        ));
        // wrong number of inputs.
        assert!(matches!(
            eval_metal_op(&ctx, &OpKind::MatMul, &[&a]).unwrap_err(),
            VokraError::InvalidArgument(_)
        ));
    }
}
