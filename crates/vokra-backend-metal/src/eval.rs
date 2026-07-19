//! Graph-level per-op evaluation for the Metal backend (Phase 2).
//!
//! This is the [`Backend::eval_op`](vokra_core::Backend::eval_op) surface the
//! graph evaluator ([`vokra_core::run_graph`]) drives on the GPU. Every covered
//! op routes into an *existing* Metal compute kernel — there is **no second
//! kernel** — so a Metal graph run and a CPU graph run of the same graph agree
//! within the FP32 bound (NFR-QL-01, `atol = 0.01`):
//!
//! - `MatMul` → [`MetalContext::gemm_f32`] (the `vokra_gemm_f32` kernel), the
//!   exact shape/semantics contract of the CPU `kernels::gemm_f32`.
//! - `Add` → [`MetalContext::residual_add_dev`] (the `vokra_add_assign_f32`
//!   kernel: a single FP32 `dst[i] + src[i]` per element, bit-identical to the
//!   CPU `kernels::add_f32`). The two operands are uploaded, summed on device,
//!   and read back — the same GPU add the imperative decode path uses.
//! - `Softmax` → [`MetalContext::softmax_f32`] (the `vokra_softmax_f32` kernel),
//!   row-wise over the innermost axis exactly like the CPU `kernels::softmax_f32`.
//! - `Mul` → [`MetalContext::mul_dev`] (the cc-27 `vokra_mul_f32` kernel: a
//!   single FP32 `dst[i] * src[i]` per element, measured bit-identical to the
//!   CPU `kernels::mul_f32` over normal-range operands).
//! - `Copy` → [`MetalContext::copy_dev`] (the cc-27 `vokra_copy_f32` kernel:
//!   `dst[i] = src[i]` as a real compute dispatch, not a host memcpy).
//!
//! `Mul` / `Copy` were previously an explicit `UnsupportedOp` here because no
//! `vokra_mul_f32` / compute-copy kernel existed — that was an honesty stance
//! about a missing kernel, not a design decision, and cc-27 closes it by
//! writing the two kernels. The Metal graph arm now covers the same op set as
//! the CUDA / Vulkan / WebGPU arms.
//!
//! Every op without a kernel still surfaces as an explicit
//! [`VokraError::UnsupportedOp`] — never a silent CPU fallback (FR-EX-08). The
//! engine's coverage precheck already rejects uncovered ops before they reach
//! here, but the catch-all keeps the contract honest even when `eval_op` is
//! called directly (keeps `supports()` and `eval_op()` in sync).

use vokra_core::{OpKind, Result, Tensor, VokraError};

use crate::context::MetalContext;

/// Evaluates one op on resolved host-resident inputs by dispatching to the GPU
/// (see module docs). Mirrors the CPU backend's `eval_cpu_op` op-for-op for the
/// covered ops so the two graph paths are differentially comparable.
pub(crate) fn eval_metal_op(
    ctx: &MetalContext,
    op: &OpKind,
    inputs: &[&Tensor],
) -> Result<Vec<Tensor>> {
    match op {
        OpKind::MatMul => eval_matmul(ctx, inputs),
        OpKind::Add => eval_add(ctx, inputs),
        OpKind::Mul => eval_mul(ctx, inputs),
        OpKind::Copy => eval_copy(ctx, inputs),
        OpKind::Softmax => eval_softmax(ctx, inputs),
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

/// Element-wise `out = a + b` on the GPU, preserving shape; both operands must
/// be identically shaped (no broadcast in the MVP, mirroring the CPU arm).
///
/// The sum is computed on device through [`MetalContext::residual_add_dev`] —
/// the `vokra_add_assign_f32` kernel — by uploading a fresh device copy of `a`
/// (the accumulator, so the host inputs are untouched), uploading `b`, adding
/// in place, and reading back. The kernel is a single FP32 add per element, so
/// the result matches the CPU backend's `kernels::add_f32` (bit-identical; see
/// the `residual_add_dev == host add` oracle in `parity_kernels_metal.rs`).
fn eval_add(ctx: &MetalContext, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
    let (a, b) = take2(inputs, "Add")?;
    if a.shape != b.shape {
        return Err(VokraError::InvalidArgument(format!(
            "Add: operand shapes {:?} and {:?} differ (element-wise op, no broadcast)",
            a.shape, b.shape
        )));
    }
    let av = a.as_f32()?;
    // `dst` is an independent device copy of `a`; `residual_add_dev` accumulates
    // `b` into it (`dst[i] += src[i]`), leaving the host `a`/`b` slices intact.
    let mut dst = ctx.upload(av)?;
    let src = ctx.upload(b.as_f32()?)?;
    ctx.residual_add_dev(&mut dst, &src)?;
    let mut out = vec![0.0f32; av.len()];
    ctx.download(&dst, &mut out)?;
    Ok(vec![Tensor::host_f32(a.shape.clone(), out)?])
}

/// Element-wise `out = a * b` on the GPU, preserving shape; both operands must
/// be identically shaped (no broadcast in the MVP, mirroring the CPU arm).
///
/// Structurally identical to [`eval_add`], only the device call differs: an
/// independent device copy of `a` is multiplied in place by `b` through
/// [`MetalContext::mul_dev`] (the cc-27 `vokra_mul_f32` kernel), leaving the
/// host slices intact. One FP32 multiply per element with no reduction order,
/// so the result matches the CPU backend's `kernels::mul_f32` (measured
/// bit-identical over normal-range operands — see
/// `tests/graph_metal.rs::mul_matches_cpu_backend`).
fn eval_mul(ctx: &MetalContext, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
    let (a, b) = take2(inputs, "Mul")?;
    if a.shape != b.shape {
        return Err(VokraError::InvalidArgument(format!(
            "Mul: operand shapes {:?} and {:?} differ (element-wise op, no broadcast)",
            a.shape, b.shape
        )));
    }
    let av = a.as_f32()?;
    let mut dst = ctx.upload(av)?;
    let src = ctx.upload(b.as_f32()?)?;
    ctx.mul_dev(&mut dst, &src)?;
    let mut out = vec![0.0f32; av.len()];
    ctx.download(&dst, &mut out)?;
    Ok(vec![Tensor::host_f32(a.shape.clone(), out)?])
}

/// Identity element-wise copy `out = x` on the GPU via
/// [`MetalContext::copy_dev`] (the cc-27 `vokra_copy_f32` kernel), preserving
/// shape. Mirrors the Vulkan / WebGPU `Copy` arms.
///
/// The copy runs as a real compute dispatch into a separate device buffer, so
/// the value is moved by the GPU rather than by the host round-trip alone; the
/// output is therefore bit-identical to the input by construction (an FP32 move
/// performs no arithmetic).
fn eval_copy(ctx: &MetalContext, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
    let x = take1(inputs, "Copy")?;
    let xv = x.as_f32()?;
    let src = ctx.upload(xv)?;
    let mut dst = ctx.alloc_dev(xv.len())?;
    ctx.copy_dev(&mut dst, &src)?;
    let mut out = vec![0.0f32; xv.len()];
    ctx.download(&dst, &mut out)?;
    Ok(vec![Tensor::host_f32(x.shape.clone(), out)?])
}

/// Row-wise softmax over the innermost axis on the GPU via
/// [`MetalContext::softmax_f32`] (the `vokra_softmax_f32` kernel); the output
/// keeps the input shape. Same `(rows, cols)` split as the CPU arm: `cols` is
/// the innermost axis and `rows` the product of the outer axes.
fn eval_softmax(ctx: &MetalContext, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
    let x = take1(inputs, "Softmax")?;
    let (rows, cols) = rows_cols(&x.shape)?;
    let xv = x.as_f32()?;
    let mut out = vec![0.0f32; xv.len()];
    ctx.softmax_f32(xv, &mut out, rows, cols)?;
    Ok(vec![Tensor::host_f32(x.shape.clone(), out)?])
}

// ---- input-arity / shape helpers (mirror the CPU backend's `eval.rs`) ----

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
/// axis, `rows` the product of the rest. A scalar (empty shape) is rejected
/// (mirrors the CPU backend's `rows_cols`).
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
        // `Stft` is a real audio-dialect op with no Metal graph kernel — it
        // must surface as UnsupportedOp, not run on the CPU.
        let a = Tensor::zeros_f32(vec![2, 2]);
        let err = eval_metal_op(
            &ctx,
            &OpKind::Stft(vokra_core::ir::graph::StftAttrs::new(400, 160)),
            &[&a],
        )
        .unwrap_err();
        assert!(matches!(err, VokraError::UnsupportedOp(_)));
    }

    /// `Mul` is genuinely wired (cc-27): a small case computes `a * b` on the
    /// GPU and returns the input shape. The single FP32 multiply per element
    /// carries one rounding, so the exact host product is reproduced.
    /// Differential correctness vs the CPU backend is `tests/graph_metal.rs`.
    #[test]
    fn mul_is_wired_and_shapes_output() {
        let Ok(ctx) = MetalContext::new() else {
            eprintln!("no Metal device; skipping eval Mul wiring test");
            return;
        };
        let a = Tensor::host_f32(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Tensor::host_f32(vec![2, 2], vec![10.0, 20.0, 30.0, 40.0]).unwrap();
        let out = eval_metal_op(&ctx, &OpKind::Mul, &[&a, &b]).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].shape, vec![2, 2]);
        assert_eq!(out[0].as_f32().unwrap(), &[10.0, 40.0, 90.0, 160.0]);
    }

    /// `Mul` shape / arity errors are explicit `InvalidArgument` (mirrors the
    /// `Add` arm's validation).
    #[test]
    fn mul_rejects_bad_shapes_and_arity() {
        let Ok(ctx) = MetalContext::new() else {
            eprintln!("no Metal device; skipping eval Mul validation test");
            return;
        };
        let a = Tensor::host_f32(vec![2, 2], vec![0.0; 4]).unwrap();
        let mismatched = Tensor::host_f32(vec![4], vec![0.0; 4]).unwrap(); // same numel, diff shape
        assert!(matches!(
            eval_metal_op(&ctx, &OpKind::Mul, &[&a, &mismatched]).unwrap_err(),
            VokraError::InvalidArgument(_)
        ));
        assert!(matches!(
            eval_metal_op(&ctx, &OpKind::Mul, &[&a]).unwrap_err(),
            VokraError::InvalidArgument(_)
        ));
    }

    /// `Copy` is genuinely wired (cc-27): the output is the input, element for
    /// element, with the shape preserved. An FP32 move performs no arithmetic,
    /// so equality here is exact by construction.
    #[test]
    fn copy_is_wired_and_is_the_identity() {
        let Ok(ctx) = MetalContext::new() else {
            eprintln!("no Metal device; skipping eval Copy wiring test");
            return;
        };
        let x = Tensor::host_f32(vec![2, 3], vec![1.5, -2.25, 3.0, 0.0, 1e-7, 4096.5]).unwrap();
        let out = eval_metal_op(&ctx, &OpKind::Copy, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].shape, vec![2, 3]);
        assert_eq!(out[0].as_f32().unwrap(), x.as_f32().unwrap());
    }

    /// `Copy` arity errors are explicit `InvalidArgument`.
    #[test]
    fn copy_rejects_bad_arity() {
        let Ok(ctx) = MetalContext::new() else {
            eprintln!("no Metal device; skipping eval Copy validation test");
            return;
        };
        let a = Tensor::host_f32(vec![2, 2], vec![0.0; 4]).unwrap();
        assert!(matches!(
            eval_metal_op(&ctx, &OpKind::Copy, &[&a, &a]).unwrap_err(),
            VokraError::InvalidArgument(_)
        ));
    }

    /// `Add` is genuinely wired: a small case computes `a + b` on the GPU and
    /// returns the input shape (differential correctness vs the CPU backend is
    /// `tests/graph_metal.rs`).
    #[test]
    fn add_is_wired_and_shapes_output() {
        let Ok(ctx) = MetalContext::new() else {
            eprintln!("no Metal device; skipping eval Add wiring test");
            return;
        };
        let a = Tensor::host_f32(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Tensor::host_f32(vec![2, 2], vec![10.0, 20.0, 30.0, 40.0]).unwrap();
        let out = eval_metal_op(&ctx, &OpKind::Add, &[&a, &b]).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].shape, vec![2, 2]);
        // Single FP32 add per element → bit-identical to the host sum.
        assert_eq!(out[0].as_f32().unwrap(), &[11.0, 22.0, 33.0, 44.0]);
    }

    /// `Add` shape / arity errors are explicit `InvalidArgument`, not a GPU
    /// fault (mirrors the CPU arm's validation).
    #[test]
    fn add_rejects_bad_shapes_and_arity() {
        let Ok(ctx) = MetalContext::new() else {
            eprintln!("no Metal device; skipping eval Add validation test");
            return;
        };
        let a = Tensor::host_f32(vec![2, 2], vec![0.0; 4]).unwrap();
        let mismatched = Tensor::host_f32(vec![4], vec![0.0; 4]).unwrap(); // same numel, diff shape
        assert!(matches!(
            eval_metal_op(&ctx, &OpKind::Add, &[&a, &mismatched]).unwrap_err(),
            VokraError::InvalidArgument(_)
        ));
        // wrong number of inputs.
        assert!(matches!(
            eval_metal_op(&ctx, &OpKind::Add, &[&a]).unwrap_err(),
            VokraError::InvalidArgument(_)
        ));
    }

    /// `Softmax` is genuinely wired: it keeps the input shape and normalises
    /// row-wise (each row sums to ~1). Numerical parity vs CPU is in
    /// `tests/graph_metal.rs`.
    #[test]
    fn softmax_is_wired_and_keeps_shape() {
        let Ok(ctx) = MetalContext::new() else {
            eprintln!("no Metal device; skipping eval Softmax wiring test");
            return;
        };
        let x = Tensor::host_f32(vec![2, 3], vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0]).unwrap();
        let out = eval_metal_op(&ctx, &OpKind::Softmax, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].shape, vec![2, 3]);
        let got = out[0].as_f32().unwrap();
        for r in 0..2 {
            let s: f32 = got[r * 3..r * 3 + 3].iter().sum();
            assert!((s - 1.0).abs() <= 1e-4, "row {r} softmax sums to {s}");
        }
    }

    /// `Softmax` rejects a scalar (no axis) and bad arity as explicit
    /// `InvalidArgument` (mirrors the CPU arm).
    #[test]
    fn softmax_rejects_scalar_and_bad_arity() {
        let Ok(ctx) = MetalContext::new() else {
            eprintln!("no Metal device; skipping eval Softmax validation test");
            return;
        };
        let scalar = Tensor::host_f32(vec![], vec![1.0]).unwrap();
        assert!(matches!(
            eval_metal_op(&ctx, &OpKind::Softmax, &[&scalar]).unwrap_err(),
            VokraError::InvalidArgument(_)
        ));
        let x = Tensor::host_f32(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
        assert!(matches!(
            eval_metal_op(&ctx, &OpKind::Softmax, &[&x, &x]).unwrap_err(),
            VokraError::InvalidArgument(_)
        ));
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
