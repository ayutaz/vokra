//! Graph-level per-op evaluation for the CPU backend (Phase 1).
//!
//! This is the [`Backend::eval_op`](vokra_core::Backend::eval_op) surface the
//! graph evaluator ([`vokra_core::run_graph`]) drives. Every op here routes
//! straight into the existing dispatched kernels in [`crate::kernels`] — there
//! is **no second kernel implementation**. The same `kernels::*` functions back
//! both this graph entry point and the imperative model call sites, so a graph
//! run and a direct kernel call are bit-identical.
//!
//! Output shapes are derived from the op semantics and the input shapes; the
//! engine ([`vokra_core::run_graph`]) is what checks them against the declared
//! [`TensorDesc`](vokra_core::TensorDesc)s, so this layer only allocates the
//! output buffer, runs the kernel and wraps the result in a [`Tensor`].

use vokra_core::{OpKind, Result, Tensor, VokraError};

use crate::kernels;

/// Evaluates one op on resolved CPU-resident inputs (see module docs).
///
/// Covers the same op set as [`CpuBackend::supports`](crate::CpuBackend), so
/// [`vokra_core::run_graph`]'s coverage precheck guarantees only those ops ever
/// reach here; a direct call with any other op is an explicit
/// [`VokraError::UnsupportedOp`] (FR-EX-08, never a silent fallback).
pub(crate) fn eval_cpu_op(op: &OpKind, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
    match op {
        OpKind::MatMul => eval_matmul(inputs),
        OpKind::Add => eval_binary(inputs, "Add", kernels::add_f32),
        OpKind::Mul => eval_binary(inputs, "Mul", kernels::mul_f32),
        OpKind::Softmax => eval_softmax(inputs),
        other => Err(VokraError::UnsupportedOp(format!(
            "cpu backend has no graph kernel for {other:?}"
        ))),
    }
}

/// `out[m,n] = sum_k a[m,k] * b[k,n]` via [`kernels::gemm_f32`] (no bias). Both
/// operands are rank-2 and their inner dimensions must agree.
fn eval_matmul(inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
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
    kernels::gemm_f32(m, n, k, a.as_f32()?, b.as_f32()?, None, &mut out)?;
    Ok(vec![Tensor::host_f32(vec![m, n], out)?])
}

/// Element-wise binary op preserving shape; both operands must be identically
/// shaped (no broadcast in the MVP).
fn eval_binary(
    inputs: &[&Tensor],
    name: &str,
    kernel: impl Fn(&[f32], &[f32], &mut [f32]) -> Result<()>,
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
    kernel(av, b.as_f32()?, &mut out)?;
    Ok(vec![Tensor::host_f32(a.shape.clone(), out)?])
}

/// Row-wise softmax over the innermost axis via [`kernels::softmax_f32`]; the
/// output keeps the input shape.
fn eval_softmax(inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
    let x = take1(inputs, "Softmax")?;
    let (rows, cols) = rows_cols(&x.shape)?;
    let xv = x.as_f32()?;
    let mut out = vec![0.0f32; xv.len()];
    kernels::softmax_f32(xv, &mut out, rows, cols)?;
    Ok(vec![Tensor::host_f32(x.shape.clone(), out)?])
}

// ---- input-arity / shape helpers ----

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
fn rows_cols(shape: &[usize]) -> Result<(usize, usize)> {
    match shape.split_last() {
        // `rest` are the outer axes; their product divides the total element
        // count, so it cannot overflow a `usize` the tensor already occupies.
        Some((&cols, rest)) => Ok((rest.iter().product(), cols)),
        None => Err(VokraError::InvalidArgument(
            "Softmax requires a tensor with at least one axis (got a scalar)".to_owned(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::ir::graph::StftAttrs;

    fn tensor(shape: &[usize], data: Vec<f32>) -> Tensor {
        Tensor::host_f32(shape.to_vec(), data).unwrap()
    }

    #[test]
    fn matmul_matches_direct_gemm() {
        // [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]].
        let a = tensor(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let b = tensor(&[2, 2], vec![5.0, 6.0, 7.0, 8.0]);
        let out = eval_cpu_op(&OpKind::MatMul, &[&a, &b]).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].shape, vec![2, 2]);

        let mut expected = vec![0.0f32; 4];
        kernels::gemm_f32(
            2,
            2,
            2,
            a.as_f32().unwrap(),
            b.as_f32().unwrap(),
            None,
            &mut expected,
        )
        .unwrap();
        assert_eq!(out[0].as_f32().unwrap(), expected.as_slice());
    }

    #[test]
    fn add_and_mul_are_elementwise() {
        let a = tensor(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let b = tensor(&[2, 2], vec![10.0, 20.0, 30.0, 40.0]);
        let sum = eval_cpu_op(&OpKind::Add, &[&a, &b]).unwrap();
        assert_eq!(sum[0].as_f32().unwrap(), &[11.0, 22.0, 33.0, 44.0]);
        let prod = eval_cpu_op(&OpKind::Mul, &[&a, &b]).unwrap();
        assert_eq!(prod[0].as_f32().unwrap(), &[10.0, 40.0, 90.0, 160.0]);
    }

    #[test]
    fn softmax_keeps_shape_and_matches_kernel() {
        let x = tensor(&[2, 3], vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0]);
        let out = eval_cpu_op(&OpKind::Softmax, &[&x]).unwrap();
        assert_eq!(out[0].shape, vec![2, 3]);
        let mut expected = vec![0.0f32; 6];
        kernels::softmax_f32(x.as_f32().unwrap(), &mut expected, 2, 3).unwrap();
        assert_eq!(out[0].as_f32().unwrap(), expected.as_slice());
    }

    #[test]
    fn matmul_rejects_inner_dim_mismatch() {
        let a = tensor(&[2, 3], vec![0.0; 6]);
        let b = tensor(&[2, 2], vec![0.0; 4]); // rhs rows (2) != lhs cols (3)
        let err = eval_cpu_op(&OpKind::MatMul, &[&a, &b]).unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn matmul_rejects_non_rank2() {
        let a = tensor(&[4], vec![0.0; 4]);
        let b = tensor(&[2, 2], vec![0.0; 4]);
        assert!(matches!(
            eval_cpu_op(&OpKind::MatMul, &[&a, &b]).unwrap_err(),
            VokraError::InvalidArgument(_)
        ));
    }

    #[test]
    fn binary_rejects_shape_mismatch_and_bad_arity() {
        let a = tensor(&[2, 2], vec![0.0; 4]);
        let b = tensor(&[4], vec![0.0; 4]); // same numel, different shape
        assert!(matches!(
            eval_cpu_op(&OpKind::Add, &[&a, &b]).unwrap_err(),
            VokraError::InvalidArgument(_)
        ));
        // wrong number of inputs.
        assert!(matches!(
            eval_cpu_op(&OpKind::Add, &[&a]).unwrap_err(),
            VokraError::InvalidArgument(_)
        ));
    }

    #[test]
    fn softmax_rejects_scalar_and_bad_arity() {
        let scalar = tensor(&[], vec![1.0]);
        assert!(matches!(
            eval_cpu_op(&OpKind::Softmax, &[&scalar]).unwrap_err(),
            VokraError::InvalidArgument(_)
        ));
        let x = tensor(&[3], vec![1.0, 2.0, 3.0]);
        assert!(matches!(
            eval_cpu_op(&OpKind::Softmax, &[&x, &x]).unwrap_err(),
            VokraError::InvalidArgument(_)
        ));
    }

    #[test]
    fn uncovered_op_is_explicit_unsupported() {
        // A front-end op has no CPU graph kernel here → explicit UnsupportedOp,
        // consistent with `CpuBackend::supports` returning false (FR-EX-08).
        let x = tensor(&[400], vec![0.0; 400]);
        let err = eval_cpu_op(&OpKind::Stft(StftAttrs::new(400, 160)), &[&x]).unwrap_err();
        assert!(matches!(err, VokraError::UnsupportedOp(_)));
    }
}
