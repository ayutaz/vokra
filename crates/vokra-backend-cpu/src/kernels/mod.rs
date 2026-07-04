//! CPU compute kernels and their safe public wrappers (M0-08-T02, T05..T16).
//!
//! # Confirmed spike kernel set (M0-08-T02)
//!
//! Back-derived from the ops Whisper base (M0-06) and Silero VAD (M0-05)
//! need. dtype is **f32 only** in the spike (aligned with the FP32 parity
//! bound NFR-QL-01 atol = 0.01; f16 / K-quant kernels are FR-QT-01 = v0.1
//! MVP and later). Threading is intentionally **not** introduced in M0 (the
//! rayon / OpenMP-alternative decision is deferred — NFR-LC-03).
//!
//! | kernel | SIMD? | rationale |
//! |--------|-------|-----------|
//! | [`gemm_f32`] (bias = linear) | yes | dominant Whisper attention / FFN cost |
//! | [`gemv_f32`] (bias = per-row) | yes | tied logits head `token_emb[v,d] @ h[d]` (the `gemm` `n=1` scalar-tail case, M1) |
//! | [`add_f32`] / [`mul_f32`] | yes | residual add, gating |
//! | [`relu_f32`] | yes | Silero VAD conv stack |
//! | [`sigmoid_f32`] | scalar-backed; SIMD under `simd-transcendental` | VAD output / LSTM gate; exp-bound (`vexp`, M1-05-EXP) |
//! | [`tanh_f32`] | scalar-backed; SIMD under `simd-transcendental` | LSTM cell; exp-bound (`vexp`, M1-05-EXP) |
//! | [`gelu_f32`] | scalar-backed; SIMD under `simd-transcendental` | Whisper MLP (exact/erf form); exp-bound (`vexp`, M1-05-EXP) |
//! | [`softmax_f32`] | yes (exp scalar; SIMD under `simd-transcendental`) | Whisper attention |
//! | [`layer_norm_f32`] | yes | Whisper pre-norm blocks |
//! | [`conv1d_f32`] | via GEMM | Whisper encoder stem; im2col + [`gemm_f32`] |
//!
//! **Deliberately not SIMD kernels here** (memory-bound / structural, left to
//! scalar or the model layer's `vokra-ops` reference — M0-06-T03): embedding
//! lookup, transpose, reshape.
//!
//! `conv1d_f32` has no dedicated SIMD kernel: it lowers to im2col + the
//! dispatched [`gemm_f32`], so it inherits AVX2 / NEON automatically
//! (M0-08-T08/T12/T15).
//!
//! # Boundary with `vokra-ops`
//!
//! This crate owns the **dispatch-target compute kernels** (the functions
//! below). `vokra-ops` owns the **operator definitions** (front-end / speech
//! ops and their attributes) and any scalar op *reference* used by the parity
//! harness. New "missing op" requests raised by M0-06-T02 are folded in by
//! appending to the table above and adding the kernel, up to (but not after)
//! WP completion (M0-08-T19), to avoid re-opening a finished WP.
//!
//! # Function boundary for M0-06
//!
//! M0-06's encoder / decoder call these safe wrappers directly:
//! [`gemm_f32`], [`add_f32`], [`mul_f32`], [`relu_f32`], [`sigmoid_f32`],
//! [`tanh_f32`], [`gelu_f32`], [`softmax_f32`], [`layer_norm_f32`],
//! [`conv1d_f32`], plus [`crate::active_isa`] for the demo's ISA log. Each
//! validates its shapes at the boundary and returns
//! [`VokraError::InvalidArgument`] on a mismatch (NFR-RL-07); the `*_on`
//! variants force a specific [`IsaPath`] for differential testing.

pub(crate) mod scalar;

// Native vectorized `exp` shared by the AVX2 / NEON transcendental kernels
// (M1-05-EXP). Compiled only under the `simd-transcendental` feature; without
// it, `sigmoid` / `tanh` / `gelu` / softmax-exp stay scalar-backed and this
// module is not built.
#[cfg(feature = "simd-transcendental")]
pub(crate) mod vexp;

#[cfg(target_arch = "x86_64")]
pub(crate) mod avx2;

#[cfg(target_arch = "aarch64")]
pub(crate) mod neon;

use vokra_core::{Result, VokraError};

use crate::dispatch;
use crate::features::IsaPath;

// ---- production GEMM / GEMV execution (row-parallel when `parallel` is on) ----
//
// The `*_f32` public wrappers below route through these. When the `parallel`
// feature is on (native, multi-core host) the large GEMM/GEMV are split over
// disjoint output rows by `crate::pool` — bit-identical to the inline call (same
// per-element FMA chain), so parity is preserved. The `*_f32_on` differential
// entry points deliberately do NOT use the pool: they stay single-thread so a
// forced-ISA run is the single-thread numeric reference the pool is compared
// against. Off `parallel` (or on WASM / single core) both paths run inline.

#[cfg(all(feature = "parallel", not(target_family = "wasm")))]
#[allow(clippy::too_many_arguments)]
fn run_gemm(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    crate::pool::parallel_gemm(dispatch::table().gemm, m, n, k, a, b, bias, out);
}

#[cfg(not(all(feature = "parallel", not(target_family = "wasm"))))]
#[allow(clippy::too_many_arguments)]
fn run_gemm(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    (dispatch::table().gemm)(m, n, k, a, b, bias, out);
}

#[cfg(all(feature = "parallel", not(target_family = "wasm")))]
fn run_gemv(m: usize, k: usize, a: &[f32], x: &[f32], bias: Option<&[f32]>, out: &mut [f32]) {
    crate::pool::parallel_gemv(dispatch::table().gemv, m, k, a, x, bias, out);
}

#[cfg(not(all(feature = "parallel", not(target_family = "wasm"))))]
fn run_gemv(m: usize, k: usize, a: &[f32], x: &[f32], bias: Option<&[f32]>, out: &mut [f32]) {
    (dispatch::table().gemv)(m, k, a, x, bias, out);
}

/// Default layer-norm epsilon (PyTorch `nn.LayerNorm` default `1e-5`, which
/// OpenAI Whisper inherits). Exposed for M0-06 call sites.
pub const LAYER_NORM_DEFAULT_EPS: f32 = scalar::LAYER_NORM_DEFAULT_EPS;

// ---- boundary validation helpers (NFR-RL-07) ----

fn checked_mul(a: usize, b: usize, what: &str) -> Result<usize> {
    a.checked_mul(b).ok_or_else(|| {
        VokraError::InvalidArgument(format!("{what}: dimension product overflows usize"))
    })
}

fn expect_len(name: &str, got: usize, want: usize) -> Result<()> {
    if got == want {
        Ok(())
    } else {
        Err(VokraError::InvalidArgument(format!(
            "{name} length {got} does not match expected {want}"
        )))
    }
}

fn validate_gemm(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    bias: Option<&[f32]>,
    out: &[f32],
) -> Result<()> {
    expect_len("gemm a", a.len(), checked_mul(m, k, "gemm m*k")?)?;
    expect_len("gemm b", b.len(), checked_mul(k, n, "gemm k*n")?)?;
    expect_len("gemm out", out.len(), checked_mul(m, n, "gemm m*n")?)?;
    if let Some(bias) = bias {
        expect_len("gemm bias", bias.len(), n)?;
    }
    Ok(())
}

fn validate_gemv(
    m: usize,
    k: usize,
    a: &[f32],
    x: &[f32],
    bias: Option<&[f32]>,
    out: &[f32],
) -> Result<()> {
    expect_len("gemv a", a.len(), checked_mul(m, k, "gemv m*k")?)?;
    expect_len("gemv x", x.len(), k)?;
    expect_len("gemv out", out.len(), m)?;
    if let Some(bias) = bias {
        expect_len("gemv bias", bias.len(), m)?;
    }
    Ok(())
}

fn validate_binary(a: &[f32], b: &[f32], out: &[f32]) -> Result<()> {
    expect_len("binary b", b.len(), a.len())?;
    expect_len("binary out", out.len(), a.len())
}

fn validate_unary(x: &[f32], out: &[f32]) -> Result<()> {
    expect_len("unary out", out.len(), x.len())
}

fn validate_rows_cols(input: &[f32], out: &[f32], rows: usize, cols: usize) -> Result<()> {
    let total = checked_mul(rows, cols, "rows*cols")?;
    expect_len("input", input.len(), total)?;
    expect_len("out", out.len(), total)
}

// ---- dot product & GEMM (M0-08-T05) ----

/// Dot product of two equal-length f32 slices.
///
/// A scalar building block (no dispatch table entry); a length mismatch is an
/// explicit [`VokraError::InvalidArgument`].
pub fn vec_dot_f32(a: &[f32], b: &[f32]) -> Result<f32> {
    expect_len("vec_dot b", b.len(), a.len())?;
    Ok(scalar::vec_dot(a, b))
}

/// Row-major GEMM with optional per-column bias (bias = affine `linear`):
/// `out[i, j] = bias[j] + sum_l a[i, l] * b[l, j]`.
///
/// `a` is `m x k`, `b` is `k x n`, `out` is `m x n`, and `bias` (when `Some`)
/// has length `n`. Runs on [`crate::active_isa`]. A shape mismatch is an
/// explicit [`VokraError::InvalidArgument`].
pub fn gemm_f32(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
) -> Result<()> {
    validate_gemm(m, n, k, a, b, bias, out)?;
    run_gemm(m, n, k, a, b, bias, out);
    Ok(())
}

/// [`gemm_f32`] forced onto a specific `isa` (differential testing).
///
/// Always single-thread (never the pool): this is the numeric reference the
/// row-parallel production path is checked bit-for-bit against.
#[allow(clippy::too_many_arguments)] // mirrors gemm_f32 plus the forced isa
pub fn gemm_f32_on(
    isa: IsaPath,
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
) -> Result<()> {
    validate_gemm(m, n, k, a, b, bias, out)?;
    (dispatch::table_for(isa)?.gemm)(m, n, k, a, b, bias, out);
    Ok(())
}

/// Row-major matrix-vector product with an optional per-row bias:
/// `out[i] = bias[i] + sum_l a[i, l] * x[l]`.
///
/// `a` is `m x k`, `x` has length `k`, `out` has length `m`, and `bias` (when
/// `Some`) has length `m`. This is the `n = 1` case of [`gemm_f32`], but rather
/// than falling through that kernel's scalar column tail it streams each row of
/// `a` contiguously and reduces it with a wide SIMD FMA + horizontal sum. It is
/// the fast path for Whisper's tied logits head (`token_emb[v, d] @ h[d]`, the
/// single biggest per-token decode matmul). Runs on [`crate::active_isa`]; a
/// shape mismatch is an explicit [`VokraError::InvalidArgument`].
pub fn gemv_f32(
    m: usize,
    k: usize,
    a: &[f32],
    x: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
) -> Result<()> {
    validate_gemv(m, k, a, x, bias, out)?;
    run_gemv(m, k, a, x, bias, out);
    Ok(())
}

/// [`gemv_f32`] forced onto a specific `isa` (differential testing).
///
/// Always single-thread (never the pool): the numeric reference for the
/// row-parallel production path.
#[allow(clippy::too_many_arguments)] // mirrors gemv_f32 plus the forced isa
pub fn gemv_f32_on(
    isa: IsaPath,
    m: usize,
    k: usize,
    a: &[f32],
    x: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
) -> Result<()> {
    validate_gemv(m, k, a, x, bias, out)?;
    (dispatch::table_for(isa)?.gemv)(m, k, a, x, bias, out);
    Ok(())
}

// ---- element-wise & activations (M0-08-T06) ----

macro_rules! binary_wrapper {
    ($name:ident, $name_on:ident, $field:ident, $doc:literal) => {
        #[doc = $doc]
        ///
        /// `a`, `b`, `out` must have equal length. Runs on
        /// [`crate::active_isa`].
        pub fn $name(a: &[f32], b: &[f32], out: &mut [f32]) -> Result<()> {
            validate_binary(a, b, out)?;
            (dispatch::table().$field)(a, b, out);
            Ok(())
        }

        #[doc = concat!("[`", stringify!($name), "`] forced onto a specific `isa`.")]
        pub fn $name_on(isa: IsaPath, a: &[f32], b: &[f32], out: &mut [f32]) -> Result<()> {
            validate_binary(a, b, out)?;
            (dispatch::table_for(isa)?.$field)(a, b, out);
            Ok(())
        }
    };
}

macro_rules! unary_wrapper {
    ($name:ident, $name_on:ident, $field:ident, $doc:literal) => {
        #[doc = $doc]
        ///
        /// `x` and `out` must have equal length. Runs on
        /// [`crate::active_isa`].
        pub fn $name(x: &[f32], out: &mut [f32]) -> Result<()> {
            validate_unary(x, out)?;
            (dispatch::table().$field)(x, out);
            Ok(())
        }

        #[doc = concat!("[`", stringify!($name), "`] forced onto a specific `isa`.")]
        pub fn $name_on(isa: IsaPath, x: &[f32], out: &mut [f32]) -> Result<()> {
            validate_unary(x, out)?;
            (dispatch::table_for(isa)?.$field)(x, out);
            Ok(())
        }
    };
}

binary_wrapper!(add_f32, add_f32_on, add, "Element-wise `out = a + b`.");
binary_wrapper!(mul_f32, mul_f32_on, mul, "Element-wise `out = a * b`.");
unary_wrapper!(
    relu_f32,
    relu_f32_on,
    relu,
    "Element-wise ReLU `out = max(0, x)`."
);
unary_wrapper!(
    sigmoid_f32,
    sigmoid_f32_on,
    sigmoid,
    "Element-wise logistic sigmoid `out = 1 / (1 + exp(-x))`."
);
unary_wrapper!(
    tanh_f32,
    tanh_f32_on,
    tanh,
    "Element-wise hyperbolic tangent."
);
unary_wrapper!(
    gelu_f32,
    gelu_f32_on,
    gelu,
    "Element-wise exact (erf-based) GELU, matching Whisper's `nn.GELU()`."
);

// ---- softmax (M0-08-T07) ----

/// Row-wise softmax over the innermost dimension of a `rows x cols`
/// row-major buffer (numerically stabilised by the row max). Each output row
/// sums to 1 within FP32 rounding. Runs on [`crate::active_isa`].
pub fn softmax_f32(input: &[f32], out: &mut [f32], rows: usize, cols: usize) -> Result<()> {
    validate_rows_cols(input, out, rows, cols)?;
    (dispatch::table().softmax)(input, out, rows, cols);
    Ok(())
}

/// [`softmax_f32`] forced onto a specific `isa`.
pub fn softmax_f32_on(
    isa: IsaPath,
    input: &[f32],
    out: &mut [f32],
    rows: usize,
    cols: usize,
) -> Result<()> {
    validate_rows_cols(input, out, rows, cols)?;
    (dispatch::table_for(isa)?.softmax)(input, out, rows, cols);
    Ok(())
}

// ---- layer norm (M0-08-T07) ----

fn validate_layer_norm(
    input: &[f32],
    out: &[f32],
    rows: usize,
    cols: usize,
    gamma: &[f32],
    beta: &[f32],
) -> Result<()> {
    validate_rows_cols(input, out, rows, cols)?;
    expect_len("layer_norm gamma", gamma.len(), cols)?;
    expect_len("layer_norm beta", beta.len(), cols)
}

/// Row-wise layer normalisation with affine parameters over the innermost
/// dimension: `out[r, c] = (x[r, c] - mean_r) / sqrt(var_r + eps) *
/// gamma[c] + beta[c]`, using the biased (population) variance to match
/// PyTorch `nn.LayerNorm`. `gamma` / `beta` have length `cols`. See
/// [`LAYER_NORM_DEFAULT_EPS`]. Runs on [`crate::active_isa`].
pub fn layer_norm_f32(
    input: &[f32],
    out: &mut [f32],
    rows: usize,
    cols: usize,
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
) -> Result<()> {
    validate_layer_norm(input, out, rows, cols, gamma, beta)?;
    (dispatch::table().layer_norm)(input, out, rows, cols, gamma, beta, eps);
    Ok(())
}

/// [`layer_norm_f32`] forced onto a specific `isa`.
#[allow(clippy::too_many_arguments)] // mirrors layer_norm_f32 plus the forced isa
pub fn layer_norm_f32_on(
    isa: IsaPath,
    input: &[f32],
    out: &mut [f32],
    rows: usize,
    cols: usize,
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
) -> Result<()> {
    validate_layer_norm(input, out, rows, cols, gamma, beta)?;
    (dispatch::table_for(isa)?.layer_norm)(input, out, rows, cols, gamma, beta, eps);
    Ok(())
}

// ---- conv1d via im2col + GEMM (M0-08-T08) ----

/// 1-D convolution via im2col + [`gemm_f32`], so it rides the dispatched
/// SIMD GEMM (no dedicated conv SIMD kernel).
///
/// Layout: `input` is `in_ch x in_len` row-major, `weight` is
/// `out_ch x in_ch x kernel`, optional `bias` has length `out_ch`, and `out`
/// is `out_ch x out_len` where
/// `out_len = (in_len + 2 * padding - kernel) / stride + 1`. `stride` and
/// `kernel` must be non-zero and the padded length must be at least `kernel`;
/// any shape mismatch is an explicit [`VokraError::InvalidArgument`]. The
/// im2col buffer is allocated per call in M0; a static arena (M1-04, FR-EX-05)
/// can replace it later without changing this signature.
#[allow(clippy::too_many_arguments)] // convolution's intrinsic parameter set
pub fn conv1d_f32(
    input: &[f32],
    in_ch: usize,
    in_len: usize,
    weight: &[f32],
    out_ch: usize,
    kernel: usize,
    bias: Option<&[f32]>,
    stride: usize,
    padding: usize,
    out: &mut [f32],
) -> Result<()> {
    conv1d_dispatch(
        None, input, in_ch, in_len, weight, out_ch, kernel, bias, stride, padding, out,
    )
}

/// [`conv1d_f32`] forced onto a specific `isa` (drives the GEMM path).
#[allow(clippy::too_many_arguments)] // convolution's intrinsic parameter set
pub fn conv1d_f32_on(
    isa: IsaPath,
    input: &[f32],
    in_ch: usize,
    in_len: usize,
    weight: &[f32],
    out_ch: usize,
    kernel: usize,
    bias: Option<&[f32]>,
    stride: usize,
    padding: usize,
    out: &mut [f32],
) -> Result<()> {
    conv1d_dispatch(
        Some(isa),
        input,
        in_ch,
        in_len,
        weight,
        out_ch,
        kernel,
        bias,
        stride,
        padding,
        out,
    )
}

#[allow(clippy::too_many_arguments)] // convolution's intrinsic parameter set
fn conv1d_dispatch(
    force: Option<IsaPath>,
    input: &[f32],
    in_ch: usize,
    in_len: usize,
    weight: &[f32],
    out_ch: usize,
    kernel: usize,
    bias: Option<&[f32]>,
    stride: usize,
    padding: usize,
    out: &mut [f32],
) -> Result<()> {
    if stride == 0 {
        return Err(VokraError::InvalidArgument(
            "conv1d stride must be >= 1".into(),
        ));
    }
    if kernel == 0 {
        return Err(VokraError::InvalidArgument(
            "conv1d kernel must be >= 1".into(),
        ));
    }
    let padded = in_len
        .checked_add(checked_mul(2, padding, "conv1d 2*padding")?)
        .ok_or_else(|| VokraError::InvalidArgument("conv1d padded length overflow".into()))?;
    if padded < kernel {
        return Err(VokraError::InvalidArgument(format!(
            "conv1d padded length {padded} is smaller than kernel {kernel}"
        )));
    }
    let out_len = (padded - kernel) / stride + 1;

    expect_len(
        "conv1d input",
        input.len(),
        checked_mul(in_ch, in_len, "conv1d in_ch*in_len")?,
    )?;
    let k = checked_mul(in_ch, kernel, "conv1d in_ch*kernel")?;
    expect_len(
        "conv1d weight",
        weight.len(),
        checked_mul(out_ch, k, "conv1d out_ch*k")?,
    )?;
    expect_len(
        "conv1d out",
        out.len(),
        checked_mul(out_ch, out_len, "conv1d out_ch*out_len")?,
    )?;
    if let Some(bias) = bias {
        expect_len("conv1d bias", bias.len(), out_ch)?;
    }

    // im2col: col is [in_ch*kernel, out_len] row-major.
    let mut col = vec![0.0f32; k * out_len];
    for c in 0..in_ch {
        for kk in 0..kernel {
            let row = c * kernel + kk;
            for t in 0..out_len {
                let pos = t * stride + kk;
                if pos >= padding && pos < padding + in_len {
                    col[row * out_len + t] = input[c * in_len + (pos - padding)];
                }
            }
        }
    }

    // weight [out_ch, k] * col [k, out_len] = out [out_ch, out_len].
    match force {
        Some(isa) => gemm_f32_on(isa, out_ch, out_len, k, weight, &col, None, out)?,
        None => gemm_f32(out_ch, out_len, k, weight, &col, None, out)?,
    }

    // Per-output-channel bias (broadcast over out_len).
    if let Some(bias) = bias {
        for (oc, &b) in bias.iter().enumerate() {
            for v in &mut out[oc * out_len..oc * out_len + out_len] {
                *v += b;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemm_rejects_bad_shapes() {
        let a = [1.0, 2.0];
        let b = [1.0, 2.0];
        let mut out = [0.0; 4];
        // a should be m*k = 2*2 = 4 long, but it is 2 → explicit error.
        let err = gemm_f32(2, 2, 2, &a, &b, None, &mut out).unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn gemm_on_scalar_matches_hand_value() {
        // [[1,2],[3,4]] * [[1,0],[0,1]] + bias[100,200].
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [1.0, 0.0, 0.0, 1.0];
        let bias = [100.0, 200.0];
        let mut out = [0.0; 4];
        gemm_f32_on(IsaPath::Scalar, 2, 2, 2, &a, &b, Some(&bias), &mut out).unwrap();
        assert_eq!(out, [101.0, 202.0, 103.0, 204.0]);
    }

    #[test]
    fn gemv_on_scalar_matches_hand_value() {
        // a = [[1,2,3],[4,5,6]] (2x3), x = [1,0,-1], bias = [100, 200].
        // row0 = 1*1 + 2*0 + 3*-1 = -2 (+100 = 98); row1 = 4 - 6 = -2 (+200 = 198).
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = [1.0, 0.0, -1.0];
        let bias = [100.0, 200.0];
        let mut out = [0.0; 2];
        gemv_f32_on(IsaPath::Scalar, 2, 3, &a, &x, Some(&bias), &mut out).unwrap();
        assert_eq!(out, [98.0, 198.0]);
        // No-bias variant: exactly the n=1 column of gemm on the same data.
        gemv_f32_on(IsaPath::Scalar, 2, 3, &a, &x, None, &mut out).unwrap();
        assert_eq!(out, [-2.0, -2.0]);
    }

    #[test]
    fn gemv_matches_gemm_n1_column() {
        // gemv(m,k) must equal gemm(m, n=1, k) with x as the single b-column
        // (this is the equivalence the tied-logits-head routing relies on).
        let a = [0.5, -1.0, 2.0, 3.0, 0.25, -0.5]; // 2x3
        let x = [0.1, -0.2, 0.3];
        let mut g_out = [0.0; 2];
        gemm_f32(2, 1, 3, &a, &x, None, &mut g_out).unwrap();
        let mut v_out = [0.0; 2];
        gemv_f32_on(IsaPath::Scalar, 2, 3, &a, &x, None, &mut v_out).unwrap();
        assert_eq!(g_out, v_out);
    }

    #[test]
    fn gemv_rejects_bad_shapes() {
        // m=2, k=3: a needs 6, x needs 3, out needs 2, bias needs 2.
        let a = [0.0; 6];
        let x = [0.0; 3];
        let mut out = [0.0; 2];
        // `a` too short (5 != m*k = 6).
        assert!(gemv_f32(2, 3, &[0.0; 5], &x, None, &mut out).is_err());
        // `x` length != k (2 != 3).
        assert!(gemv_f32(2, 3, &a, &[0.0; 2], None, &mut out).is_err());
        // `out` length != m (3 != 2).
        assert!(gemv_f32(2, 3, &a, &x, None, &mut [0.0; 3]).is_err());
        // `bias` length != m (1 != 2).
        assert!(gemv_f32(2, 3, &a, &x, Some(&[0.0; 1]), &mut out).is_err());
        // m*k overflows usize -> explicit error via checked_mul (no kernel run).
        assert!(gemv_f32(usize::MAX, 2, &[], &[], None, &mut []).is_err());
    }

    #[test]
    fn binary_and_unary_reject_length_mismatch() {
        let mut out2 = [0.0; 2];
        assert!(add_f32(&[1.0, 2.0], &[1.0], &mut out2).is_err());
        let mut out1 = [0.0; 1];
        assert!(relu_f32(&[1.0, 2.0], &mut out1).is_err());
    }

    #[test]
    fn conv1d_single_channel_hand_fixture() {
        // input [1,5] = 1..5, weight [1,1,3] = [1,1,1], stride 1, pad 0.
        // out_len = 5-3+1 = 3; sliding sums: 1+2+3, 2+3+4, 3+4+5.
        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        let weight = [1.0, 1.0, 1.0];
        let mut out = [0.0; 3];
        conv1d_f32(&input, 1, 5, &weight, 1, 3, None, 1, 0, &mut out).unwrap();
        assert_eq!(out, [6.0, 9.0, 12.0]);
    }

    #[test]
    fn conv1d_padding_and_bias() {
        // input [1,3] = [1,2,3], weight [1,1,3] = [1,0,-1], pad 1, stride 1,
        // bias [10]. padded = [0,1,2,3,0], out_len = (5-3)/1+1 = 3.
        // windows: [0,1,2]·[1,0,-1] = -2; [1,2,3] = -2; [2,3,0] = 2. +bias 10.
        let input = [1.0, 2.0, 3.0];
        let weight = [1.0, 0.0, -1.0];
        let bias = [10.0];
        let mut out = [0.0; 3];
        conv1d_f32(&input, 1, 3, &weight, 1, 3, Some(&bias), 1, 1, &mut out).unwrap();
        assert_eq!(out, [8.0, 8.0, 12.0]);
    }

    #[test]
    fn conv1d_rejects_kernel_larger_than_padded_input() {
        let input = [1.0, 2.0];
        let weight = [1.0; 5];
        let mut out = [0.0; 1];
        let err = conv1d_f32(&input, 1, 2, &weight, 1, 5, None, 1, 0, &mut out).unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn conv1d_multichannel_stride2() {
        // 2 in-ch, len 4; 1 out-ch; kernel 2; stride 2; no pad.
        // in ch0 = [1,2,3,4], ch1 = [10,20,30,40].
        // weight [1,2,2] = ch0:[1,1], ch1:[1,1].
        // out_len = (4-2)/2+1 = 2.
        // t=0: ch0(1+2)+ch1(10+20)=3+30=33; t=1: ch0(3+4)+ch1(30+40)=7+70=77.
        let input = [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let weight = [1.0, 1.0, 1.0, 1.0];
        let mut out = [0.0; 2];
        conv1d_f32(&input, 2, 4, &weight, 1, 2, None, 2, 0, &mut out).unwrap();
        assert_eq!(out, [33.0, 77.0]);
    }

    #[test]
    fn vec_dot_hand_value_and_length_mismatch() {
        // 1*-1 + 2*0 + 3*2 = -1 + 0 + 6 = 5 (all terms exactly representable).
        let dot = vec_dot_f32(&[1.0, 2.0, 3.0], &[-1.0, 0.0, 2.0]).unwrap();
        assert!((dot - 5.0).abs() < 1e-6, "dot = {dot}, want 5.0");
        // Unequal lengths are an explicit error.
        let err = vec_dot_f32(&[1.0, 2.0], &[1.0]).unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn conv1d_nondivisible_stride_drops_trailing_window() {
        // input [1,6] = 1..=6, weight [1,1,2] = [1,1], stride 3, pad 0.
        // out_len = (6-2)/3+1 = 2. Windows begin at pos 0 and 3:
        // [1,2]·[1,1] = 3, [4,5]·[1,1] = 9. input[2]=3 and input[5]=6 lie past
        // the last full window and are intentionally dropped (im2col guard).
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let weight = [1.0, 1.0];
        let mut out = [0.0; 2];
        conv1d_f32(&input, 1, 6, &weight, 1, 2, None, 3, 0, &mut out).unwrap();
        assert_eq!(out, [3.0, 9.0]);
    }

    #[test]
    fn conv1d_padding_ge_in_len_zeros_outside_real_input() {
        // input [1,2] = [2,3], weight [1,1,1] = [1], stride 1, pad 2.
        // padded = 2 + 2*2 = 6, out_len = (6-1)/1+1 = 6. Only positions 2 and 3
        // fall inside the real input; the rest are pure zero-padding.
        let input = [2.0, 3.0];
        let weight = [1.0];
        let mut out = [0.0; 6];
        conv1d_f32(&input, 1, 2, &weight, 1, 1, None, 1, 2, &mut out).unwrap();
        assert_eq!(out, [0.0, 0.0, 2.0, 3.0, 0.0, 0.0]);
    }

    #[test]
    fn conv1d_rejects_zero_stride_and_zero_kernel() {
        // stride == 0 would divide-by-zero in the out_len formula.
        let mut out = [0.0; 1];
        let err = conv1d_f32(&[1.0, 2.0], 1, 2, &[1.0], 1, 1, None, 0, 0, &mut out).unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
        // kernel == 0 would mis-size the im2col matrix.
        let err = conv1d_f32(&[1.0, 2.0], 1, 2, &[], 1, 0, None, 1, 0, &mut out).unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn softmax_and_layer_norm_reject_shape_mismatch() {
        // softmax: length 6 does not match rows*cols = 2*4 = 8.
        let err = softmax_f32(&[0.0; 6], &mut [0.0; 6], 2, 4).unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
        // layer_norm: rows*cols is consistent (8), but gamma length 3 != cols 4.
        let err = layer_norm_f32(
            &[0.0; 8],
            &mut [0.0; 8],
            2,
            4,
            &[1.0; 3],
            &[0.0; 4],
            LAYER_NORM_DEFAULT_EPS,
        )
        .unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
        // ... and a beta length != cols is rejected by the same validator.
        let err = layer_norm_f32(
            &[0.0; 8],
            &mut [0.0; 8],
            2,
            4,
            &[1.0; 4],
            &[0.0; 3],
            LAYER_NORM_DEFAULT_EPS,
        )
        .unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn gemm_rejects_bad_b_out_bias_and_overflow() {
        // m=2, n=2, k=2: a needs 4, b needs k*n=4, out needs m*n=4, bias needs n=2.
        let a = [0.0; 4];
        let good_b = [0.0; 4];
        let mut out = [0.0; 4];
        // `b` too short (3 != k*n = 4).
        let err = gemm_f32(2, 2, 2, &a, &[0.0; 3], None, &mut out).unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
        // `out` too short (3 != m*n = 4).
        let mut short_out = [0.0; 3];
        let err = gemm_f32(2, 2, 2, &a, &good_b, None, &mut short_out).unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
        // `bias` length != n (1 != 2).
        let err = gemm_f32(2, 2, 2, &a, &good_b, Some(&[0.0; 1]), &mut out).unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
        // m*k overflows usize -> explicit error via the checked_mul guard (and
        // no kernel is ever entered with a bogus dimension product).
        let err = gemm_f32(usize::MAX, 1, 2, &[], &[], None, &mut []).unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }
}
