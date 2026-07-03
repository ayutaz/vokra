//! Portable scalar f32 kernels (M0-08-T05..T08).
//!
//! These are 100% safe Rust and serve two roles (M0-08-T05):
//!
//! 1. the **fallback path** on x86-64 CPUs without AVX2 (M0-08-T01 (d)); and
//! 2. the **differential oracle** for the SIMD kernels (M0-08-T09): every
//!    AVX2 / NEON kernel is checked against the matching scalar kernel here.
//!
//! All functions in this module take *already validated* slices: the public
//! safe wrappers in [`super`] check lengths / shapes and return
//! [`VokraError::InvalidArgument`](vokra_core::VokraError::InvalidArgument)
//! before dispatching here (NFR-RL-07 API-boundary safety). The signatures
//! therefore mirror the kernel function-pointer types in
//! [`crate::dispatch`] and do not return `Result`.
//!
//! # Numeric notes
//!
//! `sigmoid` / `tanh` / `gelu` use `f32` transcendental functions from the
//! Rust standard library, so NaN in ⇒ NaN out and the usual IEEE-754
//! saturation applies (e.g. `sigmoid(+inf) == 1.0`, `sigmoid(-inf) == 0.0`).

/// Default epsilon for [`layer_norm`], matching PyTorch `nn.LayerNorm`'s
/// documented default (`eps = 1e-5`), which OpenAI Whisper's `LayerNorm`
/// subclass inherits unchanged (it only overrides the dtype cast). Callers
/// may pass a different value through the public wrapper.
pub(crate) const LAYER_NORM_DEFAULT_EPS: f32 = 1e-5;

/// Dot product of two equal-length slices.
///
/// Precondition (checked by the caller): `a.len() == b.len()`.
pub(crate) fn vec_dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

/// Row-major GEMM with an optional per-column bias:
/// `out[i, j] = bias[j] + sum_l a[i, l] * b[l, j]`.
///
/// Shapes (checked by the caller): `a` is `m x k`, `b` is `k x n`,
/// `out` is `m x n`, and `bias` (when `Some`) has length `n`.
#[allow(clippy::needless_range_loop)] // explicit index math is clearer for GEMM
pub(crate) fn gemm(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    for i in 0..m {
        let out_row = &mut out[i * n..i * n + n];
        match bias {
            Some(bias) => out_row.copy_from_slice(bias),
            None => out_row.fill(0.0),
        }
        for l in 0..k {
            let a_il = a[i * k + l];
            let b_row = &b[l * n..l * n + n];
            for j in 0..n {
                out_row[j] += a_il * b_row[j];
            }
        }
    }
}

/// Element-wise `out[i] = a[i] + b[i]` (precondition: equal lengths).
pub(crate) fn add(a: &[f32], b: &[f32], out: &mut [f32]) {
    for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
        *o = x + y;
    }
}

/// Element-wise `out[i] = a[i] * b[i]` (precondition: equal lengths).
pub(crate) fn mul(a: &[f32], b: &[f32], out: &mut [f32]) {
    for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
        *o = x * y;
    }
}

/// Element-wise ReLU: `out[i] = max(0, x[i])`.
pub(crate) fn relu(x: &[f32], out: &mut [f32]) {
    for (o, &v) in out.iter_mut().zip(x) {
        *o = v.max(0.0);
    }
}

/// Element-wise logistic sigmoid `out[i] = 1 / (1 + exp(-x[i]))`.
pub(crate) fn sigmoid(x: &[f32], out: &mut [f32]) {
    for (o, &v) in out.iter_mut().zip(x) {
        *o = 1.0 / (1.0 + (-v).exp());
    }
}

/// Element-wise hyperbolic tangent.
pub(crate) fn tanh(x: &[f32], out: &mut [f32]) {
    for (o, &v) in out.iter_mut().zip(x) {
        *o = v.tanh();
    }
}

// Abramowitz & Stegun 7.1.26 erf coefficients (max abs error 1.5e-7, well
// inside the FP32 parity ceiling atol = 0.01, NFR-QL-01). Exposed at module
// scope so the SIMD `gelu` kernels (M1-05-EXP) reuse the *identical* constants
// — the only numeric difference between scalar and vectorized `gelu` is then
// the vectorized `exp(-x^2)`, keeping them within a bounded ULP delta.
#[allow(clippy::excessive_precision)] // canonical A&S constants kept verbatim (auditable); excess digits round to f32 harmlessly
pub(crate) const ERF_P: f32 = 0.327_591_1;
#[allow(clippy::excessive_precision)]
pub(crate) const ERF_A1: f32 = 0.254_829_592;
#[allow(clippy::excessive_precision)]
pub(crate) const ERF_A2: f32 = -0.284_496_736;
#[allow(clippy::excessive_precision)]
pub(crate) const ERF_A3: f32 = 1.421_413_741;
#[allow(clippy::excessive_precision)]
pub(crate) const ERF_A4: f32 = -1.453_152_027;
#[allow(clippy::excessive_precision)]
pub(crate) const ERF_A5: f32 = 1.061_405_429;

/// Error function approximation, Abramowitz & Stegun 7.1.26.
///
/// Maximum absolute error 1.5e-7 over all `x` (A&S), i.e. well inside the
/// FP32 parity ceiling atol = 0.01 (NFR-QL-01). Verified against reference
/// `erf` values in the unit tests.
fn erf(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + ERF_P * x);
    let poly = ((((ERF_A5 * t + ERF_A4) * t + ERF_A3) * t + ERF_A2) * t + ERF_A1) * t;
    let y = 1.0 - poly * (-x * x).exp();
    sign * y
}

/// Element-wise exact (erf-based) GELU:
/// `out[i] = 0.5 * x[i] * (1 + erf(x[i] / sqrt(2)))`.
///
/// This matches OpenAI Whisper's `nn.GELU()` (default `approximate='none'`,
/// i.e. the exact/erf form, not the tanh approximation). `erf` uses the A&S
/// 7.1.26 approximation; end-to-end bit-parity against PyTorch is validated
/// by M0-06's parity harness (NFR-QL-01).
pub(crate) fn gelu(x: &[f32], out: &mut [f32]) {
    for (o, &v) in out.iter_mut().zip(x) {
        *o = 0.5 * v * (1.0 + erf(v * std::f32::consts::FRAC_1_SQRT_2));
    }
}

/// Row-wise softmax over the innermost dimension, numerically stabilised by
/// subtracting the row maximum.
///
/// `input` / `out` are `rows x cols` row-major (preconditions checked by the
/// caller). Each output row sums to 1 (within FP32 rounding).
pub(crate) fn softmax(input: &[f32], out: &mut [f32], rows: usize, cols: usize) {
    for r in 0..rows {
        let in_row = &input[r * cols..r * cols + cols];
        let out_row = &mut out[r * cols..r * cols + cols];
        let max = in_row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for (o, &v) in out_row.iter_mut().zip(in_row) {
            let e = (v - max).exp();
            *o = e;
            sum += e;
        }
        let inv = 1.0 / sum;
        for o in out_row.iter_mut() {
            *o *= inv;
        }
    }
}

/// Row-wise layer normalisation with affine parameters:
/// `out[r, c] = (x[r, c] - mean_r) / sqrt(var_r + eps) * gamma[c] + beta[c]`.
///
/// `input` / `out` are `rows x cols` row-major; `gamma` / `beta` have length
/// `cols` (preconditions checked by the caller). `var_r` is the biased
/// (population) variance over the row, matching PyTorch `nn.LayerNorm`.
pub(crate) fn layer_norm(
    input: &[f32],
    out: &mut [f32],
    rows: usize,
    cols: usize,
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
) {
    let inv_cols = 1.0 / cols as f32;
    for r in 0..rows {
        let in_row = &input[r * cols..r * cols + cols];
        let out_row = &mut out[r * cols..r * cols + cols];
        let mean = in_row.iter().sum::<f32>() * inv_cols;
        let var = in_row.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() * inv_cols;
        let inv_std = 1.0 / (var + eps).sqrt();
        for ((o, &v), (&g, &b)) in out_row.iter_mut().zip(in_row).zip(gamma.iter().zip(beta)) {
            *o = (v - mean) * inv_std * g + b;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erf_matches_reference_values() {
        // Reference erf values (double precision, rounded).
        let cases = [
            (0.0f32, 0.0f32),
            (0.5, 0.520_499_9),
            (1.0, 0.842_700_8),
            (2.0, 0.995_322_3),
            (-1.0, -0.842_700_8),
        ];
        for (x, want) in cases {
            let got = erf(x);
            assert!((got - want).abs() < 1e-5, "erf({x}) = {got}, want ~{want}");
        }
    }

    #[test]
    fn gemm_identity_and_bias() {
        // 2x2 identity times [[1,2],[3,4]] plus bias [10, 20].
        let a = [1.0, 0.0, 0.0, 1.0];
        let b = [1.0, 2.0, 3.0, 4.0];
        let bias = [10.0, 20.0];
        let mut out = [0.0; 4];
        gemm(2, 2, 2, &a, &b, Some(&bias), &mut out);
        assert_eq!(out, [11.0, 22.0, 13.0, 24.0]);
    }

    #[test]
    fn gemm_hand_computed_no_bias() {
        // [[1,2,3]] (1x3) * [[1],[0],[-1]] (3x1) = [[-2]].
        let a = [1.0, 2.0, 3.0];
        let b = [1.0, 0.0, -1.0];
        let mut out = [0.0; 1];
        gemm(1, 1, 3, &a, &b, None, &mut out);
        assert_eq!(out, [-2.0]);
    }

    #[test]
    fn activations_known_points() {
        let mut out = [0.0; 3];
        relu(&[-1.0, 0.0, 2.0], &mut out);
        assert_eq!(out, [0.0, 0.0, 2.0]);

        sigmoid(&[0.0, 100.0, -100.0], &mut out);
        assert!((out[0] - 0.5).abs() < 1e-6);
        assert!((out[1] - 1.0).abs() < 1e-6);
        assert!(out[2].abs() < 1e-6);

        tanh(&[0.0, 30.0, -30.0], &mut out);
        assert!(out[0].abs() < 1e-6);
        assert!((out[1] - 1.0).abs() < 1e-6);
        assert!((out[2] + 1.0).abs() < 1e-6);

        // gelu(0) = 0, gelu is (0,1) mass scaled; check monotone-ish points.
        let mut g = [0.0; 3];
        gelu(&[0.0, 1.0, -1.0], &mut g);
        assert!(g[0].abs() < 1e-6);
        assert!((g[1] - 0.841_192).abs() < 1e-3); // gelu(1) ~= 0.8412
        assert!(g[2] < 0.0 && g[2] > -0.2);
    }

    #[test]
    fn softmax_sums_to_one_and_shift_invariant() {
        let mut out = [0.0; 4];
        softmax(&[1.0, 2.0, 3.0, 4.0], &mut out, 1, 4);
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Adding a constant to all inputs leaves softmax unchanged.
        let mut shifted = [0.0; 4];
        softmax(&[101.0, 102.0, 103.0, 104.0], &mut shifted, 1, 4);
        for (a, b) in out.iter().zip(&shifted) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn softmax_uniform_input_uniform_output() {
        let mut out = [0.0; 5];
        softmax(&[7.0; 5], &mut out, 1, 5);
        for &v in &out {
            assert!((v - 0.2).abs() < 1e-6);
        }
    }

    #[test]
    fn layer_norm_zero_mean_unit_var() {
        let mut out = [0.0; 4];
        let gamma = [1.0; 4];
        let beta = [0.0; 4];
        layer_norm(
            &[1.0, 2.0, 3.0, 4.0],
            &mut out,
            1,
            4,
            &gamma,
            &beta,
            LAYER_NORM_DEFAULT_EPS,
        );
        let mean: f32 = out.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-4, "mean = {mean}");
        let var: f32 = out.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / 4.0;
        assert!((var - 1.0).abs() < 1e-3, "var = {var}");
    }
}
