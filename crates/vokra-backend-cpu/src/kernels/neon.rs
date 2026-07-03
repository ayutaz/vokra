//! NEON f32 kernels for AArch64 (M0-08-T13..T15).
//!
//! NEON is the ARMv8-A baseline — present on every AArch64 CPU (CLAUDE.md
//! "NEON (ARMv8-A baseline、常時対応)"), so unlike AVX2 there is no runtime
//! feature branch: on `aarch64` the NEON table is always installed
//! (M0-08-T03). 128-bit vectors hold four f32 lanes; the ragged tail
//! (`n % 4`) is handled scalar. The upper ISA tiers (dotprod / i8mm / bf16 /
//! SVE / SVE2 / SME) are out of the spike scope (FR-BE-01; CLAUDE.md
//! "ARM64 系").
//!
//! # Unsafe boundary (NFR-RL-07, M0-08-T01/T13)
//!
//! Same structure as [`super::avx2`]: private
//! `#[target_feature(enable = "neon")] unsafe fn` cores behind safe
//! `pub(crate) fn` wrappers. On AArch64 NEON is baseline so availability is
//! unconditional; the `// SAFETY:` comments rely on that plus the length
//! validation performed by the public wrappers in [`super`]. No JIT is used
//! (NFR-RL-05): these are statically compiled functions reached through a
//! function pointer.
//!
//! Transcendental activations (`sigmoid` / `tanh` / `gelu`) delegate to the
//! scalar path by default; under the `simd-transcendental` feature (M1-05-EXP)
//! they use the native vectorized `exp` in [`super::vexp`] (and softmax's
//! pass-2 `exp` is vectorized too), mirroring the AVX2 module. The feature is
//! OFF by default — a small controlled ULP delta vs `std::exp`, gated on the
//! Whisper RTF + parity re-check in M1-11.

use core::arch::aarch64::*;

use super::scalar;
#[cfg(feature = "simd-transcendental")]
use super::vexp;

/// # Safety
/// Requires `neon` (baseline on AArch64); shapes as documented on
/// [`scalar::gemm`].
#[target_feature(enable = "neon")]
unsafe fn gemm_impl(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    // SAFETY: NEON is baseline on AArch64. Every 4-wide load/store is guarded
    // by `j + 4 <= n`, keeping accesses inside the length-`n` rows of `b` /
    // `out` / `bias` (all validated by the public wrapper); the scalar tail
    // covers the remainder.
    unsafe {
        for i in 0..m {
            let out_row = &mut out[i * n..i * n + n];
            let mut j = 0;
            while j + 4 <= n {
                let mut acc = match bias {
                    Some(bias) => vld1q_f32(bias[j..].as_ptr()),
                    None => vdupq_n_f32(0.0),
                };
                for l in 0..k {
                    let a_il = vdupq_n_f32(a[i * k + l]);
                    let b_lj = vld1q_f32(b[l * n + j..].as_ptr());
                    acc = vfmaq_f32(acc, a_il, b_lj);
                }
                vst1q_f32(out_row[j..].as_mut_ptr(), acc);
                j += 4;
            }
            while j < n {
                let mut s = bias.map_or(0.0, |bias| bias[j]);
                for l in 0..k {
                    s += a[i * k + l] * b[l * n + j];
                }
                out_row[j] = s;
                j += 1;
            }
        }
    }
}

/// # Safety
/// Requires `neon`; `a.len() == b.len() == out.len()`.
#[target_feature(enable = "neon")]
unsafe fn add_impl(a: &[f32], b: &[f32], out: &mut [f32]) {
    // SAFETY: NEON baseline; `j + 4 <= len` bounds every access to the
    // equal-length, wrapper-validated slices.
    unsafe {
        let len = out.len();
        let mut j = 0;
        while j + 4 <= len {
            let r = vaddq_f32(vld1q_f32(a[j..].as_ptr()), vld1q_f32(b[j..].as_ptr()));
            vst1q_f32(out[j..].as_mut_ptr(), r);
            j += 4;
        }
        while j < len {
            out[j] = a[j] + b[j];
            j += 1;
        }
    }
}

/// # Safety
/// Requires `neon`; `a.len() == b.len() == out.len()`.
#[target_feature(enable = "neon")]
unsafe fn mul_impl(a: &[f32], b: &[f32], out: &mut [f32]) {
    // SAFETY: NEON baseline; `j + 4 <= len` bounds every access to the
    // equal-length, wrapper-validated slices.
    unsafe {
        let len = out.len();
        let mut j = 0;
        while j + 4 <= len {
            let r = vmulq_f32(vld1q_f32(a[j..].as_ptr()), vld1q_f32(b[j..].as_ptr()));
            vst1q_f32(out[j..].as_mut_ptr(), r);
            j += 4;
        }
        while j < len {
            out[j] = a[j] * b[j];
            j += 1;
        }
    }
}

/// # Safety
/// Requires `neon`; `x.len() == out.len()`.
#[target_feature(enable = "neon")]
unsafe fn relu_impl(x: &[f32], out: &mut [f32]) {
    // SAFETY: NEON baseline; `j + 4 <= len` bounds every access to the
    // equal-length, wrapper-validated slices.
    unsafe {
        let len = out.len();
        let zero = vdupq_n_f32(0.0);
        let mut j = 0;
        while j + 4 <= len {
            let r = vmaxq_f32(vld1q_f32(x[j..].as_ptr()), zero);
            vst1q_f32(out[j..].as_mut_ptr(), r);
            j += 4;
        }
        while j < len {
            out[j] = x[j].max(0.0);
            j += 1;
        }
    }
}

/// # Safety
/// Requires `neon`; `input.len() == out.len() == rows * cols`.
#[target_feature(enable = "neon")]
unsafe fn softmax_impl(input: &[f32], out: &mut [f32], rows: usize, cols: usize) {
    // SAFETY: NEON baseline. Every 4-wide access is guarded by `j + 4 <= cols`
    // over the length-`cols` row slices carved from the wrapper-validated
    // `rows * cols` buffers.
    unsafe {
        for r in 0..rows {
            let in_row = &input[r * cols..r * cols + cols];
            let out_row = &mut out[r * cols..r * cols + cols];

            // Pass 1: row maximum.
            let mut vmax = vdupq_n_f32(f32::NEG_INFINITY);
            let mut j = 0;
            while j + 4 <= cols {
                vmax = vmaxq_f32(vmax, vld1q_f32(in_row[j..].as_ptr()));
                j += 4;
            }
            let mut max = vmaxvq_f32(vmax);
            while j < cols {
                max = max.max(in_row[j]);
                j += 1;
            }

            // Pass 2: out = exp(in - max); accumulate the row sum. The
            // vectorized `exp` (feature `simd-transcendental`) still handles the
            // ragged tail scalar for an exact-oracle match on `cols % 4`.
            #[cfg(feature = "simd-transcendental")]
            let sum = {
                let vmaxb = vdupq_n_f32(max);
                let mut vsum = vdupq_n_f32(0.0);
                let mut j = 0;
                while j + 4 <= cols {
                    let e = vexp::exp_ps_neon(vsubq_f32(vld1q_f32(in_row[j..].as_ptr()), vmaxb));
                    vst1q_f32(out_row[j..].as_mut_ptr(), e);
                    vsum = vaddq_f32(vsum, e);
                    j += 4;
                }
                let mut sum = vaddvq_f32(vsum);
                while j < cols {
                    let e = (in_row[j] - max).exp();
                    out_row[j] = e;
                    sum += e;
                    j += 1;
                }
                sum
            };
            #[cfg(not(feature = "simd-transcendental"))]
            let sum = {
                let mut sum = 0.0f32;
                for (o, &v) in out_row.iter_mut().zip(in_row) {
                    let e = (v - max).exp();
                    *o = e;
                    sum += e;
                }
                sum
            };

            // Pass 3: scale by 1/sum.
            let inv = 1.0 / sum;
            let vinv = vdupq_n_f32(inv);
            let mut j = 0;
            while j + 4 <= cols {
                let v = vld1q_f32(out_row[j..].as_ptr());
                vst1q_f32(out_row[j..].as_mut_ptr(), vmulq_f32(v, vinv));
                j += 4;
            }
            while j < cols {
                out_row[j] *= inv;
                j += 1;
            }
        }
    }
}

/// # Safety
/// Requires `neon`; shapes as documented on [`scalar::layer_norm`].
#[target_feature(enable = "neon")]
unsafe fn layer_norm_impl(
    input: &[f32],
    out: &mut [f32],
    rows: usize,
    cols: usize,
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
) {
    // SAFETY: NEON baseline. All 4-wide accesses are guarded by `j + 4 <= cols`
    // over the length-`cols` row slices and the length-`cols` `gamma` / `beta`
    // (validated by the public wrapper).
    unsafe {
        let inv_cols = 1.0 / cols as f32;
        for r in 0..rows {
            let in_row = &input[r * cols..r * cols + cols];
            let out_row = &mut out[r * cols..r * cols + cols];

            // Pass 1: mean.
            let mut vsum = vdupq_n_f32(0.0);
            let mut j = 0;
            while j + 4 <= cols {
                vsum = vaddq_f32(vsum, vld1q_f32(in_row[j..].as_ptr()));
                j += 4;
            }
            let mut sum = vaddvq_f32(vsum);
            while j < cols {
                sum += in_row[j];
                j += 1;
            }
            let mean = sum * inv_cols;

            // Pass 2: variance (two-pass, matching the scalar oracle).
            let vmean = vdupq_n_f32(mean);
            let mut vvar = vdupq_n_f32(0.0);
            let mut j = 0;
            while j + 4 <= cols {
                let d = vsubq_f32(vld1q_f32(in_row[j..].as_ptr()), vmean);
                vvar = vfmaq_f32(vvar, d, d);
                j += 4;
            }
            let mut var = vaddvq_f32(vvar);
            while j < cols {
                let d = in_row[j] - mean;
                var += d * d;
                j += 1;
            }
            var *= inv_cols;
            let inv_std = 1.0 / (var + eps).sqrt();

            // Pass 3: normalise, scale, shift.
            let vinv_std = vdupq_n_f32(inv_std);
            let mut j = 0;
            while j + 4 <= cols {
                let d = vsubq_f32(vld1q_f32(in_row[j..].as_ptr()), vmean);
                let norm = vmulq_f32(d, vinv_std);
                let g = vld1q_f32(gamma[j..].as_ptr());
                let b = vld1q_f32(beta[j..].as_ptr());
                vst1q_f32(out_row[j..].as_mut_ptr(), vfmaq_f32(b, norm, g));
                j += 4;
            }
            while j < cols {
                out_row[j] = (in_row[j] - mean) * inv_std * gamma[j] + beta[j];
                j += 1;
            }
        }
    }
}

// ---- Safe wrappers installed into the dispatch table (see module docs) ----

/// NEON GEMM. See [`scalar::gemm`] for shapes.
pub(crate) fn gemm(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    // SAFETY: NEON is baseline on AArch64 (this module only compiles there);
    // slice shapes validated by the public wrapper in `super`.
    unsafe { gemm_impl(m, n, k, a, b, bias, out) }
}

/// NEON element-wise add.
pub(crate) fn add(a: &[f32], b: &[f32], out: &mut [f32]) {
    // SAFETY: NEON baseline; equal slice lengths validated by the wrapper.
    unsafe { add_impl(a, b, out) }
}

/// NEON element-wise multiply.
pub(crate) fn mul(a: &[f32], b: &[f32], out: &mut [f32]) {
    // SAFETY: NEON baseline; equal slice lengths validated by the wrapper.
    unsafe { mul_impl(a, b, out) }
}

/// NEON ReLU.
pub(crate) fn relu(x: &[f32], out: &mut [f32]) {
    // SAFETY: NEON baseline; equal slice lengths validated by the wrapper.
    unsafe { relu_impl(x, out) }
}

// ---- vectorized transcendentals (feature `simd-transcendental`, M1-05-EXP) ----

/// # Safety
/// Requires `neon` (baseline); `x.len() == out.len()`.
#[cfg(feature = "simd-transcendental")]
#[target_feature(enable = "neon")]
unsafe fn sigmoid_impl(x: &[f32], out: &mut [f32]) {
    // SAFETY: NEON baseline; `j + 4 <= len` bounds every load/store to the
    // equal-length, wrapper-validated slices; the ragged tail is delegated to
    // the scalar oracle for an exact tail match.
    unsafe {
        let len = out.len();
        let one = vdupq_n_f32(1.0);
        let mut j = 0;
        while j + 4 <= len {
            let xv = vld1q_f32(x[j..].as_ptr());
            let e = vexp::exp_ps_neon(vnegq_f32(xv)); // exp(-x)
            let r = vdivq_f32(one, vaddq_f32(one, e));
            vst1q_f32(out[j..].as_mut_ptr(), r);
            j += 4;
        }
        if j < len {
            scalar::sigmoid(&x[j..], &mut out[j..]);
        }
    }
}

/// # Safety
/// Requires `neon` (baseline); `x.len() == out.len()`.
#[cfg(feature = "simd-transcendental")]
#[target_feature(enable = "neon")]
unsafe fn tanh_impl(x: &[f32], out: &mut [f32]) {
    // SAFETY: NEON baseline; `j + 4 <= len` bounds every load/store; the ragged
    // tail delegates to the scalar oracle.
    unsafe {
        let len = out.len();
        let one = vdupq_n_f32(1.0);
        let two = vdupq_n_f32(2.0);
        let mut j = 0;
        while j + 4 <= len {
            let xv = vld1q_f32(x[j..].as_ptr());
            // tanh(x) = 1 - 2/(e^{2x}+1); saturates correctly at ±1.
            let e2 = vexp::exp_ps_neon(vmulq_f32(two, xv));
            let r = vsubq_f32(one, vdivq_f32(two, vaddq_f32(e2, one)));
            vst1q_f32(out[j..].as_mut_ptr(), r);
            j += 4;
        }
        if j < len {
            scalar::tanh(&x[j..], &mut out[j..]);
        }
    }
}

/// # Safety
/// Requires `neon` (baseline); `x.len() == out.len()`.
#[cfg(feature = "simd-transcendental")]
#[target_feature(enable = "neon")]
unsafe fn gelu_impl(x: &[f32], out: &mut [f32]) {
    // SAFETY: NEON baseline; `j + 4 <= len` bounds every load/store; the ragged
    // tail delegates to the scalar oracle. Reuses the exact A&S erf constants
    // from `scalar` so only `exp(-z²)` differs.
    unsafe {
        let len = out.len();
        let one = vdupq_n_f32(1.0);
        let half = vdupq_n_f32(0.5);
        let inv_sqrt2 = vdupq_n_f32(std::f32::consts::FRAC_1_SQRT_2);
        let sign_mask = vdupq_n_u32(0x8000_0000);
        let p = vdupq_n_f32(scalar::ERF_P);
        let a1 = vdupq_n_f32(scalar::ERF_A1);
        let a2 = vdupq_n_f32(scalar::ERF_A2);
        let a3 = vdupq_n_f32(scalar::ERF_A3);
        let a4 = vdupq_n_f32(scalar::ERF_A4);
        let a5 = vdupq_n_f32(scalar::ERF_A5);
        let mut j = 0;
        while j + 4 <= len {
            let xv = vld1q_f32(x[j..].as_ptr());
            let z = vmulq_f32(xv, inv_sqrt2);
            let az = vabsq_f32(z); // |z|
            let t = vdivq_f32(one, vfmaq_f32(one, p, az)); // 1/(1+P|z|)
            // poly = ((((A5*t + A4)*t + A3)*t + A2)*t + A1)*t
            let mut poly = vfmaq_f32(a4, a5, t);
            poly = vfmaq_f32(a3, poly, t);
            poly = vfmaq_f32(a2, poly, t);
            poly = vfmaq_f32(a1, poly, t);
            poly = vmulq_f32(poly, t);
            let ez2 = vexp::exp_ps_neon(vnegq_f32(vmulq_f32(az, az))); // e^{-z²}
            let erf_abs = vsubq_f32(one, vmulq_f32(poly, ez2)); // erf(|z|) ≥ 0
            // copysign(erf_abs, z): OR z's sign bit onto the non-negative magnitude.
            let sign = vandq_u32(vreinterpretq_u32_f32(z), sign_mask);
            let erf = vreinterpretq_f32_u32(vorrq_u32(sign, vreinterpretq_u32_f32(erf_abs)));
            let g = vmulq_f32(vmulq_f32(half, xv), vaddq_f32(one, erf));
            vst1q_f32(out[j..].as_mut_ptr(), g);
            j += 4;
        }
        if j < len {
            scalar::gelu(&x[j..], &mut out[j..]);
        }
    }
}

/// NEON sigmoid — vectorized `exp` under `simd-transcendental`, else scalar.
#[cfg(feature = "simd-transcendental")]
pub(crate) fn sigmoid(x: &[f32], out: &mut [f32]) {
    // SAFETY: NEON baseline; equal slice lengths validated by the wrapper.
    unsafe { sigmoid_impl(x, out) }
}

/// NEON sigmoid — scalar-backed (default; SIMD under `simd-transcendental`).
#[cfg(not(feature = "simd-transcendental"))]
pub(crate) fn sigmoid(x: &[f32], out: &mut [f32]) {
    scalar::sigmoid(x, out);
}

/// NEON tanh — vectorized `exp` under `simd-transcendental`, else scalar.
#[cfg(feature = "simd-transcendental")]
pub(crate) fn tanh(x: &[f32], out: &mut [f32]) {
    // SAFETY: NEON baseline; equal slice lengths validated by the wrapper.
    unsafe { tanh_impl(x, out) }
}

/// NEON tanh — scalar-backed (default; SIMD under `simd-transcendental`).
#[cfg(not(feature = "simd-transcendental"))]
pub(crate) fn tanh(x: &[f32], out: &mut [f32]) {
    scalar::tanh(x, out);
}

/// NEON GELU — vectorized `exp` under `simd-transcendental`, else scalar.
#[cfg(feature = "simd-transcendental")]
pub(crate) fn gelu(x: &[f32], out: &mut [f32]) {
    // SAFETY: NEON baseline; equal slice lengths validated by the wrapper.
    unsafe { gelu_impl(x, out) }
}

/// NEON GELU — scalar-backed (default; SIMD under `simd-transcendental`).
#[cfg(not(feature = "simd-transcendental"))]
pub(crate) fn gelu(x: &[f32], out: &mut [f32]) {
    scalar::gelu(x, out);
}

/// NEON row-wise softmax.
pub(crate) fn softmax(input: &[f32], out: &mut [f32], rows: usize, cols: usize) {
    // SAFETY: NEON baseline; `input.len() == out.len() == rows * cols`
    // validated by the wrapper.
    unsafe { softmax_impl(input, out, rows, cols) }
}

/// NEON row-wise layer norm.
pub(crate) fn layer_norm(
    input: &[f32],
    out: &mut [f32],
    rows: usize,
    cols: usize,
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
) {
    // SAFETY: NEON baseline; all shapes validated by the public wrapper.
    unsafe { layer_norm_impl(input, out, rows, cols, gamma, beta, eps) }
}
