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
//! scalar path for the same reason as the AVX2 module (M0-08-T14).

use core::arch::aarch64::*;

use super::scalar;

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

            // Pass 2: exp (scalar) and running sum.
            let mut sum = 0.0f32;
            for (o, &v) in out_row.iter_mut().zip(in_row) {
                let e = (v - max).exp();
                *o = e;
                sum += e;
            }

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

/// Sigmoid — scalar-backed in the spike (see module docs; M0-08-T14).
pub(crate) fn sigmoid(x: &[f32], out: &mut [f32]) {
    scalar::sigmoid(x, out);
}

/// Tanh — scalar-backed in the spike (see module docs; M0-08-T14).
pub(crate) fn tanh(x: &[f32], out: &mut [f32]) {
    scalar::tanh(x, out);
}

/// GELU — scalar-backed in the spike (see module docs; M0-08-T14).
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
