//! AVX2 + FMA f32 kernels for x86-64 (M0-08-T10..T12).
//!
//! `AVX2 + FMA3` is Vokra's x86-64 main path (FR-BE-01; CLAUDE.md "AVX2 +
//! FMA3 + F16C + BMI1/2 (Haswell 2013+) — 主力パス"). 256-bit vectors hold
//! eight f32 lanes; the ragged tail (`n % 8`) is handled scalar.
//!
//! # Unsafe boundary (NFR-RL-07, M0-08-T01/T10)
//!
//! Each kernel splits into a private `#[target_feature(enable = "avx2,fma")]
//! unsafe fn` doing the intrinsics, and a safe `pub(crate) fn` wrapper. The
//! wrapper is only ever installed into the [`crate::dispatch::KernelTable`]
//! or reached via `*_on(IsaPath::Avx2, ..)` **after**
//! [`crate::features::CpuFeatures::detect`] confirmed AVX2+FMA on this host
//! (see [`crate::dispatch`]), which is the invariant every `// SAFETY:`
//! comment below relies on. Slice lengths are validated by the public
//! wrappers in [`super`] first. No JIT / runtime code generation is involved
//! (NFR-RL-05): these are ordinary statically compiled functions selected by
//! a function pointer.
//!
//! The transcendental activations (`sigmoid` / `tanh` / `gelu`) currently
//! delegate to the portable scalar path: their cost is dominated by the
//! per-lane `exp`, so a vectorised `exp` approximation is a follow-up gated
//! on differential parity (M0-08-T11). Keeping them here fills the AVX2
//! kernel table for those op kinds so ISA-path coverage stays symmetric
//! (M0-08-T12).

use core::arch::x86_64::*;

use super::scalar;

/// Horizontal sum of the eight lanes of `v`.
///
/// # Safety
/// Requires the `avx2` target feature at the call site.
#[target_feature(enable = "avx2")]
unsafe fn hsum256(v: __m256) -> f32 {
    // SAFETY: the caller guarantees `avx2`; the store targets a fully owned,
    // correctly sized stack buffer.
    unsafe {
        let mut tmp = [0.0f32; 8];
        _mm256_storeu_ps(tmp.as_mut_ptr(), v);
        tmp.iter().sum()
    }
}

/// Horizontal maximum of the eight lanes of `v`.
///
/// # Safety
/// Requires the `avx2` target feature at the call site.
#[target_feature(enable = "avx2")]
unsafe fn hmax256(v: __m256) -> f32 {
    // SAFETY: the caller guarantees `avx2`; the store targets a fully owned,
    // correctly sized stack buffer.
    unsafe {
        let mut tmp = [0.0f32; 8];
        _mm256_storeu_ps(tmp.as_mut_ptr(), v);
        tmp.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    }
}

/// # Safety
/// Requires `avx2,fma`; shapes as documented on [`scalar::gemm`].
#[target_feature(enable = "avx2,fma")]
unsafe fn gemm_impl(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    // SAFETY: `avx2,fma` are guaranteed by the safe wrapper's dispatch
    // invariant. Every load/store below is bounded: the `j + 8 <= n` guard
    // keeps 8-wide accesses inside the length-`n` rows of `b` / `out` / `bias`
    // (all validated by the public wrapper), and the scalar tail covers the
    // remainder.
    unsafe {
        for i in 0..m {
            let out_row = &mut out[i * n..i * n + n];
            let mut j = 0;
            while j + 8 <= n {
                let mut acc = match bias {
                    Some(bias) => _mm256_loadu_ps(bias[j..].as_ptr()),
                    None => _mm256_setzero_ps(),
                };
                for l in 0..k {
                    let a_il = _mm256_set1_ps(a[i * k + l]);
                    let b_lj = _mm256_loadu_ps(b[l * n + j..].as_ptr());
                    acc = _mm256_fmadd_ps(a_il, b_lj, acc);
                }
                _mm256_storeu_ps(out_row[j..].as_mut_ptr(), acc);
                j += 8;
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
/// Requires `avx2`; `a.len() == b.len() == out.len()`.
#[target_feature(enable = "avx2")]
unsafe fn add_impl(a: &[f32], b: &[f32], out: &mut [f32]) {
    // SAFETY: `avx2` guaranteed by dispatch; `j + 8 <= len` bounds every load
    // and store to the equal-length, wrapper-validated slices.
    unsafe {
        let len = out.len();
        let mut j = 0;
        while j + 8 <= len {
            let r = _mm256_add_ps(
                _mm256_loadu_ps(a[j..].as_ptr()),
                _mm256_loadu_ps(b[j..].as_ptr()),
            );
            _mm256_storeu_ps(out[j..].as_mut_ptr(), r);
            j += 8;
        }
        while j < len {
            out[j] = a[j] + b[j];
            j += 1;
        }
    }
}

/// # Safety
/// Requires `avx2`; `a.len() == b.len() == out.len()`.
#[target_feature(enable = "avx2")]
unsafe fn mul_impl(a: &[f32], b: &[f32], out: &mut [f32]) {
    // SAFETY: `avx2` guaranteed by dispatch; `j + 8 <= len` bounds every load
    // and store to the equal-length, wrapper-validated slices.
    unsafe {
        let len = out.len();
        let mut j = 0;
        while j + 8 <= len {
            let r = _mm256_mul_ps(
                _mm256_loadu_ps(a[j..].as_ptr()),
                _mm256_loadu_ps(b[j..].as_ptr()),
            );
            _mm256_storeu_ps(out[j..].as_mut_ptr(), r);
            j += 8;
        }
        while j < len {
            out[j] = a[j] * b[j];
            j += 1;
        }
    }
}

/// # Safety
/// Requires `avx2`; `x.len() == out.len()`.
#[target_feature(enable = "avx2")]
unsafe fn relu_impl(x: &[f32], out: &mut [f32]) {
    // SAFETY: `avx2` guaranteed by dispatch; `j + 8 <= len` bounds every load
    // and store to the equal-length, wrapper-validated slices.
    unsafe {
        let len = out.len();
        let zero = _mm256_setzero_ps();
        let mut j = 0;
        while j + 8 <= len {
            let r = _mm256_max_ps(_mm256_loadu_ps(x[j..].as_ptr()), zero);
            _mm256_storeu_ps(out[j..].as_mut_ptr(), r);
            j += 8;
        }
        while j < len {
            out[j] = x[j].max(0.0);
            j += 1;
        }
    }
}

/// # Safety
/// Requires `avx2`; `input.len() == out.len() == rows * cols`.
#[target_feature(enable = "avx2")]
unsafe fn softmax_impl(input: &[f32], out: &mut [f32], rows: usize, cols: usize) {
    // SAFETY: `avx2` guaranteed by dispatch. Every 8-wide access is guarded
    // by `j + 8 <= cols` and confined to the length-`cols` row slices carved
    // from the wrapper-validated `rows * cols` buffers.
    unsafe {
        for r in 0..rows {
            let in_row = &input[r * cols..r * cols + cols];
            let out_row = &mut out[r * cols..r * cols + cols];

            // Pass 1: row maximum (SIMD reduction + scalar tail).
            let mut vmax = _mm256_set1_ps(f32::NEG_INFINITY);
            let mut j = 0;
            while j + 8 <= cols {
                vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(in_row[j..].as_ptr()));
                j += 8;
            }
            let mut max = hmax256(vmax);
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

            // Pass 3: scale by 1/sum (SIMD).
            let vinv = _mm256_set1_ps(1.0 / sum);
            let mut j = 0;
            while j + 8 <= cols {
                let v = _mm256_loadu_ps(out_row[j..].as_ptr());
                _mm256_storeu_ps(out_row[j..].as_mut_ptr(), _mm256_mul_ps(v, vinv));
                j += 8;
            }
            let inv = 1.0 / sum;
            while j < cols {
                out_row[j] *= inv;
                j += 1;
            }
        }
    }
}

/// # Safety
/// Requires `avx2,fma`; shapes as documented on [`scalar::layer_norm`].
#[target_feature(enable = "avx2,fma")]
unsafe fn layer_norm_impl(
    input: &[f32],
    out: &mut [f32],
    rows: usize,
    cols: usize,
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
) {
    // SAFETY: `avx2,fma` guaranteed by dispatch. All 8-wide accesses are
    // guarded by `j + 8 <= cols` over the length-`cols` row slices and the
    // length-`cols` `gamma` / `beta` (validated by the public wrapper).
    unsafe {
        let inv_cols = 1.0 / cols as f32;
        for r in 0..rows {
            let in_row = &input[r * cols..r * cols + cols];
            let out_row = &mut out[r * cols..r * cols + cols];

            // Pass 1: mean.
            let mut vsum = _mm256_setzero_ps();
            let mut j = 0;
            while j + 8 <= cols {
                vsum = _mm256_add_ps(vsum, _mm256_loadu_ps(in_row[j..].as_ptr()));
                j += 8;
            }
            let mut sum = hsum256(vsum);
            while j < cols {
                sum += in_row[j];
                j += 1;
            }
            let mean = sum * inv_cols;

            // Pass 2: variance (two-pass, matching the scalar oracle).
            let vmean = _mm256_set1_ps(mean);
            let mut vvar = _mm256_setzero_ps();
            let mut j = 0;
            while j + 8 <= cols {
                let d = _mm256_sub_ps(_mm256_loadu_ps(in_row[j..].as_ptr()), vmean);
                vvar = _mm256_fmadd_ps(d, d, vvar);
                j += 8;
            }
            let mut var = hsum256(vvar);
            while j < cols {
                let d = in_row[j] - mean;
                var += d * d;
                j += 1;
            }
            var *= inv_cols;
            let inv_std = 1.0 / (var + eps).sqrt();

            // Pass 3: normalise, scale, shift.
            let vinv_std = _mm256_set1_ps(inv_std);
            let mut j = 0;
            while j + 8 <= cols {
                let d = _mm256_sub_ps(_mm256_loadu_ps(in_row[j..].as_ptr()), vmean);
                let norm = _mm256_mul_ps(d, vinv_std);
                let g = _mm256_loadu_ps(gamma[j..].as_ptr());
                let b = _mm256_loadu_ps(beta[j..].as_ptr());
                _mm256_storeu_ps(out_row[j..].as_mut_ptr(), _mm256_fmadd_ps(norm, g, b));
                j += 8;
            }
            while j < cols {
                out_row[j] = (in_row[j] - mean) * inv_std * gamma[j] + beta[j];
                j += 1;
            }
        }
    }
}

// ---- Safe wrappers installed into the dispatch table (see module docs) ----

/// AVX2 GEMM. See [`scalar::gemm`] for shapes.
pub(crate) fn gemm(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    // SAFETY: reached only when AVX2+FMA were detected on this host (dispatch
    // invariant); slice shapes validated by the public wrapper in `super`.
    unsafe { gemm_impl(m, n, k, a, b, bias, out) }
}

/// AVX2 element-wise add.
pub(crate) fn add(a: &[f32], b: &[f32], out: &mut [f32]) {
    // SAFETY: reached only when AVX2 was detected (dispatch invariant);
    // equal slice lengths validated by the public wrapper.
    unsafe { add_impl(a, b, out) }
}

/// AVX2 element-wise multiply.
pub(crate) fn mul(a: &[f32], b: &[f32], out: &mut [f32]) {
    // SAFETY: reached only when AVX2 was detected (dispatch invariant);
    // equal slice lengths validated by the public wrapper.
    unsafe { mul_impl(a, b, out) }
}

/// AVX2 ReLU.
pub(crate) fn relu(x: &[f32], out: &mut [f32]) {
    // SAFETY: reached only when AVX2 was detected (dispatch invariant);
    // equal slice lengths validated by the public wrapper.
    unsafe { relu_impl(x, out) }
}

/// Sigmoid — scalar-backed in the spike (see module docs; M0-08-T11).
pub(crate) fn sigmoid(x: &[f32], out: &mut [f32]) {
    scalar::sigmoid(x, out);
}

/// Tanh — scalar-backed in the spike (see module docs; M0-08-T11).
pub(crate) fn tanh(x: &[f32], out: &mut [f32]) {
    scalar::tanh(x, out);
}

/// GELU — scalar-backed in the spike (see module docs; M0-08-T11).
pub(crate) fn gelu(x: &[f32], out: &mut [f32]) {
    scalar::gelu(x, out);
}

/// AVX2 row-wise softmax.
pub(crate) fn softmax(input: &[f32], out: &mut [f32], rows: usize, cols: usize) {
    // SAFETY: reached only when AVX2 was detected (dispatch invariant);
    // `input.len() == out.len() == rows * cols` validated by the wrapper.
    unsafe { softmax_impl(input, out, rows, cols) }
}

/// AVX2 row-wise layer norm.
pub(crate) fn layer_norm(
    input: &[f32],
    out: &mut [f32],
    rows: usize,
    cols: usize,
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
) {
    // SAFETY: reached only when AVX2+FMA were detected (dispatch invariant);
    // all shapes validated by the public wrapper in `super`.
    unsafe { layer_norm_impl(input, out, rows, cols, gamma, beta, eps) }
}
