//! AVX2 + FMA f32 kernels for x86-64 (M0-08-T10..T12).
//!
//! `AVX2 + FMA3` is Vokra's x86-64 main path (FR-BE-01; CLAUDE.md "AVX2 +
//! FMA3 + F16C + BMI1/2 (Haswell 2013+) — 主力パス"). 256-bit vectors hold
//! eight f32 lanes; the ragged tail (`n % 8`) is handled scalar.
//!
//! `gemm` is a **register-blocked** microkernel (M1-08): an `MR`×`NR` output
//! tile is held in `MR * NR / 8` independent `__m256` accumulators (see
//! [`MR`] / [`NR_VEC`]), so the `k`-loop keeps many FMA chains in flight and is
//! no longer FMA-latency-bound — the win for the `m = 1500` Whisper encoder
//! GEMMs. It stays a pure reordering of the same per-element FMA chains as the
//! scalar oracle (lane-aligned shapes are bit-identical to the pre-blocking
//! kernel), and the row/column remainders fall back to the original single-row
//! vector + scalar paths.
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
//! The transcendental activations (`sigmoid` / `tanh` / `gelu`) delegate to
//! the portable scalar path by default. Under the `simd-transcendental`
//! feature (M1-05-EXP) they instead use the native vectorized `exp` in
//! [`super::vexp`] — `sigmoid` = `1/(1+e^-x)`, `tanh` = `1 - 2/(e^{2x}+1)`,
//! `gelu` via the A&S erf with a vectorized `e^{-z²}` — and softmax's pass-2
//! `exp` is vectorized too. The feature is OFF by default: it moves these
//! paths off the scalar `std::exp` (a small controlled ULP delta), so
//! default-enablement is gated on the Whisper RTF + parity re-check in M1-11.

use core::arch::x86_64::*;

use super::scalar;
#[cfg(feature = "simd-transcendental")]
use super::vexp;

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

/// Register-block tile height (rows of `a` / `out` computed together).
///
/// `MR = 6` rows × `NR = 16` cols (two 8-lane vectors) holds `6 * 16 / 8 = 12`
/// live `__m256` accumulators; with the two `b` vectors and one `a` broadcast
/// (≈ 15) that stays inside x86-64's 16 YMM registers. This is the **tunable**
/// default: 12 independent FMA chains lift the encoder GEMMs off the old
/// single-accumulator, FMA-latency-bound path. See [`gemm_impl`].
const MR: usize = 6;
/// Register-block tile width in 8-lane AVX2 vectors (`NR = NR_VEC * 8 = 16`).
const NR_VEC: usize = 2;

/// # Safety
/// Requires `avx2,fma`; shapes as documented on [`scalar::gemm`].
///
/// Register-blocked `MR`×`NR` microkernel (`NR = NR_VEC * 8`): each B-load is
/// reused across the `MR` rows of the tile and each A-broadcast across the
/// `NR_VEC` B-vectors, so the `k`-loop runs `MR * NR_VEC` **independent**
/// accumulators and hides FMA latency. Every output element is still the same
/// bias-seeded FMA chain over `l = 0..k` as the scalar oracle (just tiled), so
/// results stay within the GEMM differential tolerance and — on lane-aligned
/// shapes — bit-identical to the pre-blocking AVX2 kernel.
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::needless_range_loop)] // explicit tile index math is clearer here
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
    // invariant. In the row-block loop `i + MR <= m` and `r < MR`, so
    // `i + r <= m - 1`; the column guards (`j + 16 <= n`, `j + 8 <= n`,
    // `j < n`) keep every 8-wide load/store inside the length-`n` rows of
    // `b` / `out` / `bias` and every `a[(i + r) * k + l]` inside `a` (all
    // lengths validated by the public wrapper). The row-tail loop repeats the
    // original single-row path with the same guards.
    unsafe {
        let mut i = 0;
        // ---- main path: full blocks of MR rows ----
        while i + MR <= m {
            let mut j = 0;
            // NR = NR_VEC * 8 columns: an MR × NR_VEC accumulator tile.
            while j + NR_VEC * 8 <= n {
                let (mut c0, mut c1) = match bias {
                    Some(bias) => (
                        [_mm256_loadu_ps(bias[j..].as_ptr()); MR],
                        [_mm256_loadu_ps(bias[j + 8..].as_ptr()); MR],
                    ),
                    None => ([_mm256_setzero_ps(); MR], [_mm256_setzero_ps(); MR]),
                };
                for l in 0..k {
                    let bl0 = _mm256_loadu_ps(b[l * n + j..].as_ptr());
                    let bl1 = _mm256_loadu_ps(b[l * n + j + 8..].as_ptr());
                    for r in 0..MR {
                        let ar = _mm256_set1_ps(a[(i + r) * k + l]);
                        c0[r] = _mm256_fmadd_ps(ar, bl0, c0[r]);
                        c1[r] = _mm256_fmadd_ps(ar, bl1, c1[r]);
                    }
                }
                for r in 0..MR {
                    _mm256_storeu_ps(out[(i + r) * n + j..].as_mut_ptr(), c0[r]);
                    _mm256_storeu_ps(out[(i + r) * n + j + 8..].as_mut_ptr(), c1[r]);
                }
                j += NR_VEC * 8;
            }
            // 8-wide column remainder: an MR × 1 accumulator tile.
            while j + 8 <= n {
                let mut c = match bias {
                    Some(bias) => [_mm256_loadu_ps(bias[j..].as_ptr()); MR],
                    None => [_mm256_setzero_ps(); MR],
                };
                for l in 0..k {
                    let bl = _mm256_loadu_ps(b[l * n + j..].as_ptr());
                    for r in 0..MR {
                        c[r] = _mm256_fmadd_ps(_mm256_set1_ps(a[(i + r) * k + l]), bl, c[r]);
                    }
                }
                for r in 0..MR {
                    _mm256_storeu_ps(out[(i + r) * n + j..].as_mut_ptr(), c[r]);
                }
                j += 8;
            }
            // Scalar column remainder: MR independent scalar accumulators.
            while j < n {
                let mut s = [0.0f32; MR];
                if let Some(bias) = bias {
                    for r in 0..MR {
                        s[r] = bias[j];
                    }
                }
                for l in 0..k {
                    let bv = b[l * n + j];
                    for r in 0..MR {
                        s[r] += a[(i + r) * k + l] * bv;
                    }
                }
                for r in 0..MR {
                    out[(i + r) * n + j] = s[r];
                }
                j += 1;
            }
            i += MR;
        }
        // ---- row tail: the leftover `m % MR` rows, one row at a time ----
        while i < m {
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
                _mm256_storeu_ps(out[i * n + j..].as_mut_ptr(), acc);
                j += 8;
            }
            while j < n {
                let mut s = bias.map_or(0.0, |bias| bias[j]);
                for l in 0..k {
                    s += a[i * k + l] * b[l * n + j];
                }
                out[i * n + j] = s;
                j += 1;
            }
            i += 1;
        }
    }
}

/// # Safety
/// Requires `avx2,fma`; shapes as documented on [`scalar::gemv`].
///
/// Per output row `i`, the dot product `sum_l a[i, l] * x[l]` is computed with
/// four independent 8-lane FMA accumulators (32 lanes per iteration) so the
/// `k`-loop keeps several FMA chains in flight and is not latency-bound, then
/// reduced by a tree add + [`hsum256`]; the `k % 32` (8-wide) and `k % 8`
/// (scalar) remainders follow. This is the same arithmetic as [`scalar::gemv`]
/// with a reordered reduction (within the gemv differential tolerance), and it
/// streams the `[m, k]` matrix `a` row-contiguously — the win over routing the
/// tied logits head through the `gemm` `n = 1` scalar tail.
#[target_feature(enable = "avx2,fma")]
unsafe fn gemv_impl(
    m: usize,
    k: usize,
    a: &[f32],
    x: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    // SAFETY: `avx2,fma` guaranteed by the safe wrapper's dispatch invariant.
    // For each row `i < m`, `base = i * k` and every 8-wide load is guarded by
    // `l + 32 <= k` / `l + 8 <= k`, so the top index `base + l + 31` (resp.
    // `+ 7`) stays inside the length-`m * k` slice `a` and each `x[l + ..]`
    // inside the length-`k` slice `x` (both validated by the public wrapper).
    // `out[i]` and `bias[i]` are inside their length-`m` slices.
    unsafe {
        for i in 0..m {
            let base = i * k;
            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();
            let mut acc2 = _mm256_setzero_ps();
            let mut acc3 = _mm256_setzero_ps();
            let mut l = 0;
            while l + 32 <= k {
                acc0 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(a[base + l..].as_ptr()),
                    _mm256_loadu_ps(x[l..].as_ptr()),
                    acc0,
                );
                acc1 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(a[base + l + 8..].as_ptr()),
                    _mm256_loadu_ps(x[l + 8..].as_ptr()),
                    acc1,
                );
                acc2 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(a[base + l + 16..].as_ptr()),
                    _mm256_loadu_ps(x[l + 16..].as_ptr()),
                    acc2,
                );
                acc3 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(a[base + l + 24..].as_ptr()),
                    _mm256_loadu_ps(x[l + 24..].as_ptr()),
                    acc3,
                );
                l += 32;
            }
            // 8-wide remainder folds into the first accumulator.
            while l + 8 <= k {
                acc0 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(a[base + l..].as_ptr()),
                    _mm256_loadu_ps(x[l..].as_ptr()),
                    acc0,
                );
                l += 8;
            }
            let acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
            let mut s = hsum256(acc);
            // Scalar `k % 8` tail.
            while l < k {
                s += a[base + l] * x[l];
                l += 1;
            }
            if let Some(bias) = bias {
                s += bias[i];
            }
            out[i] = s;
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
/// Requires `avx2,fma`; `input.len() == out.len() == rows * cols`. (FMA is
/// only actually used by the `simd-transcendental` pass-2 `exp`, but the
/// `Avx2` dispatch path is only ever selected when `avx2 && fma`, so requiring
/// it here is invariant-safe and keeps one attribute across feature configs.)
#[target_feature(enable = "avx2,fma")]
unsafe fn softmax_impl(input: &[f32], out: &mut [f32], rows: usize, cols: usize) {
    // SAFETY: `avx2,fma` guaranteed by dispatch. Every 8-wide access is guarded
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

            // Pass 2: out = exp(in - max); accumulate the row sum. The
            // vectorized `exp` (feature `simd-transcendental`) still handles the
            // ragged tail scalar for an exact-oracle match on `cols % 8`.
            #[cfg(feature = "simd-transcendental")]
            let sum = {
                let vmaxb = _mm256_set1_ps(max);
                let mut vsum = _mm256_setzero_ps();
                let mut j = 0;
                while j + 8 <= cols {
                    let e = vexp::exp_ps_avx2(_mm256_sub_ps(
                        _mm256_loadu_ps(in_row[j..].as_ptr()),
                        vmaxb,
                    ));
                    _mm256_storeu_ps(out_row[j..].as_mut_ptr(), e);
                    vsum = _mm256_add_ps(vsum, e);
                    j += 8;
                }
                let mut sum = hsum256(vsum);
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

/// AVX2 GEMV (matrix-vector). See [`scalar::gemv`] for shapes.
pub(crate) fn gemv(
    m: usize,
    k: usize,
    a: &[f32],
    x: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    // SAFETY: reached only when AVX2+FMA were detected on this host (dispatch
    // invariant); slice shapes validated by the public wrapper in `super`.
    unsafe { gemv_impl(m, k, a, x, bias, out) }
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

// ---- vectorized transcendentals (feature `simd-transcendental`, M1-05-EXP) ----

/// # Safety
/// Requires `avx2,fma`; `x.len() == out.len()`.
#[cfg(feature = "simd-transcendental")]
#[target_feature(enable = "avx2,fma")]
unsafe fn sigmoid_impl(x: &[f32], out: &mut [f32]) {
    // SAFETY: `avx2,fma` guaranteed by dispatch; `j + 8 <= len` bounds every
    // load/store to the equal-length, wrapper-validated slices; the ragged
    // tail is delegated to the scalar oracle for an exact tail match.
    unsafe {
        let len = out.len();
        let one = _mm256_set1_ps(1.0);
        let zero = _mm256_setzero_ps();
        let mut j = 0;
        while j + 8 <= len {
            let xv = _mm256_loadu_ps(x[j..].as_ptr());
            let e = vexp::exp_ps_avx2(_mm256_sub_ps(zero, xv)); // exp(-x)
            let r = _mm256_div_ps(one, _mm256_add_ps(one, e));
            _mm256_storeu_ps(out[j..].as_mut_ptr(), r);
            j += 8;
        }
        if j < len {
            scalar::sigmoid(&x[j..], &mut out[j..]);
        }
    }
}

/// # Safety
/// Requires `avx2,fma`; `x.len() == out.len()`.
#[cfg(feature = "simd-transcendental")]
#[target_feature(enable = "avx2,fma")]
unsafe fn tanh_impl(x: &[f32], out: &mut [f32]) {
    // SAFETY: `avx2,fma` guaranteed by dispatch; `j + 8 <= len` bounds every
    // load/store; the ragged tail delegates to the scalar oracle.
    unsafe {
        let len = out.len();
        let one = _mm256_set1_ps(1.0);
        let two = _mm256_set1_ps(2.0);
        let mut j = 0;
        while j + 8 <= len {
            let xv = _mm256_loadu_ps(x[j..].as_ptr());
            // tanh(x) = 1 - 2/(e^{2x}+1); saturates correctly at the clamped
            // exp domain (±1 for large |x|).
            let e2 = vexp::exp_ps_avx2(_mm256_mul_ps(two, xv));
            let r = _mm256_sub_ps(one, _mm256_div_ps(two, _mm256_add_ps(e2, one)));
            _mm256_storeu_ps(out[j..].as_mut_ptr(), r);
            j += 8;
        }
        if j < len {
            scalar::tanh(&x[j..], &mut out[j..]);
        }
    }
}

/// # Safety
/// Requires `avx2,fma`; `x.len() == out.len()`.
#[cfg(feature = "simd-transcendental")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gelu_impl(x: &[f32], out: &mut [f32]) {
    // SAFETY: `avx2,fma` guaranteed by dispatch; `j + 8 <= len` bounds every
    // load/store; the ragged tail delegates to the scalar oracle. Reuses the
    // exact A&S erf constants from `scalar` so only `exp(-z²)` differs.
    unsafe {
        let len = out.len();
        let one = _mm256_set1_ps(1.0);
        let zero = _mm256_setzero_ps();
        let half = _mm256_set1_ps(0.5);
        let inv_sqrt2 = _mm256_set1_ps(std::f32::consts::FRAC_1_SQRT_2);
        let sign_mask = _mm256_set1_ps(-0.0); // 0x8000_0000
        let p = _mm256_set1_ps(scalar::ERF_P);
        let a1 = _mm256_set1_ps(scalar::ERF_A1);
        let a2 = _mm256_set1_ps(scalar::ERF_A2);
        let a3 = _mm256_set1_ps(scalar::ERF_A3);
        let a4 = _mm256_set1_ps(scalar::ERF_A4);
        let a5 = _mm256_set1_ps(scalar::ERF_A5);
        let mut j = 0;
        while j + 8 <= len {
            let xv = _mm256_loadu_ps(x[j..].as_ptr());
            let z = _mm256_mul_ps(xv, inv_sqrt2);
            let az = _mm256_andnot_ps(sign_mask, z); // |z|
            let t = _mm256_div_ps(one, _mm256_fmadd_ps(p, az, one)); // 1/(1+P|z|)
            // poly = ((((A5*t + A4)*t + A3)*t + A2)*t + A1)*t
            let mut poly = _mm256_fmadd_ps(a5, t, a4);
            poly = _mm256_fmadd_ps(poly, t, a3);
            poly = _mm256_fmadd_ps(poly, t, a2);
            poly = _mm256_fmadd_ps(poly, t, a1);
            poly = _mm256_mul_ps(poly, t);
            let ez2 = vexp::exp_ps_avx2(_mm256_sub_ps(zero, _mm256_mul_ps(az, az))); // e^{-z²}
            let erf_abs = _mm256_sub_ps(one, _mm256_mul_ps(poly, ez2)); // erf(|z|) ≥ 0
            // copysign(erf_abs, z): OR z's sign bit onto the non-negative magnitude.
            let erf = _mm256_or_ps(_mm256_and_ps(sign_mask, z), erf_abs);
            let g = _mm256_mul_ps(_mm256_mul_ps(half, xv), _mm256_add_ps(one, erf));
            _mm256_storeu_ps(out[j..].as_mut_ptr(), g);
            j += 8;
        }
        if j < len {
            scalar::gelu(&x[j..], &mut out[j..]);
        }
    }
}

/// AVX2 sigmoid — vectorized `exp` under `simd-transcendental`, else scalar.
#[cfg(feature = "simd-transcendental")]
pub(crate) fn sigmoid(x: &[f32], out: &mut [f32]) {
    // SAFETY: reached only when AVX2+FMA were detected (dispatch invariant);
    // equal slice lengths validated by the public wrapper.
    unsafe { sigmoid_impl(x, out) }
}

/// AVX2 sigmoid — scalar-backed (default; SIMD under `simd-transcendental`).
#[cfg(not(feature = "simd-transcendental"))]
pub(crate) fn sigmoid(x: &[f32], out: &mut [f32]) {
    scalar::sigmoid(x, out);
}

/// AVX2 tanh — vectorized `exp` under `simd-transcendental`, else scalar.
#[cfg(feature = "simd-transcendental")]
pub(crate) fn tanh(x: &[f32], out: &mut [f32]) {
    // SAFETY: reached only when AVX2+FMA were detected (dispatch invariant);
    // equal slice lengths validated by the public wrapper.
    unsafe { tanh_impl(x, out) }
}

/// AVX2 tanh — scalar-backed (default; SIMD under `simd-transcendental`).
#[cfg(not(feature = "simd-transcendental"))]
pub(crate) fn tanh(x: &[f32], out: &mut [f32]) {
    scalar::tanh(x, out);
}

/// AVX2 GELU — vectorized `exp` under `simd-transcendental`, else scalar.
#[cfg(feature = "simd-transcendental")]
pub(crate) fn gelu(x: &[f32], out: &mut [f32]) {
    // SAFETY: reached only when AVX2+FMA were detected (dispatch invariant);
    // equal slice lengths validated by the public wrapper.
    unsafe { gelu_impl(x, out) }
}

/// AVX2 GELU — scalar-backed (default; SIMD under `simd-transcendental`).
#[cfg(not(feature = "simd-transcendental"))]
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
