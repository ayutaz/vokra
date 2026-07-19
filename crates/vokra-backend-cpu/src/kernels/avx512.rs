//! AVX-512 f32 kernels for x86-64 (M4-17-T07..T09) plus the VNNI INT8 and
//! BF16 matmul cores (M4-17-T10/T11).
//!
//! The f32 tier mirrors [`super::avx2`] widened to 16 f32 lanes per `__m512`,
//! compiled with the `avx512f,avx512dq,avx512bw,avx512vl` bundle — the four
//! ship together on every Skylake-X 2017+ / Zen4 part (FR-BE-01), which is
//! why [`crate::features::CpuFeatures::supports`] gates [`IsaPath::Avx512`]
//! on the whole bundle (ADR M4-17 §(b)-4). Ragged tails use AVX-512 masked
//! loads/stores (`_mm512_maskz_loadu_ps` / `_mm512_mask_storeu_ps`), which
//! compute the same per-element FMA chains as the scalar oracle — the masked
//! path is a lane subset, not a different reduction order.
//!
//! # Unsafe boundary (NFR-RL-07)
//!
//! Same structure as [`super::avx2`]: private `#[target_feature]` `unsafe fn`
//! cores behind safe `pub(crate) fn` wrappers, installed into the dispatch
//! table only after [`crate::features::CpuFeatures::detect`] confirmed the
//! bundle on this host. No JIT (NFR-RL-05); intrinsics are stable on the
//! pinned rustc (1.95 — verified by compile probe, ADR M4-17 §(g)).
//!
//! Transcendental activations (`sigmoid` / `tanh` / `gelu`) are NOT
//! implemented here: the dispatch table delegates them to the AVX2 kernels
//! (any AVX-512 host is an AVX2+FMA host by the `supports` gate), so the
//! `simd-transcendental` feature posture stays in sync with the AVX2 tier
//! automatically. `softmax`'s pass-2 `exp` stays scalar in this kernel (it
//! bit-matches the oracle's `exp` exactly); passes 1/3 are 16-lane. The
//! `fused_logmel` mel-band accumulation is 16-lane FMA and the final
//! `log10(max(acc, floor))` uses `std` `log10` — identical to the scalar
//! reference's `log10`, so only the accumulation order differs (the 512-bit
//! `vlog10` polynomial port is a perf follow-up; perf here is advisory,
//! owner-measured — M4-17-T23).

use core::arch::x86_64::*;

/// Register-block tile height (rows of `a` / `out` computed together).
///
/// `MR = 6` rows × `NR = 32` cols (two 16-lane vectors) holds `12` live
/// `__m512` accumulators; with two `b` vectors and one `a` broadcast (≈ 15)
/// that sits comfortably inside AVX-512's 32 zmm registers. Same independent
/// FMA-chain rationale as [`super::avx2`] (M1-08).
const MR: usize = 6;
/// Register-block tile width in 16-lane vectors (`NR = NR_VEC * 16 = 32`).
const NR_VEC: usize = 2;

/// Bitmask selecting the low `rem` (< 16) lanes of a `__m512`.
#[inline]
fn tail_mask(rem: usize) -> __mmask16 {
    debug_assert!(rem < 16);
    ((1u32 << rem) - 1) as __mmask16
}

/// # Safety
/// Requires the AVX-512 f32 bundle; shapes as documented on [`scalar::gemm`].
///
/// Register-blocked `MR`×`NR` microkernel — the [`super::avx2::gemm`]
/// structure at 16 lanes. Every output element is the same bias-seeded FMA
/// chain over `l = 0..k` as the scalar oracle (tiled); the masked column
/// tail computes identical per-element chains on a lane subset.
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl")]
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
    // SAFETY: the AVX-512 bundle is guaranteed by the safe wrapper's dispatch
    // invariant. In the row-block loop `i + MR <= m` and `r < MR`, so
    // `i + r <= m - 1`; the column guards (`j + 32 <= n`, `j + 16 <= n`) keep
    // every 16-wide load/store inside the length-`n` rows of `b` / `out` /
    // `bias`, and the masked tail (`rem = n - j < 16`) touches only the low
    // `rem` lanes of those rows. Every `a[(i + r) * k + l]` is inside `a`
    // (all lengths validated by the public wrapper).
    unsafe {
        let mut i = 0;
        // ---- main path: full blocks of MR rows ----
        while i + MR <= m {
            let mut j = 0;
            // NR = NR_VEC * 16 columns: an MR × NR_VEC accumulator tile.
            while j + NR_VEC * 16 <= n {
                let (mut c0, mut c1) = match bias {
                    Some(bias) => (
                        [_mm512_loadu_ps(bias[j..].as_ptr()); MR],
                        [_mm512_loadu_ps(bias[j + 16..].as_ptr()); MR],
                    ),
                    None => ([_mm512_setzero_ps(); MR], [_mm512_setzero_ps(); MR]),
                };
                for l in 0..k {
                    let bl0 = _mm512_loadu_ps(b[l * n + j..].as_ptr());
                    let bl1 = _mm512_loadu_ps(b[l * n + j + 16..].as_ptr());
                    for r in 0..MR {
                        let ar = _mm512_set1_ps(a[(i + r) * k + l]);
                        c0[r] = _mm512_fmadd_ps(ar, bl0, c0[r]);
                        c1[r] = _mm512_fmadd_ps(ar, bl1, c1[r]);
                    }
                }
                for r in 0..MR {
                    _mm512_storeu_ps(out[(i + r) * n + j..].as_mut_ptr(), c0[r]);
                    _mm512_storeu_ps(out[(i + r) * n + j + 16..].as_mut_ptr(), c1[r]);
                }
                j += NR_VEC * 16;
            }
            // 16-wide column remainder: an MR × 1 accumulator tile.
            while j + 16 <= n {
                let mut c = match bias {
                    Some(bias) => [_mm512_loadu_ps(bias[j..].as_ptr()); MR],
                    None => [_mm512_setzero_ps(); MR],
                };
                for l in 0..k {
                    let bl = _mm512_loadu_ps(b[l * n + j..].as_ptr());
                    for r in 0..MR {
                        c[r] = _mm512_fmadd_ps(_mm512_set1_ps(a[(i + r) * k + l]), bl, c[r]);
                    }
                }
                for r in 0..MR {
                    _mm512_storeu_ps(out[(i + r) * n + j..].as_mut_ptr(), c[r]);
                }
                j += 16;
            }
            // Masked column remainder (< 16 columns) — same per-element FMA
            // chains on the low `rem` lanes only.
            if j < n {
                let mask = tail_mask(n - j);
                let mut c = match bias {
                    Some(bias) => [_mm512_maskz_loadu_ps(mask, bias[j..].as_ptr()); MR],
                    None => [_mm512_setzero_ps(); MR],
                };
                for l in 0..k {
                    let bl = _mm512_maskz_loadu_ps(mask, b[l * n + j..].as_ptr());
                    for r in 0..MR {
                        c[r] = _mm512_fmadd_ps(_mm512_set1_ps(a[(i + r) * k + l]), bl, c[r]);
                    }
                }
                for r in 0..MR {
                    _mm512_mask_storeu_ps(out[(i + r) * n + j..].as_mut_ptr(), mask, c[r]);
                }
            }
            i += MR;
        }
        // ---- row tail: the leftover `m % MR` rows, one row at a time ----
        while i < m {
            let mut j = 0;
            while j + 16 <= n {
                let mut acc = match bias {
                    Some(bias) => _mm512_loadu_ps(bias[j..].as_ptr()),
                    None => _mm512_setzero_ps(),
                };
                for l in 0..k {
                    let a_il = _mm512_set1_ps(a[i * k + l]);
                    let b_lj = _mm512_loadu_ps(b[l * n + j..].as_ptr());
                    acc = _mm512_fmadd_ps(a_il, b_lj, acc);
                }
                _mm512_storeu_ps(out[i * n + j..].as_mut_ptr(), acc);
                j += 16;
            }
            if j < n {
                let mask = tail_mask(n - j);
                let mut acc = match bias {
                    Some(bias) => _mm512_maskz_loadu_ps(mask, bias[j..].as_ptr()),
                    None => _mm512_setzero_ps(),
                };
                for l in 0..k {
                    let a_il = _mm512_set1_ps(a[i * k + l]);
                    let b_lj = _mm512_maskz_loadu_ps(mask, b[l * n + j..].as_ptr());
                    acc = _mm512_fmadd_ps(a_il, b_lj, acc);
                }
                _mm512_mask_storeu_ps(out[i * n + j..].as_mut_ptr(), mask, acc);
            }
            i += 1;
        }
    }
}

/// # Safety
/// Requires the AVX-512 f32 bundle; shapes as documented on [`scalar::gemv`].
///
/// [`super::avx2::gemv`] at 16 lanes: four independent 16-lane FMA
/// accumulators (64 lanes per iteration), a 16-wide remainder, a tree add +
/// `_mm512_reduce_add_ps`, then the scalar `k % 16` tail — a reordered
/// reduction of the same arithmetic as [`scalar::gemv`] (within the GEMV
/// differential tolerance).
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl")]
unsafe fn gemv_impl(
    m: usize,
    k: usize,
    a: &[f32],
    x: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    // SAFETY: bundle guaranteed by the dispatch invariant. For each row
    // `i < m`, every 16-wide load is guarded by `l + 64 <= k` / `l + 16 <= k`
    // so the top index stays inside the wrapper-validated `a` / `x`; `out[i]`
    // and `bias[i]` are inside their length-`m` slices.
    unsafe {
        for i in 0..m {
            let base = i * k;
            let mut acc0 = _mm512_setzero_ps();
            let mut acc1 = _mm512_setzero_ps();
            let mut acc2 = _mm512_setzero_ps();
            let mut acc3 = _mm512_setzero_ps();
            let mut l = 0;
            while l + 64 <= k {
                acc0 = _mm512_fmadd_ps(
                    _mm512_loadu_ps(a[base + l..].as_ptr()),
                    _mm512_loadu_ps(x[l..].as_ptr()),
                    acc0,
                );
                acc1 = _mm512_fmadd_ps(
                    _mm512_loadu_ps(a[base + l + 16..].as_ptr()),
                    _mm512_loadu_ps(x[l + 16..].as_ptr()),
                    acc1,
                );
                acc2 = _mm512_fmadd_ps(
                    _mm512_loadu_ps(a[base + l + 32..].as_ptr()),
                    _mm512_loadu_ps(x[l + 32..].as_ptr()),
                    acc2,
                );
                acc3 = _mm512_fmadd_ps(
                    _mm512_loadu_ps(a[base + l + 48..].as_ptr()),
                    _mm512_loadu_ps(x[l + 48..].as_ptr()),
                    acc3,
                );
                l += 64;
            }
            while l + 16 <= k {
                acc0 = _mm512_fmadd_ps(
                    _mm512_loadu_ps(a[base + l..].as_ptr()),
                    _mm512_loadu_ps(x[l..].as_ptr()),
                    acc0,
                );
                l += 16;
            }
            let acc = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
            let mut s = _mm512_reduce_add_ps(acc);
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
/// Requires the AVX-512 f32 bundle; `a.len() == b.len() == out.len()`.
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl")]
unsafe fn add_impl(a: &[f32], b: &[f32], out: &mut [f32]) {
    // SAFETY: bundle guaranteed by dispatch; `j + 16 <= len` bounds the full
    // vectors and the masked tail touches only the low `len - j` lanes of the
    // equal-length, wrapper-validated slices. Element-wise add is per-lane
    // exact, so this is bit-identical to the scalar oracle.
    unsafe {
        let len = out.len();
        let mut j = 0;
        while j + 16 <= len {
            let r = _mm512_add_ps(
                _mm512_loadu_ps(a[j..].as_ptr()),
                _mm512_loadu_ps(b[j..].as_ptr()),
            );
            _mm512_storeu_ps(out[j..].as_mut_ptr(), r);
            j += 16;
        }
        if j < len {
            let mask = tail_mask(len - j);
            let r = _mm512_add_ps(
                _mm512_maskz_loadu_ps(mask, a[j..].as_ptr()),
                _mm512_maskz_loadu_ps(mask, b[j..].as_ptr()),
            );
            _mm512_mask_storeu_ps(out[j..].as_mut_ptr(), mask, r);
        }
    }
}

/// # Safety
/// Requires the AVX-512 f32 bundle; `a.len() == b.len() == out.len()`.
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl")]
unsafe fn mul_impl(a: &[f32], b: &[f32], out: &mut [f32]) {
    // SAFETY: same bounds argument as `add_impl` (per-lane exact multiply).
    unsafe {
        let len = out.len();
        let mut j = 0;
        while j + 16 <= len {
            let r = _mm512_mul_ps(
                _mm512_loadu_ps(a[j..].as_ptr()),
                _mm512_loadu_ps(b[j..].as_ptr()),
            );
            _mm512_storeu_ps(out[j..].as_mut_ptr(), r);
            j += 16;
        }
        if j < len {
            let mask = tail_mask(len - j);
            let r = _mm512_mul_ps(
                _mm512_maskz_loadu_ps(mask, a[j..].as_ptr()),
                _mm512_maskz_loadu_ps(mask, b[j..].as_ptr()),
            );
            _mm512_mask_storeu_ps(out[j..].as_mut_ptr(), mask, r);
        }
    }
}

/// # Safety
/// Requires the AVX-512 f32 bundle; `x.len() == out.len()`.
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl")]
unsafe fn relu_impl(x: &[f32], out: &mut [f32]) {
    // SAFETY: same bounds argument as `add_impl` (per-lane exact max).
    unsafe {
        let len = out.len();
        let zero = _mm512_setzero_ps();
        let mut j = 0;
        while j + 16 <= len {
            let r = _mm512_max_ps(_mm512_loadu_ps(x[j..].as_ptr()), zero);
            _mm512_storeu_ps(out[j..].as_mut_ptr(), r);
            j += 16;
        }
        if j < len {
            let mask = tail_mask(len - j);
            let r = _mm512_max_ps(_mm512_maskz_loadu_ps(mask, x[j..].as_ptr()), zero);
            _mm512_mask_storeu_ps(out[j..].as_mut_ptr(), mask, r);
        }
    }
}

/// # Safety
/// Requires the AVX-512 f32 bundle; `input.len() == out.len() == rows * cols`.
///
/// Pass 1 (row max) and pass 3 (scale by `1/sum`) are 16-lane; pass 2's `exp`
/// stays scalar so it bit-matches the oracle's `exp` (see module docs).
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl")]
unsafe fn softmax_impl(input: &[f32], out: &mut [f32], rows: usize, cols: usize) {
    // SAFETY: bundle guaranteed by dispatch. Every 16-wide access is guarded
    // by `j + 16 <= cols` and confined to the length-`cols` row slices carved
    // from the wrapper-validated `rows * cols` buffers.
    unsafe {
        for r in 0..rows {
            let in_row = &input[r * cols..r * cols + cols];
            let out_row = &mut out[r * cols..r * cols + cols];

            // Pass 1: row maximum (16-lane reduction + scalar tail).
            let mut vmax = _mm512_set1_ps(f32::NEG_INFINITY);
            let mut j = 0;
            while j + 16 <= cols {
                vmax = _mm512_max_ps(vmax, _mm512_loadu_ps(in_row[j..].as_ptr()));
                j += 16;
            }
            let mut max = _mm512_reduce_max_ps(vmax);
            while j < cols {
                max = max.max(in_row[j]);
                j += 1;
            }

            // Pass 2: out = exp(in - max), scalar `exp` (oracle-exact).
            let mut sum = 0.0f32;
            for (o, &v) in out_row.iter_mut().zip(in_row) {
                let e = (v - max).exp();
                *o = e;
                sum += e;
            }

            // Pass 3: scale by 1/sum (16-lane + masked tail; per-lane exact
            // multiply by the same reciprocal as the scalar path).
            let vinv = _mm512_set1_ps(1.0 / sum);
            let mut j = 0;
            while j + 16 <= cols {
                let v = _mm512_loadu_ps(out_row[j..].as_ptr());
                _mm512_storeu_ps(out_row[j..].as_mut_ptr(), _mm512_mul_ps(v, vinv));
                j += 16;
            }
            if j < cols {
                let mask = tail_mask(cols - j);
                let v = _mm512_maskz_loadu_ps(mask, out_row[j..].as_ptr());
                _mm512_mask_storeu_ps(out_row[j..].as_mut_ptr(), mask, _mm512_mul_ps(v, vinv));
            }
        }
    }
}

/// # Safety
/// Requires the AVX-512 f32 bundle; shapes as documented on
/// [`scalar::layer_norm`].
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl")]
unsafe fn layer_norm_impl(
    input: &[f32],
    out: &mut [f32],
    rows: usize,
    cols: usize,
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
) {
    // SAFETY: bundle guaranteed by dispatch. All 16-wide accesses are guarded
    // by `j + 16 <= cols` over the length-`cols` row slices and the
    // length-`cols` `gamma` / `beta` (validated by the public wrapper).
    unsafe {
        let inv_cols = 1.0 / cols as f32;
        for r in 0..rows {
            let in_row = &input[r * cols..r * cols + cols];
            let out_row = &mut out[r * cols..r * cols + cols];

            // Pass 1: mean.
            let mut vsum = _mm512_setzero_ps();
            let mut j = 0;
            while j + 16 <= cols {
                vsum = _mm512_add_ps(vsum, _mm512_loadu_ps(in_row[j..].as_ptr()));
                j += 16;
            }
            let mut sum = _mm512_reduce_add_ps(vsum);
            while j < cols {
                sum += in_row[j];
                j += 1;
            }
            let mean = sum * inv_cols;

            // Pass 2: variance (two-pass, matching the scalar oracle).
            let vmean = _mm512_set1_ps(mean);
            let mut vvar = _mm512_setzero_ps();
            let mut j = 0;
            while j + 16 <= cols {
                let d = _mm512_sub_ps(_mm512_loadu_ps(in_row[j..].as_ptr()), vmean);
                vvar = _mm512_fmadd_ps(d, d, vvar);
                j += 16;
            }
            let mut var = _mm512_reduce_add_ps(vvar);
            while j < cols {
                let d = in_row[j] - mean;
                var += d * d;
                j += 1;
            }
            var *= inv_cols;
            let inv_std = 1.0 / (var + eps).sqrt();

            // Pass 3: normalise, scale, shift.
            let vinv_std = _mm512_set1_ps(inv_std);
            let mut j = 0;
            while j + 16 <= cols {
                let d = _mm512_sub_ps(_mm512_loadu_ps(in_row[j..].as_ptr()), vmean);
                let norm = _mm512_mul_ps(d, vinv_std);
                let g = _mm512_loadu_ps(gamma[j..].as_ptr());
                let b = _mm512_loadu_ps(beta[j..].as_ptr());
                _mm512_storeu_ps(out_row[j..].as_mut_ptr(), _mm512_fmadd_ps(norm, g, b));
                j += 16;
            }
            while j < cols {
                out_row[j] = (in_row[j] - mean) * inv_std * gamma[j] + beta[j];
                j += 1;
            }
        }
    }
}

/// # Safety
/// Requires the AVX-512 f32 bundle; `weights.len() == n_mels * n_bins`,
/// `power.len() == n_bins`, `out_log.len() == n_mels` (validated by the
/// dispatch wrapper).
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl")]
unsafe fn fused_logmel_impl(
    weights: &[f32],
    power: &[f32],
    n_mels: usize,
    n_bins: usize,
    floor: f32,
    out_log: &mut [f32],
) {
    // SAFETY: bundle guaranteed by dispatch; every 16-wide load is guarded by
    // `j + 16 <= n_bins` inside the length-`n_bins` row slice of `weights`
    // and `power`; the scalar tail covers the remainder. `out_log[m]` is
    // inside the length-`n_mels` slice.
    unsafe {
        for m in 0..n_mels {
            let row = &weights[m * n_bins..(m + 1) * n_bins];
            let mut vacc = _mm512_setzero_ps();
            let mut j = 0;
            while j + 16 <= n_bins {
                vacc = _mm512_fmadd_ps(
                    _mm512_loadu_ps(row[j..].as_ptr()),
                    _mm512_loadu_ps(power[j..].as_ptr()),
                    vacc,
                );
                j += 16;
            }
            let mut acc = _mm512_reduce_add_ps(vacc);
            while j < n_bins {
                acc += row[j] * power[j];
                j += 1;
            }
            let clamped = if acc > floor { acc } else { floor };
            // `std` log10 — identical to the scalar reference's log10, so the
            // only differential vs the oracle is the accumulation order.
            out_log[m] = clamped.log10();
        }
    }
}

// ---- M5-14-T07: packed-panel micro-kernel ----------------------------------

/// # Safety
/// Requires the AVX-512 f32 bundle (upheld by the dispatch invariant of the
/// safe caller chain); the caller upholds the
/// [`crate::dispatch::GemmMicroKernel`] contract — `ap` carries `kc * 8`
/// packed A elements (`[l][MR = 8]`), `bp` carries `kc * 16` packed B
/// elements (zero-padded past `ncols`), `c` addresses a tile of `rows`
/// (1..=8) valid rows at stride `ldc` × `ncols` (1..=16) valid columns owned
/// exclusively by this call, and `bias` (when `Some`) has ≥ `ncols` elements.
///
/// One `8 × ncols` output tile over packed strips. Per element this is the
/// SAME bias-seeded `_mm512_fmadd_ps` chain over ascending `l` as the legacy
/// [`gemm_impl`] (which is FMA for EVERY column, masked at the tail), so
/// results are bit-identical. `ncols < 16` follows the legacy masked-tail
/// convention: full-width FMA over the zero-padded strip, masked seed and
/// store on the valid lanes only.
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl")]
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)] // micro-kernel signature; explicit tile index math
unsafe fn gemm_micro_packed_impl(
    kc: usize,
    ap: &[f32],
    bp: &[f32],
    c: *mut f32,
    ldc: usize,
    rows: usize,
    ncols: usize,
    bias: Option<&[f32]>,
    accumulate: bool,
) {
    debug_assert!((1..=8).contains(&rows));
    debug_assert!((1..=16).contains(&ncols));
    // SAFETY: the AVX-512 bundle is guaranteed by the caller chain. Slice
    // reads are ordinary indexed accesses within the caller-guaranteed
    // lengths (masked loads touch only the low `ncols` lanes, which stay
    // inside `bias` / the tile rows); the raw `c` accesses touch exactly
    // `rows` rows × `ncols` columns from the tile origin, which the caller
    // owns exclusively.
    unsafe {
        if ncols == 16 {
            let mut acc = [_mm512_setzero_ps(); 8];
            if accumulate {
                for (r, av) in acc.iter_mut().enumerate().take(rows) {
                    *av = _mm512_loadu_ps(c.add(r * ldc));
                }
            } else if let Some(bs) = bias {
                acc = [_mm512_loadu_ps(bs.as_ptr()); 8];
            }
            for l in 0..kc {
                let bl = _mm512_loadu_ps(bp[l * 16..].as_ptr());
                let ab = &ap[l * 8..l * 8 + 8];
                acc[0] = _mm512_fmadd_ps(_mm512_set1_ps(ab[0]), bl, acc[0]);
                acc[1] = _mm512_fmadd_ps(_mm512_set1_ps(ab[1]), bl, acc[1]);
                acc[2] = _mm512_fmadd_ps(_mm512_set1_ps(ab[2]), bl, acc[2]);
                acc[3] = _mm512_fmadd_ps(_mm512_set1_ps(ab[3]), bl, acc[3]);
                acc[4] = _mm512_fmadd_ps(_mm512_set1_ps(ab[4]), bl, acc[4]);
                acc[5] = _mm512_fmadd_ps(_mm512_set1_ps(ab[5]), bl, acc[5]);
                acc[6] = _mm512_fmadd_ps(_mm512_set1_ps(ab[6]), bl, acc[6]);
                acc[7] = _mm512_fmadd_ps(_mm512_set1_ps(ab[7]), bl, acc[7]);
            }
            for (r, av) in acc.iter().enumerate().take(rows) {
                _mm512_storeu_ps(c.add(r * ldc), *av);
            }
        } else {
            // Masked tail strip (legacy convention: FMA chains on the valid
            // lanes; the zero-padded B lanes accumulate garbage that the
            // masked store discards).
            let mask = tail_mask(ncols);
            let mut acc = [_mm512_setzero_ps(); 8];
            if accumulate {
                for (r, av) in acc.iter_mut().enumerate().take(rows) {
                    *av = _mm512_maskz_loadu_ps(mask, c.add(r * ldc));
                }
            } else if let Some(bs) = bias {
                acc = [_mm512_maskz_loadu_ps(mask, bs.as_ptr()); 8];
            }
            for l in 0..kc {
                let bl = _mm512_loadu_ps(bp[l * 16..].as_ptr());
                let ab = &ap[l * 8..l * 8 + 8];
                acc[0] = _mm512_fmadd_ps(_mm512_set1_ps(ab[0]), bl, acc[0]);
                acc[1] = _mm512_fmadd_ps(_mm512_set1_ps(ab[1]), bl, acc[1]);
                acc[2] = _mm512_fmadd_ps(_mm512_set1_ps(ab[2]), bl, acc[2]);
                acc[3] = _mm512_fmadd_ps(_mm512_set1_ps(ab[3]), bl, acc[3]);
                acc[4] = _mm512_fmadd_ps(_mm512_set1_ps(ab[4]), bl, acc[4]);
                acc[5] = _mm512_fmadd_ps(_mm512_set1_ps(ab[5]), bl, acc[5]);
                acc[6] = _mm512_fmadd_ps(_mm512_set1_ps(ab[6]), bl, acc[6]);
                acc[7] = _mm512_fmadd_ps(_mm512_set1_ps(ab[7]), bl, acc[7]);
            }
            for (r, av) in acc.iter().enumerate().take(rows) {
                _mm512_mask_storeu_ps(c.add(r * ldc), mask, *av);
            }
        }
    }
}

/// Packed-panel micro-kernel table entry (plain `unsafe fn`, coercible to
/// [`crate::dispatch::GemmMicroKernel`]).
///
/// # Safety
/// See [`crate::dispatch::GemmMicroKernel`]; additionally the AVX-512
/// dispatch invariant (this entry is only installed after
/// `CpuFeatures::detect` confirmed the F/DQ/BW/VL bundle).
#[allow(clippy::too_many_arguments)] // the micro-kernel's intrinsic parameter set
pub(crate) unsafe fn gemm_micro_packed(
    kc: usize,
    ap: &[f32],
    bp: &[f32],
    c: *mut f32,
    ldc: usize,
    rows: usize,
    ncols: usize,
    bias: Option<&[f32]>,
    accumulate: bool,
) {
    // SAFETY: the AVX-512 bundle is confirmed by the dispatch invariant; the
    // caller upholds the GemmMicroKernel contract.
    unsafe { gemm_micro_packed_impl(kc, ap, bp, c, ldc, rows, ncols, bias, accumulate) }
}

// ---- M5-14-T10: m == 1 row kernel ------------------------------------------

/// # Safety
/// Requires the AVX-512 f32 bundle; contract as on
/// [`crate::dispatch::GemmM1Kernel`] — `b` carries at least
/// `(k-1)*stride + cols` elements, `bias` ≥ `cols`, `out` == `cols`.
///
/// Register-blocked row kernel over [`M1_KB`]-deep k blocks (bounds the live
/// `b` row window per pass; exact f32 round-trip of the output row per
/// block). Per element this is the legacy row-tail chain exactly: bias-seeded
/// `_mm512_fmadd_ps` over ascending `l` for the 16-aligned region and —
/// matching the legacy masked tail — for the final `cols % 16` columns as
/// well (no scalar remainder on AVX-512).
/// k-block depth for the m == 1 row kernel (see the NEON twin's rationale).
const M1_KB: usize = 64;

#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl")]
#[allow(clippy::needless_range_loop)] // explicit k-index math mirrors the legacy kernel
unsafe fn gemm_m1_impl(
    cols: usize,
    k: usize,
    stride: usize,
    a: &[f32],
    b: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    // SAFETY: the AVX-512 bundle is guaranteed by the caller chain. Every
    // 16-wide access is guarded by `j + 16·V <= n16 <= cols`; the masked tail
    // touches only the low `cols - n16` lanes; `b[l*stride + j ..]` stays
    // inside the caller-validated slice, `bias` / `out` accesses stay under
    // `cols`.
    unsafe {
        let n16 = cols & !15usize;
        let mut l0 = 0;
        loop {
            let lend = (l0 + M1_KB).min(k);
            let first = l0 == 0;
            let mut j = 0;
            // 64-column register blocks: 4 live accumulators.
            while j + 64 <= n16 {
                let mut acc = [_mm512_setzero_ps(); 4];
                if first {
                    if let Some(bs) = bias {
                        for (v, av) in acc.iter_mut().enumerate() {
                            *av = _mm512_loadu_ps(bs[j + 16 * v..].as_ptr());
                        }
                    }
                } else {
                    for (v, av) in acc.iter_mut().enumerate() {
                        *av = _mm512_loadu_ps(out[j + 16 * v..].as_ptr());
                    }
                }
                for l in l0..lend {
                    let av = _mm512_set1_ps(a[l]);
                    let base = l * stride + j;
                    acc[0] = _mm512_fmadd_ps(av, _mm512_loadu_ps(b[base..].as_ptr()), acc[0]);
                    acc[1] = _mm512_fmadd_ps(av, _mm512_loadu_ps(b[base + 16..].as_ptr()), acc[1]);
                    acc[2] = _mm512_fmadd_ps(av, _mm512_loadu_ps(b[base + 32..].as_ptr()), acc[2]);
                    acc[3] = _mm512_fmadd_ps(av, _mm512_loadu_ps(b[base + 48..].as_ptr()), acc[3]);
                }
                for (v, av) in acc.iter().enumerate() {
                    _mm512_storeu_ps(out[j + 16 * v..].as_mut_ptr(), *av);
                }
                j += 64;
            }
            // 16-column remainder blocks.
            while j + 16 <= n16 {
                let mut acc = if first {
                    match bias {
                        Some(bs) => _mm512_loadu_ps(bs[j..].as_ptr()),
                        None => _mm512_setzero_ps(),
                    }
                } else {
                    _mm512_loadu_ps(out[j..].as_ptr())
                };
                for l in l0..lend {
                    acc = _mm512_fmadd_ps(
                        _mm512_set1_ps(a[l]),
                        _mm512_loadu_ps(b[l * stride + j..].as_ptr()),
                        acc,
                    );
                }
                _mm512_storeu_ps(out[j..].as_mut_ptr(), acc);
                j += 16;
            }
            // Masked `cols % 16` tail — the legacy FMA chain on the low lanes.
            if j < cols {
                let mask = tail_mask(cols - j);
                let mut acc = if first {
                    match bias {
                        Some(bs) => _mm512_maskz_loadu_ps(mask, bs[j..].as_ptr()),
                        None => _mm512_setzero_ps(),
                    }
                } else {
                    _mm512_maskz_loadu_ps(mask, out[j..].as_ptr())
                };
                for l in l0..lend {
                    acc = _mm512_fmadd_ps(
                        _mm512_set1_ps(a[l]),
                        _mm512_maskz_loadu_ps(mask, b[l * stride + j..].as_ptr()),
                        acc,
                    );
                }
                _mm512_mask_storeu_ps(out[j..].as_mut_ptr(), mask, acc);
            }
            if lend == k {
                break;
            }
            l0 = lend;
        }
    }
}

/// AVX-512 m == 1 GEMM row kernel (dispatch-table entry, M5-14-T10). See
/// [`crate::dispatch::GemmM1Kernel`] for the contract.
pub(crate) fn gemm_m1(
    cols: usize,
    k: usize,
    stride: usize,
    a: &[f32],
    b: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    // SAFETY: the AVX-512 bundle is confirmed by the dispatch invariant (this
    // entry is only installed after `CpuFeatures::detect`); slice bounds are
    // upheld by the driver per the GemmM1Kernel contract and re-checked by
    // slice indexing inside the impl.
    unsafe { gemm_m1_impl(cols, k, stride, a, b, bias, out) }
}

// ---- Safe wrappers installed into the dispatch table (see module docs) ----

/// AVX-512 GEMM. See [`scalar::gemm`] for shapes.
pub(crate) fn gemm(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    // SAFETY: reached only when the AVX-512 f32 bundle was detected on this
    // host (dispatch invariant); slice shapes validated by the public wrapper
    // in `super`.
    unsafe { gemm_impl(m, n, k, a, b, bias, out) }
}

/// AVX-512 GEMV (matrix-vector). See [`scalar::gemv`] for shapes.
pub(crate) fn gemv(
    m: usize,
    k: usize,
    a: &[f32],
    x: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    // SAFETY: reached only when the AVX-512 f32 bundle was detected on this
    // host (dispatch invariant); slice shapes validated by the public wrapper.
    unsafe { gemv_impl(m, k, a, x, bias, out) }
}

/// AVX-512 element-wise add (bit-identical to scalar).
pub(crate) fn add(a: &[f32], b: &[f32], out: &mut [f32]) {
    // SAFETY: reached only when the bundle was detected (dispatch invariant);
    // equal slice lengths validated by the public wrapper.
    unsafe { add_impl(a, b, out) }
}

/// AVX-512 element-wise multiply (bit-identical to scalar).
pub(crate) fn mul(a: &[f32], b: &[f32], out: &mut [f32]) {
    // SAFETY: reached only when the bundle was detected (dispatch invariant);
    // equal slice lengths validated by the public wrapper.
    unsafe { mul_impl(a, b, out) }
}

/// AVX-512 ReLU (bit-identical to scalar).
pub(crate) fn relu(x: &[f32], out: &mut [f32]) {
    // SAFETY: reached only when the bundle was detected (dispatch invariant);
    // equal slice lengths validated by the public wrapper.
    unsafe { relu_impl(x, out) }
}

/// AVX-512 row-wise softmax. See module docs for the scalar-`exp` pass-2.
pub(crate) fn softmax(input: &[f32], out: &mut [f32], rows: usize, cols: usize) {
    // SAFETY: reached only when the bundle was detected (dispatch invariant);
    // `rows * cols` shapes validated by the public wrapper.
    unsafe { softmax_impl(input, out, rows, cols) }
}

/// AVX-512 row-wise layer norm. See [`scalar::layer_norm`] for shapes.
#[allow(clippy::too_many_arguments)] // mirrors the kernel-table signature
pub(crate) fn layer_norm(
    input: &[f32],
    out: &mut [f32],
    rows: usize,
    cols: usize,
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
) {
    // SAFETY: reached only when the bundle was detected (dispatch invariant);
    // shapes validated by the public wrapper.
    unsafe { layer_norm_impl(input, out, rows, cols, gamma, beta, eps) }
}

/// AVX-512 fused log-mel per-frame kernel (M4-17-T09). 16-lane FMA mel-band
/// accumulate + `std` `log10(max(acc, floor))`.
pub(crate) fn fused_logmel(
    weights: &[f32],
    power: &[f32],
    n_mels: usize,
    n_bins: usize,
    floor: f32,
    out_log: &mut [f32],
) {
    assert_eq!(weights.len(), n_mels * n_bins, "weights shape mismatch");
    assert_eq!(power.len(), n_bins, "power length mismatch");
    assert_eq!(out_log.len(), n_mels, "out_log length mismatch");
    // SAFETY: reached only when the bundle was detected (dispatch invariant);
    // shapes asserted above (mirroring the scalar table entry's contract).
    unsafe { fused_logmel_impl(weights, power, n_mels, n_bins, floor, out_log) }
}

// ---- M4-17-T10/T11: VNNI INT8 + BF16 matmul cores ----
//
// These are NOT dispatch-table kernels: they live on the separate
// specialized-tier surface (`super::kquant` / `super::gemm_bf16_on`,
// ADR M4-17 §(b)-2). The INT8 core computes the exact per-group integer sums
// `isum[g] = Σ q_u8[16g+t] · q8[16g+t]` that the shared scalar combine turns
// into f32 (so every INT8 ISA path is bit-identical to the scalar-int8
// reference); the BF16 core is the `vdpbf16ps` dot-product used by
// `gemm_bf16_on`.

/// # Safety
/// Requires `avx512f,avx512bw,avx512vnni`; `q.len() == x.len()`, both a
/// multiple of 64 (whole zmm loads), `sums.len() == q.len() / 16`.
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn vnni_group_sums_impl(q: &[u8], x: &[i8], sums: &mut [i32]) {
    // SAFETY: `avx512vnni` (+F/BW) guaranteed by the caller's `supports`
    // gate. `base + 64 <= q.len() == x.len()` bounds every 64-byte load; the
    // 16 i32 lanes are stored to a stack buffer and folded per 4 lanes into
    // `sums[4 * blk + g]`, `4 * blk + 3 < sums.len()` by the length contract.
    unsafe {
        debug_assert_eq!(q.len(), x.len());
        debug_assert_eq!(q.len() % 64, 0);
        debug_assert_eq!(sums.len() * 16, q.len());
        let mut blk = 0;
        let mut base = 0;
        while base + 64 <= q.len() {
            let qv = _mm512_loadu_si512(q[base..].as_ptr() as *const _);
            let xv = _mm512_loadu_si512(x[base..].as_ptr() as *const _);
            // vpdpbusd: unsigned bytes (weights, 0..63) × signed bytes
            // (activations) accumulated per 4-byte dword into 16 i32 lanes.
            let dp = _mm512_dpbusd_epi32(_mm512_setzero_si512(), qv, xv);
            let mut lanes = [0i32; 16];
            _mm512_storeu_si512(lanes.as_mut_ptr() as *mut _, dp);
            // Each 16-byte group spans 4 consecutive dword lanes; integer
            // adds are exact, so the fold order is irrelevant to parity.
            for g in 0..4 {
                sums[4 * blk + g] =
                    lanes[4 * g] + lanes[4 * g + 1] + lanes[4 * g + 2] + lanes[4 * g + 3];
            }
            blk += 1;
            base += 64;
        }
    }
}

/// AVX-512 VNNI per-group INT8 dot sums (M4-17-T10): `sums[g]` receives
/// `Σ_{t<16} q[16g+t] · x[16g+t]` as an exact i32 — identical to the
/// scalar-int8 reference by integer exactness.
///
/// Caller contract (checked): `q.len() == x.len()`, a multiple of 64
/// (one K-quant super-block = 256 = 4 zmm loads), `sums.len() * 16 == q.len()`.
pub(crate) fn vnni_group_sums(q: &[u8], x: &[i8], sums: &mut [i32]) {
    assert_eq!(q.len(), x.len(), "vnni_group_sums length mismatch");
    assert_eq!(q.len() % 64, 0, "vnni_group_sums needs whole zmm blocks");
    assert_eq!(sums.len() * 16, q.len(), "vnni_group_sums sums mismatch");
    // SAFETY: reached only after `CpuFeatures::supports(Avx512Vnni)` (the
    // caller in `super::kquant` gates on it); lengths asserted above.
    unsafe { vnni_group_sums_impl(q, x, sums) }
}

/// # Safety
/// Requires `avx512f,avx512bf16`; `a.len() == b.len()`, a multiple of 32
/// (whole zmm bf16 loads).
#[target_feature(enable = "avx512f,avx512bf16")]
unsafe fn bf16_dot_impl(a: &[u16], b: &[u16], k: usize) -> f32 {
    // SAFETY: `avx512bf16` (+F) guaranteed by the caller's `supports` gate.
    // `base + 32 <= k == a.len() == b.len()` bounds every 64-byte load; the
    // accumulator is reduced with `_mm512_reduce_add_ps` (deterministic
    // shuffle tree).
    unsafe {
        debug_assert_eq!(a.len(), k);
        debug_assert_eq!(b.len(), k);
        debug_assert_eq!(k % 32, 0);
        let mut acc = _mm512_setzero_ps();
        let mut base = 0;
        while base + 32 <= k {
            // 32 bf16 values per zmm, reinterpreted as `__m512bh`.
            let av = _mm512_loadu_si512(a[base..].as_ptr() as *const _);
            let bv = _mm512_loadu_si512(b[base..].as_ptr() as *const _);
            acc = _mm512_dpbf16_ps(
                acc,
                core::mem::transmute::<__m512i, __m512bh>(av),
                core::mem::transmute::<__m512i, __m512bh>(bv),
            );
            base += 32;
        }
        _mm512_reduce_add_ps(acc)
    }
}

/// AVX-512 BF16 dot product (M4-17-T11): `Σ a[l]·b[l]` over `k` bf16 values
/// (bit patterns in `u16`), f32 accumulate via `vdpbf16ps`.
///
/// The exact internal pair-accumulation order of `vdpbf16ps` is NOT asserted
/// by the parity tests (no local bf16 silicon; ADR M4-17 §(f)) — callers
/// compare against the architectural bf16 bound only. `k` must be a multiple
/// of 32; the caller (`super::kquant::gemm_bf16_on`) zero-pads (bf16 zero
/// products are exact zeros).
pub(crate) fn bf16_dot(a: &[u16], b: &[u16]) -> f32 {
    assert_eq!(a.len(), b.len(), "bf16_dot length mismatch");
    assert_eq!(a.len() % 32, 0, "bf16_dot needs whole zmm blocks");
    // SAFETY: reached only after `CpuFeatures::supports(Avx512Bf16)` (the
    // caller gates on it); lengths asserted above.
    unsafe { bf16_dot_impl(a, b, a.len()) }
}
