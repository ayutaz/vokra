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
//! `gemm` is a **register-blocked** microkernel (M1-08): an `MR`×`NR` output
//! tile is held in `MR * NR / 4` independent `float32x4_t` accumulators (see
//! [`MR`] / [`NR_VEC`]), so the `k`-loop keeps many FMA chains in flight and
//! is no longer FMA-latency-bound — the win for the `m = 1500` Whisper encoder
//! GEMMs. It stays a pure reordering of the same per-element FMA chains as the
//! scalar oracle (lane-aligned shapes are bit-identical to the pre-blocking
//! kernel), and the row/column remainders fall back to the original single-row
//! vector + scalar paths.
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

/// Register-block tile height (rows of `a` / `out` computed together).
///
/// `MR = 8` rows × `NR = 8` cols (two 4-lane vectors) holds `8 * 8 / 4 = 16`
/// live `float32x4_t` accumulators, comfortably inside AArch64's 32 SIMD
/// registers (16 acc + 2 `b` vectors + 1 `a` broadcast ≈ 19). This is the
/// **tunable** default: it gives 16 independent FMA chains, enough to cover
/// the M1 FMA latency×throughput and lift the encoder GEMMs off the old
/// single-accumulator, FMA-latency-bound path. See [`gemm_impl`].
const MR: usize = 8;
/// Register-block tile width in 4-lane NEON vectors (`NR = NR_VEC * 4 = 8`).
const NR_VEC: usize = 2;

/// # Safety
/// Requires `neon` (baseline on AArch64); shapes as documented on
/// [`scalar::gemm`].
///
/// Register-blocked `MR`×`NR` microkernel (`NR = NR_VEC * 4`): each B-load is
/// reused across the `MR` rows of the tile and each A-broadcast across the
/// `NR_VEC` B-vectors, so the `k`-loop runs `MR * NR_VEC` **independent**
/// accumulators and hides FMA latency. Every output element is still the same
/// bias-seeded FMA chain over `l = 0..k` as the scalar oracle (just tiled), so
/// results stay within the GEMM differential tolerance and — on lane-aligned
/// shapes — bit-identical to the pre-blocking NEON kernel.
#[target_feature(enable = "neon")]
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
    // SAFETY: NEON is baseline on AArch64. In the row-block loop `i + MR <= m`
    // and `r < MR`, so `i + r <= m - 1`; the column guards (`j + 8 <= n`,
    // `j + 4 <= n`, `j < n`) keep every 4-wide load/store inside the length-`n`
    // rows of `b` / `out` / `bias` and every `a[(i + r) * k + l]` inside `a`
    // (all lengths validated by the public wrapper). The row-tail loop repeats
    // the original single-row path with the same guards.
    unsafe {
        let mut i = 0;
        // ---- main path: full blocks of MR rows ----
        while i + MR <= m {
            let mut j = 0;
            // NR = NR_VEC * 4 columns: an MR × NR_VEC accumulator tile.
            while j + NR_VEC * 4 <= n {
                let (mut c0, mut c1) = match bias {
                    Some(bias) => (
                        [vld1q_f32(bias[j..].as_ptr()); MR],
                        [vld1q_f32(bias[j + 4..].as_ptr()); MR],
                    ),
                    None => ([vdupq_n_f32(0.0); MR], [vdupq_n_f32(0.0); MR]),
                };
                for l in 0..k {
                    let bl0 = vld1q_f32(b[l * n + j..].as_ptr());
                    let bl1 = vld1q_f32(b[l * n + j + 4..].as_ptr());
                    for r in 0..MR {
                        let ar = vdupq_n_f32(a[(i + r) * k + l]);
                        c0[r] = vfmaq_f32(c0[r], ar, bl0);
                        c1[r] = vfmaq_f32(c1[r], ar, bl1);
                    }
                }
                for r in 0..MR {
                    vst1q_f32(out[(i + r) * n + j..].as_mut_ptr(), c0[r]);
                    vst1q_f32(out[(i + r) * n + j + 4..].as_mut_ptr(), c1[r]);
                }
                j += NR_VEC * 4;
            }
            // 4-wide column remainder: an MR × 1 accumulator tile.
            while j + 4 <= n {
                let mut c = match bias {
                    Some(bias) => [vld1q_f32(bias[j..].as_ptr()); MR],
                    None => [vdupq_n_f32(0.0); MR],
                };
                for l in 0..k {
                    let bl = vld1q_f32(b[l * n + j..].as_ptr());
                    for r in 0..MR {
                        c[r] = vfmaq_f32(c[r], vdupq_n_f32(a[(i + r) * k + l]), bl);
                    }
                }
                for r in 0..MR {
                    vst1q_f32(out[(i + r) * n + j..].as_mut_ptr(), c[r]);
                }
                j += 4;
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
                vst1q_f32(out[i * n + j..].as_mut_ptr(), acc);
                j += 4;
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
/// Requires `neon` (baseline); shapes as documented on [`scalar::gemv`].
///
/// Per output row `i`, the dot product `sum_l a[i, l] * x[l]` is computed with
/// four independent 4-lane FMA accumulators (16 lanes per iteration) so the
/// `k`-loop keeps several FMA chains in flight and is not latency-bound, then
/// reduced by a tree add + horizontal sum; the `k % 16` (4-wide) and `k % 4`
/// (scalar) remainders follow. This is the same arithmetic as
/// [`scalar::gemv`] with a reordered reduction (within the gemv differential
/// tolerance), and it streams the `[m, k]` matrix `a` row-contiguously — the
/// win over routing the tied logits head through the `gemm` `n = 1` scalar
/// tail.
#[target_feature(enable = "neon")]
unsafe fn gemv_impl(
    m: usize,
    k: usize,
    a: &[f32],
    x: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    // SAFETY: NEON is baseline on AArch64. For each row `i < m`, `base = i * k`
    // and every 4-wide load is guarded by `l + 16 <= k` / `l + 4 <= k`, so the
    // top index `base + l + 15` (resp. `+ 3`) stays inside the length-`m * k`
    // slice `a` and each `x[l + ..]` inside the length-`k` slice `x` (both
    // validated by the public wrapper). `out[i]` and `bias[i]` are inside their
    // length-`m` slices.
    unsafe {
        for i in 0..m {
            let base = i * k;
            let mut acc0 = vdupq_n_f32(0.0);
            let mut acc1 = vdupq_n_f32(0.0);
            let mut acc2 = vdupq_n_f32(0.0);
            let mut acc3 = vdupq_n_f32(0.0);
            let mut l = 0;
            while l + 16 <= k {
                acc0 = vfmaq_f32(
                    acc0,
                    vld1q_f32(a[base + l..].as_ptr()),
                    vld1q_f32(x[l..].as_ptr()),
                );
                acc1 = vfmaq_f32(
                    acc1,
                    vld1q_f32(a[base + l + 4..].as_ptr()),
                    vld1q_f32(x[l + 4..].as_ptr()),
                );
                acc2 = vfmaq_f32(
                    acc2,
                    vld1q_f32(a[base + l + 8..].as_ptr()),
                    vld1q_f32(x[l + 8..].as_ptr()),
                );
                acc3 = vfmaq_f32(
                    acc3,
                    vld1q_f32(a[base + l + 12..].as_ptr()),
                    vld1q_f32(x[l + 12..].as_ptr()),
                );
                l += 16;
            }
            // 4-wide remainder folds into the first accumulator.
            while l + 4 <= k {
                acc0 = vfmaq_f32(
                    acc0,
                    vld1q_f32(a[base + l..].as_ptr()),
                    vld1q_f32(x[l..].as_ptr()),
                );
                l += 4;
            }
            let acc = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
            let mut s = vaddvq_f32(acc);
            // Scalar `k % 4` tail.
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

// ---- M5-14-T05/T08: packed-panel micro-kernel ------------------------------

/// # Safety
/// Requires `neon` (baseline on AArch64); the caller upholds the
/// [`crate::dispatch::GemmMicroKernel`] contract — `ap` carries `kc * 8`
/// packed A elements (`[l][MR = 8]` layout), `bp` carries `kc * w` packed B
/// elements where `w = ncols ∈ {8, 4}`, `c` addresses a tile of `rows`
/// (1..=8) valid rows at stride `ldc` × `ncols` valid columns owned
/// exclusively by this call, and `bias` (when `Some`) has ≥ `ncols` elements.
///
/// One `8 × ncols` output tile over a packed strip pair. Per element this is
/// the SAME bias-seeded fused-FMA chain over ascending `l` as the legacy
/// [`gemm_impl`] vector region — `vfmaq_laneq_f32(acc, b, a, LANE)` computes
/// `acc + b·a[LANE]` with single rounding exactly like the legacy
/// `vfmaq_f32(acc, vdupq_n_f32(a), b)` — so results are bit-identical; only
/// the operand addresses (packed, unit-stride) differ, which is what removes
/// the power-of-two `n`-stride L1 aliasing (Wave-0 finding D2-1). A-rows past
/// `rows` are zero-padded by the pack and computed but never stored; with
/// `accumulate` the tile continues the k-chain from the partials in `c`
/// (exact f32 round-trip).
#[target_feature(enable = "neon")]
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
    debug_assert!(ncols == 8 || ncols == 4);
    // SAFETY: NEON is baseline on AArch64. Slice reads (`ap` / `bp` / `bias`)
    // are ordinary indexed accesses within the caller-guaranteed lengths; the
    // raw `c` loads/stores touch exactly `rows` rows × `ncols` columns from
    // the tile origin, which the caller owns exclusively.
    unsafe {
        if ncols == 8 {
            let zero = vdupq_n_f32(0.0);
            let (mut c0, mut c1) = ([zero; 8], [zero; 8]);
            if accumulate {
                for r in 0..rows {
                    c0[r] = vld1q_f32(c.add(r * ldc));
                    c1[r] = vld1q_f32(c.add(r * ldc + 4));
                }
            } else if let Some(bs) = bias {
                let b0 = vld1q_f32(bs.as_ptr());
                let b1 = vld1q_f32(bs[4..].as_ptr());
                c0 = [b0; 8];
                c1 = [b1; 8];
            }
            for l in 0..kc {
                let bl0 = vld1q_f32(bp[l * 8..].as_ptr());
                let bl1 = vld1q_f32(bp[l * 8 + 4..].as_ptr());
                let a03 = vld1q_f32(ap[l * 8..].as_ptr());
                let a47 = vld1q_f32(ap[l * 8 + 4..].as_ptr());
                c0[0] = vfmaq_laneq_f32::<0>(c0[0], bl0, a03);
                c1[0] = vfmaq_laneq_f32::<0>(c1[0], bl1, a03);
                c0[1] = vfmaq_laneq_f32::<1>(c0[1], bl0, a03);
                c1[1] = vfmaq_laneq_f32::<1>(c1[1], bl1, a03);
                c0[2] = vfmaq_laneq_f32::<2>(c0[2], bl0, a03);
                c1[2] = vfmaq_laneq_f32::<2>(c1[2], bl1, a03);
                c0[3] = vfmaq_laneq_f32::<3>(c0[3], bl0, a03);
                c1[3] = vfmaq_laneq_f32::<3>(c1[3], bl1, a03);
                c0[4] = vfmaq_laneq_f32::<0>(c0[4], bl0, a47);
                c1[4] = vfmaq_laneq_f32::<0>(c1[4], bl1, a47);
                c0[5] = vfmaq_laneq_f32::<1>(c0[5], bl0, a47);
                c1[5] = vfmaq_laneq_f32::<1>(c1[5], bl1, a47);
                c0[6] = vfmaq_laneq_f32::<2>(c0[6], bl0, a47);
                c1[6] = vfmaq_laneq_f32::<2>(c1[6], bl1, a47);
                c0[7] = vfmaq_laneq_f32::<3>(c0[7], bl0, a47);
                c1[7] = vfmaq_laneq_f32::<3>(c1[7], bl1, a47);
            }
            for r in 0..rows {
                vst1q_f32(c.add(r * ldc), c0[r]);
                vst1q_f32(c.add(r * ldc + 4), c1[r]);
            }
        } else {
            // ncols == 4: the packed counterpart of the legacy 4-wide column
            // remainder (covers [n8, n4); `bp` is packed at width 4).
            let zero = vdupq_n_f32(0.0);
            let mut c0 = [zero; 8];
            if accumulate {
                for r in 0..rows {
                    c0[r] = vld1q_f32(c.add(r * ldc));
                }
            } else if let Some(bs) = bias {
                c0 = [vld1q_f32(bs.as_ptr()); 8];
            }
            for l in 0..kc {
                let bl = vld1q_f32(bp[l * 4..].as_ptr());
                let a03 = vld1q_f32(ap[l * 8..].as_ptr());
                let a47 = vld1q_f32(ap[l * 8 + 4..].as_ptr());
                c0[0] = vfmaq_laneq_f32::<0>(c0[0], bl, a03);
                c0[1] = vfmaq_laneq_f32::<1>(c0[1], bl, a03);
                c0[2] = vfmaq_laneq_f32::<2>(c0[2], bl, a03);
                c0[3] = vfmaq_laneq_f32::<3>(c0[3], bl, a03);
                c0[4] = vfmaq_laneq_f32::<0>(c0[4], bl, a47);
                c0[5] = vfmaq_laneq_f32::<1>(c0[5], bl, a47);
                c0[6] = vfmaq_laneq_f32::<2>(c0[6], bl, a47);
                c0[7] = vfmaq_laneq_f32::<3>(c0[7], bl, a47);
            }
            for r in 0..rows {
                vst1q_f32(c.add(r * ldc), c0[r]);
            }
        }
    }
}

/// Packed-panel micro-kernel table entry (plain `unsafe fn`, coercible to
/// [`crate::dispatch::GemmMicroKernel`]).
///
/// # Safety
/// See [`crate::dispatch::GemmMicroKernel`]; NEON availability is the module
/// compile gate (AArch64 baseline).
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
    // SAFETY: NEON is baseline on AArch64 (this module only compiles there);
    // the caller upholds the GemmMicroKernel contract.
    unsafe { gemm_micro_packed_impl(kc, ap, bp, c, ldc, rows, ncols, bias, accumulate) }
}

// ---- M5-14-T10: m == 1 row kernel ------------------------------------------

/// k-block depth for the m == 1 row kernel: bounds the live `b` row window
/// per pass (`LB` pages when the row stride spans a page, e.g. n = 4096 =
/// one 16 KiB page per row) so the strided walk stays inside the L1 dTLB and
/// the prefetcher's stream budget, at the cost of re-reading the output row
/// once per block (`2·4·cols·k/LB` bytes ≪ `b` itself). Blocking the k loop
/// does NOT change per-element accumulation order (ascending `l` with an
/// exact f32 store/reload between blocks).
const M1_KB: usize = 64;

/// # Safety
/// Requires `neon` (baseline); contract as on
/// [`crate::dispatch::GemmM1Kernel`] — `b` carries at least
/// `(k-1)*stride + cols` elements, `bias` ≥ `cols`, `out` == `cols`.
///
/// Register-blocked row kernel: 32-column output blocks held in 8
/// accumulators across a [`M1_KB`]-deep k block, `b` walked row-by-row inside
/// the block. Per element this is the legacy row-tail chain exactly:
/// bias-seeded `vfmaq_f32` over ascending `l` for the 4-aligned column
/// region, plain mul+add for the `cols % 4` scalar tail — bit-identical to
/// [`gemm_impl`]'s `m == 1` row (k-blocking only round-trips the f32 partial
/// through `out`, which is exact).
#[target_feature(enable = "neon")]
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
    // SAFETY: NEON is baseline on AArch64. Every 4-wide access is guarded by
    // `j + 4·V <= n4 <= cols`; `b[l*stride + j ..]` stays inside the
    // caller-validated `b` slice ((k-1)*stride + cols elements), `bias` /
    // `out` accesses stay under `cols`.
    unsafe {
        let n4 = cols & !3usize;
        let mut l0 = 0;
        loop {
            let lend = (l0 + M1_KB).min(k);
            let first = l0 == 0;
            let mut j = 0;
            // 32-column register blocks: 8 live accumulators.
            while j + 32 <= n4 {
                let mut acc = [vdupq_n_f32(0.0); 8];
                if first {
                    if let Some(bs) = bias {
                        for (v, av) in acc.iter_mut().enumerate() {
                            *av = vld1q_f32(bs[j + 4 * v..].as_ptr());
                        }
                    }
                } else {
                    for (v, av) in acc.iter_mut().enumerate() {
                        *av = vld1q_f32(out[j + 4 * v..].as_ptr());
                    }
                }
                for l in l0..lend {
                    let av = vdupq_n_f32(a[l]);
                    let base = l * stride + j;
                    acc[0] = vfmaq_f32(acc[0], av, vld1q_f32(b[base..].as_ptr()));
                    acc[1] = vfmaq_f32(acc[1], av, vld1q_f32(b[base + 4..].as_ptr()));
                    acc[2] = vfmaq_f32(acc[2], av, vld1q_f32(b[base + 8..].as_ptr()));
                    acc[3] = vfmaq_f32(acc[3], av, vld1q_f32(b[base + 12..].as_ptr()));
                    acc[4] = vfmaq_f32(acc[4], av, vld1q_f32(b[base + 16..].as_ptr()));
                    acc[5] = vfmaq_f32(acc[5], av, vld1q_f32(b[base + 20..].as_ptr()));
                    acc[6] = vfmaq_f32(acc[6], av, vld1q_f32(b[base + 24..].as_ptr()));
                    acc[7] = vfmaq_f32(acc[7], av, vld1q_f32(b[base + 28..].as_ptr()));
                }
                for (v, av) in acc.iter().enumerate() {
                    vst1q_f32(out[j + 4 * v..].as_mut_ptr(), *av);
                }
                j += 32;
            }
            // 4-column remainder blocks.
            while j + 4 <= n4 {
                let mut acc = if first {
                    match bias {
                        Some(bs) => vld1q_f32(bs[j..].as_ptr()),
                        None => vdupq_n_f32(0.0),
                    }
                } else {
                    vld1q_f32(out[j..].as_ptr())
                };
                for l in l0..lend {
                    acc = vfmaq_f32(
                        acc,
                        vdupq_n_f32(a[l]),
                        vld1q_f32(b[l * stride + j..].as_ptr()),
                    );
                }
                vst1q_f32(out[j..].as_mut_ptr(), acc);
                j += 4;
            }
            // Scalar `cols % 4` tail — the legacy plain mul+add chain.
            while j < cols {
                let mut s = if first {
                    bias.map_or(0.0, |bs| bs[j])
                } else {
                    out[j]
                };
                for l in l0..lend {
                    s += a[l] * b[l * stride + j];
                }
                out[j] = s;
                j += 1;
            }
            if lend == k {
                break;
            }
            l0 = lend;
        }
    }
}

/// NEON m == 1 GEMM row kernel (dispatch-table entry, M5-14-T10). See
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
    // SAFETY: NEON is baseline on AArch64 (this module only compiles there);
    // slice bounds are upheld by the driver per the GemmM1Kernel contract and
    // re-checked by slice indexing inside the impl.
    unsafe { gemm_m1_impl(cols, k, stride, a, b, bias, out) }
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

/// NEON GEMV (matrix-vector). See [`scalar::gemv`] for shapes.
pub(crate) fn gemv(
    m: usize,
    k: usize,
    a: &[f32],
    x: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    // SAFETY: NEON is baseline on AArch64 (this module only compiles there);
    // slice shapes validated by the public wrapper in `super`.
    unsafe { gemv_impl(m, k, a, x, bias, out) }
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
