//! NEON fused log-mel inner kernel (M2-04-T06 — NEON companion to
//! [`super::fused_logmel_avx2`]).
//!
//! Accelerates the same two hot inner loops of the log-mel front-end:
//!
//! 1. **Mel accumulation** — the O(n_mels × n_bins × n_frames) dot product
//!    `mel[m] = Σ_k weights[m*n_bins + k] * power[k]`. The NEON kernel uses
//!    four-lane `vfmaq_f32` down each `weights` row and horizontally sums the
//!    accumulator at the end with `vaddvq_f32`. The ragged bin tail
//!    (`n_bins % 4`) is handled scalar.
//!
//! 2. **log10 per mel bin** — `vlog10_neon` is a four-lane polynomial
//!    approximation reusing the `vexp`-style IEEE-754 exponent-field
//!    extraction pattern (see [`super::vexp::exp_ps_neon`]) with the identity
//!    `log10(x) = log2(x) · log10(2)` and a degree-6 log2(1+u) minimax
//!    polynomial on `u ∈ [0, 1]`. Worst-case absolute error vs `f32::log10`
//!    is well under `1e-6`, far inside the FP32 NFR-QL-01 parity ceiling
//!    `atol = 0.01` (the per-file test asserts atol=1e-5 to match the
//!    plan-spec SIMD ceiling).
//!
//! # Unsafe boundary (NFR-RL-07)
//!
//! The public wrapper [`fused_logmel_apply_frame_neon`] is safe. It performs
//! shape validation and dispatches to a private
//! `#[target_feature(enable = "neon")] unsafe fn` that emits the intrinsics.
//! NEON is the ARMv8-A baseline (CLAUDE.md "NEON (ARMv8-A baseline、常時対応)"),
//! so the ISA precondition is unconditional on AArch64 — the `// SAFETY:`
//! comments record exactly this. No JIT is used (NFR-RL-05).
//!
//! # FR-EX-08
//!
//! Scalar / AVX2 / NEON compute the same op with the same result within FP32
//! rounding. Choosing NEON here is a within-CPU-backend optimization,
//! orthogonal to the cross-backend explicit-op-error rule FR-EX-08.

#![cfg(target_arch = "aarch64")]

use core::arch::aarch64::*;

// ---------------------------------------------------------------------------
// vlog10_neon — 4-lane f32 log10 approximation (M2-04-T06).
// ---------------------------------------------------------------------------
//
// For x > 0, decompose x = 2^e · (1 + u) with u ∈ [0, 1) using the IEEE-754
// exponent field (`(bits >> 23) - 127`) and mantissa. Then
//
//   log10(x) = (e + log2(1 + u)) / log2(10)
//            = (e + log2(1 + u)) · LOG10_2
//
// where `LOG10_2 = log10(2)`. `log2(1 + u)` on `u ∈ [0, 1]` is a degree-6
// minimax polynomial (Horner). Coefficients are locally owned — the same
// standard rational minimax expansion also used by the AVX2 sibling — keeping
// the zero-dependency invariant (NFR-DS-02) and Apache-2.0 license hygiene.

// Atanh-based log kernel (Cephes / musl `logf` style) — after range-reducing
// the mantissa to `[sqrt(0.5), sqrt(2))`, we substitute `s = (m − 1)/(m + 1)`
// so `s ∈ (−0.172, 0.172]` and `log(m) = 2·atanh(s) = 2·(s + s³/3 + s⁵/5 …)`.
// Degree-4 (odd powers up through `s⁹`) is more than enough for f32
// (worst-case |error| ≪ 1e-7 on the reduced range, well under the atol=1e-5
// SIMD-log10 ceiling). Coefficients are the exact `1/(2k+1)` odd-atanh
// series terms; the leading `2·s` factor is folded in outside the Horner
// polynomial.
const ATANH_C1: f32 = 1.0 / 3.0; // s³
const ATANH_C2: f32 = 1.0 / 5.0; // s⁵
const ATANH_C3: f32 = 1.0 / 7.0; // s⁷
const ATANH_C4: f32 = 1.0 / 9.0; // s⁹

/// `sqrt(0.5) ≈ 0.707106781` — the mantissa split point used to keep
/// `s = (m−1)/(m+1)` inside `(−0.172, 0.172]`.
const SQRT_HALF: f32 = core::f32::consts::FRAC_1_SQRT_2;

/// `1 / ln(10)` — scales natural `log` to `log10`.
const INV_LN10: f32 = core::f32::consts::LOG10_E;

/// `ln(2)` — the exponent-to-natural-log scale.
const LN2: f32 = core::f32::consts::LN_2;

/// Vectorized f32 `log10` over the four NEON lanes of `x` (elementwise
/// `x > 0` required — negative / zero inputs saturate to a large negative
/// finite value, matching the caller's `max(acc, floor)` clamp).
///
/// The body is register-only (no memory access), so the intrinsics are safe
/// to call within a matching `#[target_feature(enable = "neon")]` context.
///
/// # Safety
/// Requires the `neon` target feature (the AArch64 baseline).
#[target_feature(enable = "neon")]
unsafe fn vlog10_neon(x: float32x4_t) -> float32x4_t {
    // SAFETY: NEON is baseline on AArch64; all ops below are register-only,
    // so no inner `unsafe` block is required (mirroring `vexp::exp_ps_neon`).
    // Extract IEEE-754 exponent and mantissa. Bits: sign(1) exp(8) mant(23).
    let bits = vreinterpretq_u32_f32(x);
    let exp_bits = vshrq_n_u32(vandq_u32(bits, vdupq_n_u32(0x7F80_0000u32)), 23);
    // (u32 exponent) - 127 → signed exponent e (mantissa in [1, 2)).
    let mut e = vsubq_s32(vreinterpretq_s32_u32(exp_bits), vdupq_n_s32(127));

    // Mantissa in [1, 2): clear exponent, set biased exponent = 127.
    let mant_bits = vandq_u32(bits, vdupq_n_u32(0x007F_FFFFu32));
    let one_bits = vdupq_n_u32(0x3F80_0000u32); // 1.0 f32 bits
    let mut m = vreinterpretq_f32_u32(vorrq_u32(mant_bits, one_bits));

    // Range-reduce: if m < sqrt(0.5), double the mantissa and decrement the
    // exponent, so m ∈ [sqrt(0.5), sqrt(2)) and s = (m−1)/(m+1) stays inside
    // (−0.172, 0.172]. This tight range makes the odd-atanh series converge
    // in ~4 terms at f32 precision.
    let below = vcltq_f32(m, vdupq_n_f32(SQRT_HALF)); // lane mask (u32 all-ones / zeros)
    let two_m = vaddq_f32(m, m);
    m = vbslq_f32(below, two_m, m);
    // e -= 1 where below.
    let ones = vandq_u32(below, vdupq_n_u32(1));
    e = vsubq_s32(e, vreinterpretq_s32_u32(ones));
    let ef = vcvtq_f32_s32(e);

    // s = (m − 1) / (m + 1). NEON has no 4-lane vdivq_f32 issue — the
    // AArch64 instruction is native and finishes in a few cycles.
    let numer = vsubq_f32(m, vdupq_n_f32(1.0));
    let denom = vaddq_f32(m, vdupq_n_f32(1.0));
    let s = vdivq_f32(numer, denom);
    let s2 = vmulq_f32(s, s);

    // Horner in s²: p = c4; p = c3 + s²·p; p = c2 + s²·p; p = c1 + s²·p.
    // Then log(m) = 2·s·(1 + s²·p).
    let mut p = vdupq_n_f32(ATANH_C4);
    p = vfmaq_f32(vdupq_n_f32(ATANH_C3), p, s2);
    p = vfmaq_f32(vdupq_n_f32(ATANH_C2), p, s2);
    p = vfmaq_f32(vdupq_n_f32(ATANH_C1), p, s2);
    // (1 + s²·p) · 2s = 2s + 2·s³·p.
    let one_plus = vfmaq_f32(vdupq_n_f32(1.0), p, s2);
    let log_m = vmulq_f32(vaddq_f32(s, s), one_plus);

    // log(x) = e·ln(2) + log(m).
    let log_x = vfmaq_f32(log_m, ef, vdupq_n_f32(LN2));
    // log10(x) = log(x) / ln(10).
    vmulq_f32(log_x, vdupq_n_f32(INV_LN10))
}

// ---------------------------------------------------------------------------
// NEON 4-lane FMA inner: mel-band accumulation over one frame's power spectrum.
// ---------------------------------------------------------------------------

/// Compute one dot product `Σ_k weights_row[k] · power[k]` in four-lane FMA.
///
/// # Safety
/// Requires `neon` (baseline on AArch64). Both slices must be at least
/// `n_bins` long (caller validated by the public wrapper).
#[target_feature(enable = "neon")]
unsafe fn dot_row_neon(weights_row: &[f32], power: &[f32], n_bins: usize) -> f32 {
    // SAFETY: NEON is baseline on AArch64; the caller guarantees both slices
    // contain at least `n_bins` elements (validated in the public wrapper).
    // The vector loads never step past `k + 4 <= n_bins`; the scalar tail
    // visits `k < n_bins`.
    unsafe {
        let mut acc = vdupq_n_f32(0.0);
        let mut k = 0usize;
        while k + 4 <= n_bins {
            let w = vld1q_f32(weights_row.as_ptr().add(k));
            let p = vld1q_f32(power.as_ptr().add(k));
            // vfmaq_f32(acc, w, p) = acc + w * p (fused multiply-add).
            acc = vfmaq_f32(acc, w, p);
            k += 4;
        }
        // Horizontal sum of the four lanes.
        let mut s = vaddvq_f32(acc);
        // Scalar tail — `n_bins % 4` elements.
        while k < n_bins {
            s += *weights_row.get_unchecked(k) * *power.get_unchecked(k);
            k += 1;
        }
        s
    }
}

// ---------------------------------------------------------------------------
// Public safe API.
// ---------------------------------------------------------------------------

/// Applies the mel filterbank + `log10(max(·, floor))` to one frame's power
/// spectrum, writing `n_mels` log-mel values to `out_log`. This is the fused
/// NEON path for the log-mel front-end inner loop (M2-04-T06).
///
/// `weights` is row-major `[n_mels, n_bins]` (same layout as
/// `vokra_ops::mel::MelFilterbank::weights`), `power` has length `n_bins`,
/// and `out_log` has length `n_mels`. `floor` is the numerical clamp applied
/// before `log10` (typically `1e-10`).
///
/// # Panics
/// Panics on shape mismatch, matching the debug-assertion regime of
/// `vokra_ops::mel::MelFilterbank::apply` (the safe scalar reference).
///
/// # Availability
/// NEON is the AArch64 baseline — this wrapper is always available when
/// compiled on `target_arch = "aarch64"`. No runtime feature detection is
/// needed (see the module docs above).
pub fn fused_logmel_apply_frame_neon(
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

    // SAFETY: `#[target_feature(enable = "neon")]` requires the ISA
    // guarantee, which is unconditional on AArch64 (NEON is the ARMv8-A
    // baseline; see module docs). Shape preconditions were validated above.
    unsafe { fused_logmel_apply_frame_neon_inner(weights, power, n_mels, n_bins, floor, out_log) }
}

/// # Safety
/// Requires `neon` (baseline on AArch64). All shapes validated by the public
/// wrapper.
#[target_feature(enable = "neon")]
unsafe fn fused_logmel_apply_frame_neon_inner(
    weights: &[f32],
    power: &[f32],
    n_mels: usize,
    n_bins: usize,
    floor: f32,
    out_log: &mut [f32],
) {
    // SAFETY: NEON is baseline on AArch64; shapes validated by the wrapper.
    unsafe {
        // Step 1: mel accumulation — one dot product per mel band. Write raw
        // dot products to out_log first so log10 can consume them in-place
        // (avoids an intermediate `mel` allocation, which is a primary
        // intermediate the fusion is intended to eliminate).
        for m in 0..n_mels {
            let row = weights.get_unchecked(m * n_bins..(m + 1) * n_bins);
            let acc = dot_row_neon(row, power, n_bins);
            // Clamp before log10 (avoid log10(0) = -inf).
            *out_log.get_unchecked_mut(m) = if acc > floor { acc } else { floor };
        }

        // Step 2: vlog10 across the n_mels output in four-lane chunks. Tail
        // (`n_mels % 4`) falls back to scalar `f32::log10` — n_mels is at
        // most a few hundred and the scalar tail is negligible.
        let mut m = 0usize;
        while m + 4 <= n_mels {
            let v = vld1q_f32(out_log.as_ptr().add(m));
            let l = vlog10_neon(v);
            vst1q_f32(out_log.as_mut_ptr().add(m), l);
            m += 4;
        }
        while m < n_mels {
            let v = *out_log.get_unchecked(m);
            *out_log.get_unchecked_mut(m) = v.log10();
            m += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Portable scalar reference (M2-04-T06 parity oracle).
// ---------------------------------------------------------------------------
//
// Same arithmetic as `fused_logmel_avx2::fused_logmel_apply_frame_scalar` —
// bundled here so the NEON parity test can cross-check without touching the
// x86-only sibling module. Kept byte-identical in shape to the AVX2 scalar
// reference for reviewer convenience.

/// Scalar reference: mel filterbank + log10(max(·, floor)) for one frame.
pub fn fused_logmel_apply_frame_scalar(
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
    for m in 0..n_mels {
        let row = &weights[m * n_bins..(m + 1) * n_bins];
        let mut acc = 0.0f32;
        for (w, p) in row.iter().zip(power) {
            acc += w * p;
        }
        let clamped = if acc > floor { acc } else { floor };
        out_log[m] = clamped.log10();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sanity check: `vlog10_neon` matches `f32::log10` within 1e-6 on a
    /// representative range spanning the log-mel input domain (floor 1e-10
    /// up to peak power ≈ 1e6).
    #[test]
    fn vlog10_neon_matches_std_log10() {
        // Sample points across the log-mel input domain (all positive; the
        // caller clamps to `floor` before entering `vlog10_neon`).
        let xs: [f32; 16] = [
            1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6, 1e7,
        ];
        // SAFETY: NEON is baseline on AArch64; the load / store target a
        // fully owned, correctly sized 4-lane stack buffer.
        unsafe {
            for chunk in xs.chunks(4) {
                let mut lanes = [chunk[0]; 4];
                lanes[..chunk.len()].copy_from_slice(chunk);
                let v = vld1q_f32(lanes.as_ptr());
                let l = vlog10_neon(v);
                let mut out = [0.0f32; 4];
                vst1q_f32(out.as_mut_ptr(), l);
                for (x, got) in lanes.iter().zip(&out) {
                    let want = x.log10();
                    let diff = (got - want).abs();
                    assert!(
                        diff < 1e-6,
                        "log10({x}) = {got}, std = {want}, |Δ| = {diff}"
                    );
                }
            }
        }
    }

    /// Small hand fixture: 2 mel bands over 5 bins with identity-ish weights.
    #[test]
    fn fused_logmel_hand_fixture_matches_scalar() {
        // weights [2, 5] — band 0 sums bins 0..3, band 1 sums bins 2..5.
        let weights: Vec<f32> = vec![
            1.0, 1.0, 1.0, 0.0, 0.0, // band 0
            0.0, 0.0, 1.0, 1.0, 1.0, // band 1
        ];
        let power = [0.5, 1.0, 2.0, 4.0, 8.0];
        let mut out_scalar = [0.0f32; 2];
        fused_logmel_apply_frame_scalar(&weights, &power, 2, 5, 1e-10, &mut out_scalar);
        // band 0: 0.5+1+2 = 3.5, log10 ≈ 0.5441; band 1: 2+4+8 = 14, log10 ≈ 1.1461.
        assert!((out_scalar[0] - 3.5_f32.log10()).abs() < 1e-6);
        assert!((out_scalar[1] - 14.0_f32.log10()).abs() < 1e-6);

        let mut out_neon = [0.0f32; 2];
        fused_logmel_apply_frame_neon(&weights, &power, 2, 5, 1e-10, &mut out_neon);
        for (s, a) in out_scalar.iter().zip(&out_neon) {
            assert!((s - a).abs() < 1e-5, "scalar {s} vs neon {a} exceeds 1e-5");
        }
    }

    /// Floor clamp: zero (and negative) accumulator saturates to log10(floor),
    /// not -inf. This matches the Whisper `1e-10` clamp used upstream in
    /// `whisper::mel::log_mel` before the dynamic-range normalization.
    #[test]
    fn fused_logmel_zero_input_clamps_to_floor() {
        let weights = vec![1.0, 1.0, 1.0, 1.0]; // 1 band × 4 bins
        let power = [0.0f32; 4];
        let mut out = [0.0f32; 1];
        fused_logmel_apply_frame_neon(&weights, &power, 1, 4, 1e-10, &mut out);
        assert!(out[0].is_finite());
        assert!((out[0] - (1e-10_f32).log10()).abs() < 1e-6);
    }

    /// Whisper-shape parity: n_mels=80, n_bins=201 exercises both the 4-lane
    /// vector chunk (`50 * 4 = 200`) and the 1-element scalar tail — bit-close
    /// to scalar within the SIMD-log10 approximation ceiling (atol=1e-5).
    #[test]
    fn parity_whisper_shape_scalar_vs_neon() {
        // Deterministic non-negative test data (log-mel input is a power
        // spectrogram, always ≥ 0). Values are hand-rolled to avoid any
        // dev-dep on a PRNG crate (NFR-DS-02).
        let (n_mels, n_bins) = (80usize, 201usize);
        let mut weights = vec![0.0f32; n_mels * n_bins];
        for (i, w) in weights.iter_mut().enumerate() {
            // Small positive weights in [0, 1) — filterbank-like.
            *w = ((i as u32).wrapping_mul(2_654_435_761) >> 8) as f32 / (1u32 << 24) as f32;
        }
        let mut power = vec![0.0f32; n_bins];
        for (i, p) in power.iter_mut().enumerate() {
            // Power spectrum in [0, 1e4).
            *p = ((i as u32).wrapping_mul(0x9E37_79B1) >> 8) as f32 / (1u32 << 24) as f32 * 1e4;
        }
        let mut out_s = vec![0.0f32; n_mels];
        let mut out_n = vec![0.0f32; n_mels];
        fused_logmel_apply_frame_scalar(&weights, &power, n_mels, n_bins, 1e-10, &mut out_s);
        fused_logmel_apply_frame_neon(&weights, &power, n_mels, n_bins, 1e-10, &mut out_n);
        let mut max_abs = 0.0f32;
        for (i, (&s, &n)) in out_s.iter().zip(&out_n).enumerate() {
            let d = (s - n).abs();
            if d > max_abs {
                max_abs = d;
            }
            assert!(
                d < 1e-5,
                "mel {i}: scalar={s}, neon={n}, |Δ|={d} exceeds atol=1e-5"
            );
        }
        eprintln!("max |Δ| = {max_abs:e} (n_mels={n_mels}, n_bins={n_bins})");
    }
}
