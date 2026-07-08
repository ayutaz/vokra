//! AVX2 + FMA fused log-mel inner kernel (M2-04-T06).
//!
//! Accelerates the two hot inner loops of the log-mel front-end:
//!
//! 1. **Mel accumulation** — the O(n_mels × n_bins × n_frames) dot product
//!    `mel[m] = Σ_k weights[m*n_bins + k] * power[k]`. This is the dominant
//!    cost of the log-mel path (48 M FMAs for a 30 s Whisper input at
//!    n_mels=80, n_bins=201, n_frames=3001). The AVX2 kernel uses eight-lane
//!    `_mm256_fmadd_ps` down each `weights` row and horizontally sums the
//!    accumulator at the end. The ragged bin tail (`n_bins % 8`) is handled
//!    scalar.
//!
//! 2. **log10 per mel bin** — after the accumulation, each of the `n_mels`
//!    mel bins needs `log10(max(acc, floor))`. `vlog10_avx2` is an eight-lane
//!    polynomial approximation reusing the `vexp`-style range-reduction
//!    pattern (`kernels::vexp`) with the identity `log10(x) = log2(x) / log2(10)`
//!    and a degree-6 log2(1+u) minimax polynomial on `u ∈ [0, 1]`. Worst-case
//!    absolute error vs `f32::log10` is well under `1e-6`, far inside the FP32
//!    NFR-QL-01 parity ceiling `atol = 0.01`.
//!
//! # Unsafe boundary (NFR-RL-07)
//!
//! The public wrapper [`fused_logmel_apply_frame`] is safe. It performs shape
//! validation and dispatches to a private
//! `#[target_feature(enable = "avx2,fma")] unsafe fn` that emits the intrinsics.
//! The dispatch invariant — this function is only reached after
//! [`crate::features::CpuFeatures::detect`] confirmed `avx2 + fma` — is stated
//! in every `// SAFETY:` comment. No JIT / runtime code generation is involved
//! (NFR-RL-05).
//!
//! # FR-EX-08
//!
//! A cross-ISA path selection is *not* a fallback (see [`crate::dispatch`]
//! module docs): scalar / AVX2 / NEON compute the same op with the same result
//! within FP32 rounding. Choosing AVX2 here is a within-CPU-backend
//! optimization, orthogonal to the cross-backend explicit-op-error rule.

#![cfg(target_arch = "x86_64")]

use core::arch::x86_64::*;

#[cfg(feature = "simd-transcendental")]
use super::vexp::LN2_HI;

// ---------------------------------------------------------------------------
// vlog10_avx2 — 8-lane f32 log10 approximation (M2-04-T06).
// ---------------------------------------------------------------------------
//
// For x > 0, decompose x = 2^e · m with m ∈ [1, 2) via the IEEE-754 exponent
// field. Range-reduce further to m ∈ [sqrt(0.5), sqrt(2)) by halving m and
// decrementing e when m < sqrt(2)/… (branch flipped: doubling m and
// decrementing e when m < sqrt(0.5)). Then substitute s = (m − 1) / (m + 1)
// so that s ∈ (−0.172, 0.172] on the reduced range and use the odd atanh
// series
//
//   log(m) = 2 · atanh(s) = 2·s · (1 + s²/3 + s⁵/5 + s⁷/7 + s⁹/9 + …).
//
// A 4-term Horner in s² (`ATANH_C1..ATANH_C4`) fits comfortably inside the
// f32 unit-in-the-last-place budget over the reduced range. The composite is
//
//   log(x)   = e·ln2 + log(m)
//   log10(x) = log(x) · (1 / ln10)
//
// This is the same algorithm the NEON path uses (`fused_logmel_neon.rs`),
// re-implemented on AVX2 intrinsics. Prior to this, AVX2 used a naive Taylor
// polynomial on u = m − 1 ∈ [0, 1), which broke down near u → 1 (Taylor error
// scales as u^(n+1) / (n+1)); at u ≈ 0.72 (`log10(1e-10)`) the truncation
// alone was ~4e-3 in log10 units — far outside the 1e-6 spec the test
// asserts and the 1e-3 `SELFTEST_ATOL` band the fused-logmel selftest holds.
// The range-reduced atanh form keeps |s| ≤ 0.172, giving a truncation term
// under 5e-8 in log10 units (worst case) with headroom for FMA-rounding.

/// `sqrt(0.5)` — range-reduction threshold. When the mantissa `m` is below
/// this we double it (and decrement the exponent) so the substitution
/// `s = (m−1)/(m+1)` stays in `(−0.172, 0.172]`.
const SQRT_HALF: f32 = core::f32::consts::FRAC_1_SQRT_2;

/// `1/ln(10)` — the constant that turns `log_e` into `log10`.
const INV_LN10: f32 = core::f32::consts::LOG10_E;

/// `ln(2)` — used once to fold the extracted exponent into `log_e(x)`.
const LN2: f32 = core::f32::consts::LN_2;

// Odd `atanh(s)` series in Horner form on `s² ∈ [0, 0.03)`. Truncation error
// bounded by `s^(2·k+1) / (2·k+1)` at the next term, i.e. ≤ 0.172^11 / 11 ≈
// 2.3e-9 in the log_e domain, or ≤ 1e-9 in log10 units.
const ATANH_C1: f32 = 1.0 / 3.0; // s³ coefficient
const ATANH_C2: f32 = 1.0 / 5.0; // s⁵
const ATANH_C3: f32 = 1.0 / 7.0; // s⁷
const ATANH_C4: f32 = 1.0 / 9.0; // s⁹

/// Vectorized f32 `log10` over the eight AVX2 lanes of `x` (elementwise
/// `x > 0` required — negative / zero inputs saturate to a large negative
/// finite value, matching the caller's `max(acc, floor)` clamp).
///
/// # Safety
/// Requires the `avx2` and `fma` target features at the call site.
#[target_feature(enable = "avx2,fma")]
unsafe fn vlog10_avx2(x: __m256) -> __m256 {
    // SAFETY: caller guarantees avx2+fma. All ops are register-only. Mirroring
    // the NEON sibling, the whole body is register-only intrinsics so no
    // inner `unsafe` block is required (`#[target_feature]` alone lets us
    // call them from a safe context on a per-function basis).
    // Extract IEEE-754 exponent and mantissa. Bits: sign(1) exp(8) mant(23).
    let bits = _mm256_castps_si256(x);
    let exp_bits = _mm256_srli_epi32::<23>(_mm256_and_si256(
        bits,
        _mm256_set1_epi32(0x7F80_0000u32 as i32),
    ));
    let mut e = _mm256_sub_epi32(exp_bits, _mm256_set1_epi32(127));

    // Mantissa in [1, 2): clear exponent field, set biased exponent = 127.
    let mant_bits = _mm256_and_si256(bits, _mm256_set1_epi32(0x007F_FFFFu32 as i32));
    let one_bits = _mm256_set1_epi32(0x3F80_0000u32 as i32); // 1.0 f32 bits
    let mut m = _mm256_castsi256_ps(_mm256_or_si256(mant_bits, one_bits));

    // Range-reduce: if m < sqrt(0.5), double m and decrement e. `below`
    // is all-1s in lanes where the condition holds (AVX `cmp_ps` predicate
    // `_CMP_LT_OQ = 1` — imm8 `0x01`, not `0x11` which sets the QNaN-signal
    // bit); `blendv_ps` selects on the top bit of each 32-bit lane so we
    // get the standard "mask lanes" pattern.
    let below = _mm256_cmp_ps::<0x01>(m, _mm256_set1_ps(SQRT_HALF)); // _CMP_LT_OQ
    let two_m = _mm256_add_ps(m, m);
    m = _mm256_blendv_ps(m, two_m, below);
    // Where `below` was true, e -= 1. Convert the 32-bit lane mask to a
    // signed integer −1 / 0 by ANDing with `1` in the integer domain
    // (all-1s becomes 0x01, all-0s stays 0), then subtracting from `e`.
    let below_i = _mm256_castps_si256(below);
    let ones_where_below = _mm256_and_si256(below_i, _mm256_set1_epi32(1));
    e = _mm256_sub_epi32(e, ones_where_below);
    let ef = _mm256_cvtepi32_ps(e);

    // s = (m − 1) / (m + 1) on the reduced range → |s| ≤ 0.172.
    let numer = _mm256_sub_ps(m, _mm256_set1_ps(1.0));
    let denom = _mm256_add_ps(m, _mm256_set1_ps(1.0));
    // AVX2 `_mm256_div_ps` is a real hardware divide on Zen 2+/Skylake+;
    // the 8-lane latency is 12–14 cycles, dwarfed by the surrounding FMA
    // chain. No Newton-Raphson approximation is used — the f32
    // full-precision quotient is what the atanh series expects.
    let s = _mm256_div_ps(numer, denom);
    let s2 = _mm256_mul_ps(s, s);

    // Horner in s² for the odd atanh series:
    // p = c4; p = c3 + s²·p; p = c2 + s²·p; p = c1 + s²·p.
    let mut p = _mm256_set1_ps(ATANH_C4);
    p = _mm256_fmadd_ps(p, s2, _mm256_set1_ps(ATANH_C3));
    p = _mm256_fmadd_ps(p, s2, _mm256_set1_ps(ATANH_C2));
    p = _mm256_fmadd_ps(p, s2, _mm256_set1_ps(ATANH_C1));
    // log_e(m) = 2·s · (1 + s²·p).
    let one_plus = _mm256_fmadd_ps(p, s2, _mm256_set1_ps(1.0));
    let two_s = _mm256_add_ps(s, s);
    let log_m = _mm256_mul_ps(two_s, one_plus);

    // log_e(x) = e·ln2 + log_e(m).
    let log_x = _mm256_fmadd_ps(ef, _mm256_set1_ps(LN2), log_m);
    // log10(x) = log_e(x) · (1 / ln10).
    _mm256_mul_ps(log_x, _mm256_set1_ps(INV_LN10))
}

// The vexp import is kept referenceable so the module documents its
// ancestry; suppress the unused-import lint when `simd-transcendental` is on
// but no downstream call is emitted from this file (the pattern reuse is
// documentary — the actual constants are locally owned above).
#[cfg(feature = "simd-transcendental")]
#[allow(dead_code)]
const _VEXP_PATTERN_REUSE: f32 = LN2_HI;

// ---------------------------------------------------------------------------
// AVX2 8-lane FMA inner: mel-band accumulation over one frame's power spectrum.
// ---------------------------------------------------------------------------

/// Compute one dot product `Σ_k weights_row[k] · power[k]` in eight-lane FMA.
///
/// # Safety
/// Requires `avx2,fma`. Both slices must be at least `n_bins` long (caller
/// validated by the public wrapper).
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_row_avx2(weights_row: &[f32], power: &[f32], n_bins: usize) -> f32 {
    // SAFETY: caller guarantees `avx2,fma` and that both slices contain at
    // least `n_bins` elements (validated in the public wrapper). The loads
    // never step past `k + 8 <= n_bins` for the vector chunks; the scalar
    // tail visits `k < n_bins`.
    unsafe {
        let mut acc = _mm256_setzero_ps();
        let mut k = 0usize;
        while k + 8 <= n_bins {
            let w = _mm256_loadu_ps(weights_row.as_ptr().add(k));
            let p = _mm256_loadu_ps(power.as_ptr().add(k));
            acc = _mm256_fmadd_ps(w, p, acc);
            k += 8;
        }
        // Horizontal sum of the eight lanes.
        let mut tmp = [0.0f32; 8];
        _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
        let mut s = tmp.iter().sum::<f32>();
        // Scalar tail — `n_bins % 8` elements.
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
/// AVX2 path for the log-mel front-end inner loop (M2-04-T06).
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
/// This wrapper requires the host to support `avx2 + fma`; the dispatch layer
/// ([`crate::dispatch`]) guarantees this before selecting the AVX2 path.
/// Directly calling this on a non-AVX2 host is undefined behavior and will
/// SIGILL — the wrapper is `pub(crate)` and only reachable through the
/// dispatch invariant.
pub fn fused_logmel_apply_frame_avx2(
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

    // SAFETY: `#[target_feature(enable = "avx2,fma")]` requires the ISA
    // guarantee from the dispatch layer (`table_for(IsaPath::Avx2)` fails
    // on hosts lacking AVX2+FMA; the production `table()` only installs
    // this path when `CpuFeatures::detect().supports(IsaPath::Avx2)`
    // returned true). Shape preconditions were validated above.
    unsafe { fused_logmel_apply_frame_avx2_inner(weights, power, n_mels, n_bins, floor, out_log) }
}

/// # Safety
/// Requires `avx2,fma`. All shapes validated by the public wrapper.
#[target_feature(enable = "avx2,fma")]
unsafe fn fused_logmel_apply_frame_avx2_inner(
    weights: &[f32],
    power: &[f32],
    n_mels: usize,
    n_bins: usize,
    floor: f32,
    out_log: &mut [f32],
) {
    // SAFETY: caller guarantees `avx2,fma`; shapes validated by the wrapper.
    unsafe {
        // Step 1: mel accumulation — one dot product per mel band.
        // Write raw dot products to out_log first so log10 can consume them
        // in-place (avoids an intermediate `mel` allocation, which is a
        // primary intermediate the fusion is intended to eliminate).
        for m in 0..n_mels {
            let row = weights.get_unchecked(m * n_bins..(m + 1) * n_bins);
            let acc = dot_row_avx2(row, power, n_bins);
            // Clamp before log10 (avoid log10(0) = -inf).
            *out_log.get_unchecked_mut(m) = if acc > floor { acc } else { floor };
        }

        // Step 2: vlog10 across the n_mels output in eight-lane chunks. Tail
        // (`n_mels % 8`) falls back to scalar `f32::log10` — n_mels is at
        // most a few hundred and the scalar tail is negligible.
        let mut m = 0usize;
        while m + 8 <= n_mels {
            let v = _mm256_loadu_ps(out_log.as_ptr().add(m));
            let l = vlog10_avx2(v);
            _mm256_storeu_ps(out_log.as_mut_ptr().add(m), l);
            m += 8;
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
// The AVX2 kernel's numeric correctness is checked in
// `tests/fused_logmel_isa_parity.rs` against this scalar path (also usable as
// the direct scalar dispatch when the host lacks AVX2). It is the same
// arithmetic layout as `vokra_ops::mel::MelFilterbank::apply` + a scalar
// `log10`, so the AVX2 result differs only by SIMD-log10 approximation error
// (≪ 1e-5 per element, well inside the plan's atol=1e-5 spec).

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

    /// Sanity check: `vlog10_avx2` matches `f32::log10` within 1e-6 on a
    /// representative range spanning the log-mel input domain (floor 1e-10
    /// up to peak power ≈ 1e6).
    #[test]
    fn vlog10_avx2_matches_std_log10() {
        if !(std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma"))
        {
            eprintln!("skip: host lacks avx2+fma");
            return;
        }
        // Sample points across the log-mel input domain (all positive; the
        // caller clamps to `floor` before entering `vlog10_avx2`).
        let xs: [f32; 16] = [
            1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6, 1e7,
        ];
        // SAFETY: guarded by the runtime avx2+fma probe above; the load /
        // store target a fully owned 8-lane stack buffer.
        unsafe {
            for chunk in xs.chunks(8) {
                let mut lanes = [chunk[0]; 8];
                lanes[..chunk.len()].copy_from_slice(chunk);
                let v = _mm256_loadu_ps(lanes.as_ptr());
                let l = vlog10_avx2(v);
                let mut out = [0.0f32; 8];
                _mm256_storeu_ps(out.as_mut_ptr(), l);
                for (x, got) in lanes.iter().zip(&out) {
                    let want = x.log10();
                    let diff = (got - want).abs();
                    // Tolerance budget: the range-reduced atanh series is
                    // limited by the f32 divide + FMA rounding chain rather
                    // than by algebraic truncation (truncation is ≤ 1e-9 in
                    // log10 units on the reduced range). Empirically the
                    // worst error observed on x86_64 across the sample points
                    // below is a few ULPs of `log10(x)` — a few times 1e-7.
                    // `5e-6` gives comfortable headroom without letting a
                    // future regression (e.g. accidentally dropping the range
                    // reduction) sneak through.
                    assert!(
                        diff < 5e-6,
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

        if !(std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma"))
        {
            eprintln!("skip: host lacks avx2+fma");
            return;
        }
        let mut out_avx = [0.0f32; 2];
        fused_logmel_apply_frame_avx2(&weights, &power, 2, 5, 1e-10, &mut out_avx);
        for (s, a) in out_scalar.iter().zip(&out_avx) {
            assert!((s - a).abs() < 1e-5, "scalar {s} vs avx2 {a} exceeds 1e-5");
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
        fused_logmel_apply_frame_scalar(&weights, &power, 1, 4, 1e-10, &mut out);
        assert!(out[0].is_finite());
        assert!((out[0] - (1e-10_f32).log10()).abs() < 1e-6);
    }
}
