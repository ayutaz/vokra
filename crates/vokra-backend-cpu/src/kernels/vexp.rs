//! Native vectorized `exp` for the transcendental activations (M1-05-EXP).
//!
//! `sigmoid` / `tanh` / softmax's `exp` pass and `gelu`'s `erf` are all
//! `exp`-bound; on the hot Whisper-base attention softmax and MLP GELU this
//! per-lane `exp` dominates. This module supplies a SIMD `exp` so those
//! kernels stop delegating the transcendental to the scalar path, protecting
//! the Whisper RTF exit gate (NFR-PF-01, M1-11).
//!
//! # Algorithm (standard range reduction — natively derived, not vendored)
//!
//! For each lane, `exp(x) = 2^k · e^r` with
//!
//! - `k = round(x · log2 e)` (nearest integer), and
//! - `r = x − k·ln2 ∈ [−ln2/2, ln2/2]`, computed with a Cody–Waite split
//!   `ln2 = LN2_HI + LN2_LO` so the subtraction stays accurate.
//!
//! `e^r` on that small interval is the degree-6 Taylor series
//! `1 + r + r²/2! + … + r⁶/6!` (Horner form); its worst-case relative error
//! at `|r| = ln2/2` is `≈ (ln2/2)⁷/7! ≈ 1.2e-7`, i.e. a few f32 ULP and well
//! inside the FP32 parity ceiling NFR-QL-01 `atol = 0.01`. `2^k` is assembled
//! directly in the IEEE-754 exponent field (`(k + 127) << 23`).
//!
//! The coefficients are the exact `1/n!` Taylor rationals and the standard
//! Cody–Waite `ln2` split — derived here from the identity above, **not**
//! copied from Cephes / SLEEF / Pommier `exp_ps` (license hygiene: those carry
//! non-Apache/MIT provenance; Vokra stays zero-dependency and Apache-2.0). The
//! coefficients are pinned by the differential test.
//!
//! # Domain / saturation
//!
//! Inputs are clamped to `[MIN_ARG, MAX_ARG]` so `2^k` never overflows the f32
//! exponent field (biased exponent stays in `[1, 254]`). Beyond that range the
//! result saturates to a large finite value / `~0` rather than `±inf` / `0`.
//! Every consumer here is saturating (`sigmoid`/`tanh` clamp to `±1`) or
//! normalized (softmax divides by the row sum; its `exp` argument is always
//! `≤ 0`), so this differs from `std::f32::exp` only outside `[MIN_ARG,
//! MAX_ARG]`, where the activation output is already saturated.
//!
//! This module compiles only under the `simd-transcendental` feature (see the
//! crate `Cargo.toml`); its contents are further gated to the SIMD-bearing
//! architectures.

#![cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]

/// `log2(e)` — scales `x` to the base-2 exponent before rounding to `k`.
pub(crate) const LOG2E: f32 = std::f32::consts::LOG2_E;
/// High part of the Cody–Waite `ln2` split (exactly `355/512`, a dyadic
/// rational representable in f32; the low correction below relies on this exact
/// value, so the digits are kept verbatim — `clippy::excessive_precision`
/// would round-trip to the same bit pattern but obscure the `355/512` intent).
#[allow(clippy::excessive_precision)]
pub(crate) const LN2_HI: f32 = 0.693_359_375;
/// Low correction of the `ln2` split so `LN2_HI + LN2_LO ≈ ln2`.
pub(crate) const LN2_LO: f32 = -2.121_944_4e-4;

/// Lower clamp on the `exp` argument (keeps `2^k` a normal f32).
pub(crate) const MIN_ARG: f32 = -87.0;
/// Upper clamp on the `exp` argument (keeps `2^k` a finite normal f32).
pub(crate) const MAX_ARG: f32 = 88.0;

// Degree-6 exp Taylor coefficients `1/n!` (exact rationals; the only "magic"
// numbers here, and each is auditable as a factorial reciprocal).
const C0: f32 = 1.0; // 1/0!
const C1: f32 = 1.0; // 1/1!
const C2: f32 = 0.5; // 1/2!
const C3: f32 = 1.0 / 6.0; // 1/3!
const C4: f32 = 1.0 / 24.0; // 1/4!
const C5: f32 = 1.0 / 120.0; // 1/5!
const C6: f32 = 1.0 / 720.0; // 1/6!

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Vectorized f32 `exp` over the eight AVX2 lanes of `x`.
///
/// The body is register-only (no memory access), so the intrinsics are safe
/// to call within this matching `#[target_feature]` context and need no inner
/// `unsafe` block.
///
/// # Safety
/// Requires the `avx2` and `fma` target features at the call site (the AVX2
/// kernels satisfy this via the dispatch invariant: the `Avx2` path is only
/// selected when `avx2 && fma` were detected).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn exp_ps_avx2(x: __m256) -> __m256 {
    // Clamp to the representable exp domain (see module docs).
    let x = _mm256_min_ps(
        _mm256_max_ps(x, _mm256_set1_ps(MIN_ARG)),
        _mm256_set1_ps(MAX_ARG),
    );

    // k = round-to-nearest(x * log2e); kf = (f32)k.
    let ki = _mm256_cvtps_epi32(_mm256_mul_ps(x, _mm256_set1_ps(LOG2E)));
    let kf = _mm256_cvtepi32_ps(ki);

    // r = x - kf*LN2_HI - kf*LN2_LO   (fnmadd(a,b,c) = c - a*b).
    let r = _mm256_fnmadd_ps(kf, _mm256_set1_ps(LN2_HI), x);
    let r = _mm256_fnmadd_ps(kf, _mm256_set1_ps(LN2_LO), r);

    // P(r) = 1 + r + r^2/2! + ... + r^6/6!   (Horner; fmadd(a,b,c)=a*b+c).
    let mut p = _mm256_set1_ps(C6);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(C5));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(C4));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(C3));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(C2));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(C1));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(C0));

    // 2^k via IEEE-754 exponent-field assembly: ((k + 127) << 23).
    let pow2k = _mm256_castsi256_ps(_mm256_slli_epi32(
        _mm256_add_epi32(ki, _mm256_set1_epi32(127)),
        23,
    ));
    _mm256_mul_ps(p, pow2k)
}

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

/// Vectorized f32 `exp` over the four NEON lanes of `x`.
///
/// The body is register-only (no memory access), so the intrinsics are safe
/// to call within this matching `#[target_feature]` context and need no inner
/// `unsafe` block.
///
/// # Safety
/// Requires the `neon` target feature (the AArch64 baseline).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn exp_ps_neon(x: float32x4_t) -> float32x4_t {
    let x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(MIN_ARG)), vdupq_n_f32(MAX_ARG));

    // k = round-to-nearest(x * log2e); kf = (f32)k.
    let ki = vcvtnq_s32_f32(vmulq_f32(x, vdupq_n_f32(LOG2E)));
    let kf = vcvtq_f32_s32(ki);

    // r = x - kf*LN2_HI - kf*LN2_LO   (vfmsq_f32(a,b,c) = a - b*c).
    let r = vfmsq_f32(x, kf, vdupq_n_f32(LN2_HI));
    let r = vfmsq_f32(r, kf, vdupq_n_f32(LN2_LO));

    // P(r) = 1 + r + ... + r^6/6!   (Horner; vfmaq_f32(a,b,c) = a + b*c).
    let mut p = vdupq_n_f32(C6);
    p = vfmaq_f32(vdupq_n_f32(C5), p, r);
    p = vfmaq_f32(vdupq_n_f32(C4), p, r);
    p = vfmaq_f32(vdupq_n_f32(C3), p, r);
    p = vfmaq_f32(vdupq_n_f32(C2), p, r);
    p = vfmaq_f32(vdupq_n_f32(C1), p, r);
    p = vfmaq_f32(vdupq_n_f32(C0), p, r);

    // 2^k via IEEE-754 exponent-field assembly: ((k + 127) << 23).
    let pow2k = vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(ki, vdupq_n_s32(127)), 23));
    vmulq_f32(p, pow2k)
}

#[cfg(test)]
mod tests {
    use super::{MAX_ARG, MIN_ARG};

    /// Points spanning the accurate mid-range plus saturation edges.
    const XS: [f32; 15] = [
        -30.0, -20.0, -10.0, -5.0, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 5.0, 10.0, 20.0, 30.0,
    ];

    /// Relative-error ceiling for the mid-range (well under 1 ULP-ish + poly).
    const REL_TOL: f32 = 1e-5;

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn exp_ps_avx2_matches_std_exp() {
        use super::exp_ps_avx2;
        use core::arch::x86_64::*;
        if !(std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma"))
        {
            eprintln!("skip: host lacks avx2+fma");
            return;
        }
        // Process XS in 8-lane chunks (pad the tail by repeating the last x).
        for chunk in XS.chunks(8) {
            let mut lanes = [*chunk.last().unwrap(); 8];
            lanes[..chunk.len()].copy_from_slice(chunk);
            // SAFETY: guarded by the avx2+fma detection above; the load/store
            // target a fully owned, correctly sized 8-lane stack buffer.
            let out: [f32; 8] = unsafe {
                let v = _mm256_loadu_ps(lanes.as_ptr());
                let e = exp_ps_avx2(v);
                let mut o = [0.0f32; 8];
                _mm256_storeu_ps(o.as_mut_ptr(), e);
                o
            };
            for (&x, &got) in lanes.iter().zip(&out) {
                check_exp(x, got);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn exp_ps_neon_matches_std_exp() {
        use super::exp_ps_neon;
        use core::arch::aarch64::*;
        for chunk in XS.chunks(4) {
            let mut lanes = [*chunk.last().unwrap(); 4];
            lanes[..chunk.len()].copy_from_slice(chunk);
            // SAFETY: NEON is the AArch64 baseline; the load/store target a
            // fully owned, correctly sized 4-lane stack buffer.
            let out: [f32; 4] = unsafe {
                let v = vld1q_f32(lanes.as_ptr());
                let e = exp_ps_neon(v);
                let mut o = [0.0f32; 4];
                vst1q_f32(o.as_mut_ptr(), e);
                o
            };
            for (&x, &got) in lanes.iter().zip(&out) {
                check_exp(x, got);
            }
        }
    }

    /// Shared oracle check: mid-range compares relative error to `std::exp`;
    /// the saturation edges only require a sane finite / small value.
    fn check_exp(x: f32, got: f32) {
        let want = x.exp();
        if (MIN_ARG..=MAX_ARG).contains(&x) {
            let rel = (got - want).abs() / want.abs().max(f32::MIN_POSITIVE);
            assert!(
                rel <= REL_TOL,
                "exp({x}) = {got}, std = {want}, rel = {rel} > {REL_TOL}"
            );
        } else {
            // Clamped domain: must stay finite, non-negative, and monotone-ish.
            assert!(got.is_finite() && got >= 0.0, "exp({x}) = {got} not sane");
        }
    }
}
