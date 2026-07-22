//! Self-contained scalar `exp` / `tanh` / `sqrt` for the no_std subset
//! (M5-03-T06).
//!
//! # Why this exists
//!
//! The Silero VAD forward calls `exp` (sigmoid, `math.rs`), `tanh` (the LSTM
//! cell) and `sqrt` (the pseudo-STFT magnitude). Those `f32` transcendentals
//! live in **`std`**, not `core`: on the Cortex-M55 (`thumbv8m-none`) Tier-3
//! target (NFR-PT-03) `f32::exp` / `tanh` / `sqrt` do not resolve (measured:
//! `E0599 no method named 'exp'/'tanh'/'sqrt' found for type 'f32'`). The obvious
//! fix — the `libm` crate — is **forbidden**: it is a non-`vokra-*` crates.io
//! dependency and would break the zero-dependency invariant (NFR-DS-02;
//! `scripts/check-zero-deps.sh` fails on any non-vokra entry in `Cargo.lock`).
//!
//! So the transcendentals are supplied here in pure `core` arithmetic, **not
//! copied** from `libm` / Cephes / SLEEF / Pommier (license hygiene; Vokra stays
//! zero-dependency and Apache-2.0). Every function is deterministic across
//! targets (plain `f32` ops, no platform rounding-mode or FMA dependence), which
//! is what lets a later wave make the std and no_std Silero forwards
//! bit-identical **by construction** (M5-03 T08/T11).
//!
//! # Scope (Wave 1)
//!
//! This module only *provides* the functions and pins their accuracy with a
//! differential property test against the `std` reference. Wiring them into the
//! Silero forward (replacing the current `f32::exp` / `tanh` / `sqrt` calls) is
//! **T08 (Wave 2)**, together with the upstream-parity re-measurement that the
//! swap forces. Until then these are exercised only by the tests below, hence
//! the module-level `#[allow(dead_code)]` on the `mod` declaration in
//! `kernels/mod.rs`.
//!
//! # `sqrt` route (undecided — ADR M5-03 §sqrt)
//!
//! `sqrt` is a Newton–Raphson refinement here: portable, `core`-only, no
//! `unsafe`, deterministic across targets. It is **not** bit-identical to a
//! hardware `vsqrt` (IEEE correctly-rounded) result — it deviates by a bounded
//! few ULP. The Silero pseudo-STFT declares itself an NFR-QL-05 red line
//! (upstream-faithful `magnitude = sqrt(re²+im²)`), so whether T08 keeps this
//! Newton `sqrt` or switches to an `asm!("vsqrt.f32 …")` (HW, IEEE-exact, but
//! `unsafe` + FP-armv8-only) is an owner decision recorded in the ADR — it is
//! **undecided** and this Newton path is the Wave-1 working default, not a
//! ratified choice.

/// `log2(e)` — scales `x` to the base-2 exponent before rounding to `k`.
const LOG2E: f32 = core::f32::consts::LOG2_E;
/// High part of the Cody–Waite `ln2` split (exactly `355/512`, a dyadic
/// rational representable in f32); the low correction below relies on this exact
/// value, so the digits are kept verbatim.
#[allow(clippy::excessive_precision)]
const LN2_HI: f32 = 0.693_359_375;
/// Low correction of the `ln2` split so `LN2_HI + LN2_LO ≈ ln2`.
const LN2_LO: f32 = -2.121_944_4e-4;
/// Lower clamp on the `exp` argument (keeps `2^k` a normal f32).
const MIN_ARG: f32 = -87.0;
/// Upper clamp on the `exp` argument (keeps `2^k` a finite normal f32).
const MAX_ARG: f32 = 88.0;

// Degree-6 `exp` Taylor coefficients `1/n!` (exact factorial reciprocals — the
// only "magic" numbers here, each auditable). `C0 == C1 == 1`.
const C0: f32 = 1.0; // 1/0!
const C1: f32 = 1.0; // 1/1!
const C2: f32 = 0.5; // 1/2!
const C3: f32 = 1.0 / 6.0; // 1/3!
const C4: f32 = 1.0 / 24.0; // 1/4!
const C5: f32 = 1.0 / 120.0; // 1/5!
const C6: f32 = 1.0 / 720.0; // 1/6!

/// Scalar `exp(x)` in pure `core` arithmetic.
///
/// Standard range reduction `exp(x) = 2^k · e^r` with `k = round(x·log2e)` and
/// `r = x − k·ln2 ∈ [−ln2/2, ln2/2]` (Cody–Waite `ln2` split), then a degree-6
/// Taylor series for `e^r` (Horner). `2^k` is assembled directly in the
/// IEEE-754 exponent field. Worst-case relative error on the accurate mid-range
/// is `≈ (ln2/2)⁷/7! ≈ 1.2e-7` (a few f32 ULP), well inside the FP32 parity
/// ceiling (NFR-QL-01 `atol = 0.01`); the property test pins the empirical
/// bound. Inputs are clamped to `[MIN_ARG, MAX_ARG]` so `2^k` never overflows
/// the exponent field — beyond that the result saturates, exactly like the
/// SIMD `vexp` kernel it mirrors.
pub(crate) fn exp(x: f32) -> f32 {
    // `f32::clamp` / `min` / `max` / `abs` ARE in `core` (verified on
    // thumbv8m-none), unlike the transcendentals; NaN clamps to NaN, matching
    // the SIMD `vexp` domain guard.
    let x = x.clamp(MIN_ARG, MAX_ARG);

    // k = round-to-nearest(x · log2e). Round-half-away-from-zero via a ±0.5 bias
    // before truncation (`as i32` truncates toward zero). `f32::round` is `std`.
    let y = x * LOG2E;
    let k = if y >= 0.0 {
        (y + 0.5) as i32
    } else {
        (y - 0.5) as i32
    };
    let kf = k as f32;

    // r = x − kf·LN2_HI − kf·LN2_LO (split subtraction keeps r accurate).
    let r = x - kf * LN2_HI - kf * LN2_LO;

    // P(r) = 1 + r + r²/2! + … + r⁶/6! (Horner).
    let mut p = C6;
    p = p * r + C5;
    p = p * r + C4;
    p = p * r + C3;
    p = p * r + C2;
    p = p * r + C1;
    p = p * r + C0;

    // 2^k via IEEE-754 exponent-field assembly: biased exponent = k + 127.
    // The clamp keeps k ∈ [-126, 127], so the biased field stays in [1, 254].
    let pow2k = f32::from_bits(((k + 127) as u32) << 23);
    p * pow2k
}

/// Scalar `tanh(x)` derived from [`exp`] (M5-03-T06: "tanh from exp").
///
/// `tanh` is odd, so it is evaluated on `|x|` and the sign is restored — this
/// keeps the `exp` argument on its well-conditioned side and avoids sign-driven
/// cancellation. `tanh(|x|) = 1 − 2/(exp(2|x|) + 1)`, which saturates cleanly to
/// `+1` as `|x|` grows (the `exp` clamp makes the denominator large-finite). The
/// property test pins the empirical absolute-error bound against `std`.
pub(crate) fn tanh(x: f32) -> f32 {
    let ax = x.abs(); // `f32::abs` IS in `core` (sign-bit clear).
    let e2 = exp(2.0 * ax);
    let t = 1.0 - 2.0 / (e2 + 1.0);
    if x < 0.0 { -t } else { t }
}

/// Scalar `sqrt(x)` by Newton–Raphson (M5-03-T06, `core`-only, no `unsafe`).
///
/// A bit-hack seed (halve the biased exponent) followed by fixed Newton
/// iterations `y ← ½·(y + x/y)`. Deterministic across targets, but **not**
/// IEEE correctly-rounded — see the module docs on the undecided `sqrt` route.
/// Special values follow IEEE: `sqrt(NaN)=NaN`, `sqrt(x<0)=NaN`, `sqrt(±0)=±0`,
/// `sqrt(+∞)=+∞`.
pub(crate) fn sqrt(x: f32) -> f32 {
    if x.is_nan() || x < 0.0 {
        return f32::NAN;
    }
    // `x == 0.0` is true for both +0.0 and −0.0; returning `x` preserves the
    // sign of zero (IEEE `sqrt(-0.0) = -0.0`). `+∞` passes through unchanged.
    if x == 0.0 || x.is_infinite() {
        return x;
    }
    // Seed: (bits >> 1) + (127 << 22) roughly halves the exponent — exact for
    // even powers of two (e.g. x=4 → seed 2.0), a good start otherwise.
    let mut y = f32::from_bits((x.to_bits() >> 1) + (127u32 << 22));
    // Four iterations take the seed to ~f32 precision across the normal range.
    y = 0.5 * (y + x / y);
    y = 0.5 * (y + x / y);
    y = 0.5 * (y + x / y);
    y = 0.5 * (y + x / y);
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Points spanning the accurate `exp` mid-range plus the saturation edges.
    /// Relative-error ceiling for the scalar `exp` mid-range. A dense sweep of
    /// the whole `[MIN_ARG, MAX_ARG]` domain (below) observes a worst case of
    /// ≈2.53e-7 — a few f32 ULP, dominated by the degree-6 Taylor truncation
    /// (`(ln2/2)⁷/7! ≈ 1.2e-7`) plus Horner rounding. This bound is ~2× that
    /// observed max, NOT loosened to force a pass (red line #3); it is far
    /// inside the FP32 parity ceiling (NFR-QL-01 `atol = 0.01`).
    const EXP_REL_TOL: f32 = 5.0e-7;

    #[test]
    fn exp_matches_std_densely_over_the_whole_domain() {
        // Dense sweep so the test actually visits the worst-case argument
        // (near an argument where |r| = ln2/2), not just convenient points.
        // Sweep only the accurate domain [MIN_ARG, MAX_ARG]; below MIN_ARG the
        // clamp intentionally diverges from std (saturation, tested separately).
        let mut max_rel = 0.0f32;
        let mut i = -1740i32;
        while i <= 1760 {
            let x = i as f32 * 0.05; // [-87.0, 88.0] step 0.05
            let rel = (exp(x) - x.exp()).abs() / x.exp().abs().max(f32::MIN_POSITIVE);
            if rel > max_rel {
                max_rel = rel;
            }
            i += 1;
        }
        assert!(
            max_rel <= EXP_REL_TOL,
            "scalar exp worst-case rel = {max_rel} exceeded honest bound {EXP_REL_TOL}"
        );
    }

    #[test]
    fn exp_clamps_beyond_domain_instead_of_producing_inf() {
        // Above MAX_ARG the result saturates to a large finite value, not +inf.
        assert!(exp(1000.0).is_finite());
        // Far below MIN_ARG it saturates toward ~0, never negative.
        assert!(exp(-1000.0) >= 0.0 && exp(-1000.0) < 1e-30);
    }

    /// Absolute-error ceiling for scalar `tanh` (values live in [-1, 1], so
    /// absolute error is the natural measure). Dense sweep observes ≈9.3e-8;
    /// this bound is ~2× that observed max.
    const TANH_ABS_TOL: f32 = 2.0e-7;

    #[test]
    fn tanh_matches_std_densely_and_is_odd_saturating() {
        let mut max_abs = 0.0f32;
        let mut i = -2000i32;
        while i <= 2000 {
            let x = i as f32 * 0.01; // [-20.0, 20.0] step 0.01
            let abs = (tanh(x) - x.tanh()).abs();
            if abs > max_abs {
                max_abs = abs;
            }
            i += 1;
        }
        assert!(
            max_abs <= TANH_ABS_TOL,
            "scalar tanh worst-case abs = {max_abs} exceeded honest bound {TANH_ABS_TOL}"
        );
        // Oddness + saturation at the tails.
        assert_eq!(tanh(0.0), 0.0);
        assert!((tanh(50.0) - 1.0).abs() < 1e-6);
        assert!((tanh(-50.0) + 1.0).abs() < 1e-6);
    }

    /// Relative-error ceiling for scalar `sqrt` (Newton–Raphson). A sweep across
    /// ~40 decades observes ≈1.19e-7; this bound is ~2× that observed max. The
    /// path is deliberately non-IEEE (see the module docs on the `sqrt` route).
    const SQRT_REL_TOL: f32 = 2.5e-7;

    #[test]
    fn sqrt_matches_std_across_forty_decades() {
        let mut max_rel = 0.0f32;
        for e in -20..20 {
            for m in 1..1000 {
                let x = (m as f32) * 10f32.powi(e);
                if x <= 0.0 || !x.is_finite() {
                    continue;
                }
                let want = x.sqrt();
                if want == 0.0 {
                    continue;
                }
                let rel = (sqrt(x) - want).abs() / want;
                if rel > max_rel {
                    max_rel = rel;
                }
            }
        }
        assert!(
            max_rel <= SQRT_REL_TOL,
            "scalar sqrt worst-case rel = {max_rel} exceeded honest bound {SQRT_REL_TOL}"
        );
    }

    #[test]
    fn sqrt_handles_special_values_like_ieee() {
        assert!(sqrt(f32::NAN).is_nan());
        assert!(sqrt(-1.0).is_nan());
        assert_eq!(sqrt(0.0), 0.0);
        assert!(sqrt(-0.0).is_sign_negative()); // sqrt(-0.0) = -0.0
        assert_eq!(sqrt(f32::INFINITY), f32::INFINITY);
        // Perfect squares are essentially exact after Newton refinement.
        assert!((sqrt(4.0) - 2.0).abs() < 1e-6);
        assert!((sqrt(144.0) - 12.0).abs() < 1e-5);
    }

    #[test]
    fn no_libm_dependency_is_documented() {
        // A canary asserting the design intent recorded in the module docs:
        // these functions use only `core` arithmetic. If someone "fixes" an
        // accuracy issue by reaching for `libm`, `scripts/check-zero-deps.sh`
        // and `cargo deny check bans` (deny.toml `libm` ban) fail the build.
        // This test documents the constraint next to the code (NFR-DS-02).
        assert_eq!(exp(0.0), 1.0);
    }
}
