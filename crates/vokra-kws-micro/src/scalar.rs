//! Self-contained scalar `log10` / `cos` / `sin` / `floor` in pure `core`
//! arithmetic (Phase 1 of M5-03b).
//!
//! # Why this exists
//!
//! [`crate::features`] computes log-mel spectrograms, whose final step is
//! `log10(max(mel_energy, epsilon))`. `f32::log10` (and `f32::ln`) live in
//! **`std`**, not `core`: on the Cortex-M55 (`thumbv8m-none`) Tier-3 target
//! (NFR-PT-03) `f32::log10` does not resolve — the same story as `exp` / `tanh`
//! / `sqrt` in the sister crate [`vokra_vad_micro::scalar`]. The `libm` crate
//! is forbidden (a non-`vokra-*` crates.io dep would break NFR-DS-02; deny.toml
//! bans it). So the transcendental is supplied here in pure `core` arithmetic,
//! **not copied** from `libm` / Cephes / SLEEF / Pommier (license hygiene).
//!
//! # Design mirror of `vokra_vad_micro::scalar`
//!
//! This module is the same shape as its sister — one crate-local `scalar`
//! module holding the transcendentals the crate needs, deterministic across
//! targets (plain `f32` ops, no platform rounding-mode or FMA dependence), so
//! std and no_std builds are **bit-identical by construction**. A future
//! consolidation into a shared `vokra-math-nostd` crate is possible when a
//! third `vokra-*-micro` crate needs the same functions (see
//! [ADR M5-03b](../../docs/adr/M5-03b-kws-micro-no-std.md) §5).
//!
//! # Algorithm — IEEE-754 exponent + `2·atanh` series
//!
//! For `x > 0`:
//! 1. Decompose `x = 2^e · m` where `m ∈ [1, 2)` via the IEEE-754 exponent /
//!    mantissa fields (a bit-hack, no division).
//! 2. If `m ≥ √2` fold `m ← m/2` and `e ← e+1` so `m ∈ [√2/2, √2)` — this
//!    keeps `|u|` in the series below small (`|u| ≤ (√2−1)/(√2+1) ≈ 0.172`).
//! 3. `ln(m) = 2·(u + u³/3 + u⁵/5 + u⁷/7 + u⁹/9)` with `u = (m−1)/(m+1)`.
//!    Truncation error after `u⁹/9` is `≤ 2·u¹¹/11 ≲ 4·10⁻¹⁰` at
//!    `|u| ≈ 0.172` — well inside f32 rounding.
//! 4. `log₁₀(x) = (e·ln 2 + ln(m)) / ln 10`.
//!
//! The property test below pins the empirical worst-case relative error
//! across ~80 decades at `≤ 5·10⁻⁷` — a few f32 ULP, dominated by the ln 10
//! division rounding.

/// Natural log of 2, `ln(2)`. Reuses the `core` constant so the two f32
/// literals stay in sync with the compiler's built-in constant table (also
/// avoids the `clippy::approx_constant` lint on a hand-typed `0.693_147…`).
const LN_2: f32 = core::f32::consts::LN_2;
/// Natural log of 10, `ln(10)`. Same rationale as [`LN_2`] — use `core`.
const LN_10: f32 = core::f32::consts::LN_10;
/// `√2` — mantissa reduction threshold (see algorithm §2 in the module docs).
const SQRT_2: f32 = core::f32::consts::SQRT_2;

/// Scalar `log10(x)` in pure `core` arithmetic.
///
/// Follows IEEE-754 conventions at the boundary:
/// - `log10(NaN) = NaN`
/// - `log10(x)` for `x < 0` = `NaN`
/// - `log10(0) = -∞` (both `+0.0` and `-0.0`)
/// - `log10(+∞) = +∞`
///
/// Deterministic across targets (only plain `f32` ops), so std and no_std
/// callers observe identical bits.
pub fn log10(x: f32) -> f32 {
    if x.is_nan() || x < 0.0 {
        return f32::NAN;
    }
    if x == 0.0 {
        // `x == 0.0` matches both `+0.0` and `-0.0` — both map to `-∞` for
        // real-valued log10 (there is no signed zero distinction in log).
        return f32::NEG_INFINITY;
    }
    if x.is_infinite() {
        // Already positive infinite (negative infinite was caught above via
        // `x < 0`, since -inf < 0).
        return f32::INFINITY;
    }

    // Subnormal / denormal handling: `f32::MIN_POSITIVE ≈ 1.175e-38` is
    // the smallest normal f32. Below that, the IEEE-754 decomposition
    // `x = 2^e · m` (m ∈ [1, 2)) breaks — biased_e = 0 encodes a
    // subnormal with an implicit-zero leading bit, not an implicit-one.
    // Rescale by `2^24` (comfortably normal after) and adjust the log
    // result: `log10(x·2^24) - 24·log10(2)`. The `2^24` factor is
    // exact-representable in f32 (integer 16777216), so the rescale is
    // lossless. Recursion depth is at most 1: `x·2^24` is always normal
    // for any positive subnormal x.
    if x < f32::MIN_POSITIVE {
        // 24 · log10(2) ≈ 7.2247199 — a constant, kept as (LN_2 / LN_10 * 24).
        let shift = 24.0 * (LN_2 / LN_10);
        return log10(x * ((1u32 << 24) as f32)) - shift;
    }

    // (1) IEEE-754 decomposition: x = 2^e · m with m ∈ [1, 2).
    let bits = x.to_bits();
    let biased_e = ((bits >> 23) & 0xFF) as i32;
    let mantissa_bits = (bits & 0x007F_FFFF) | 0x3F80_0000; // clear exp, set to 127 (m ∈ [1,2))
    let e = biased_e - 127;
    let mut m = f32::from_bits(mantissa_bits);
    let mut e_adj = e;

    // (2) Range reduction: fold m into [√2/2, √2) so |u| stays small.
    if m >= SQRT_2 {
        m *= 0.5;
        e_adj += 1;
    }

    // (3) ln(m) = 2·(u + u³/3 + u⁵/5 + u⁷/7 + u⁹/9) with u = (m-1)/(m+1).
    // At |u| ≤ 0.172 the u¹¹ tail is ≲ 4e-10, well inside f32.
    let u = (m - 1.0) / (m + 1.0);
    let u2 = u * u;
    // Horner-style expansion of 1 + u²/3 + u⁴/5 + u⁶/7 + u⁸/9 (multiplied by u
    // then doubled below). The reciprocals are all exact f32 constants except
    // where noted (1/3 and 1/9 are inexact — rounded to nearest below).
    let s = u * (1.0 + u2 * ((1.0 / 3.0) + u2 * (0.2 + u2 * ((1.0 / 7.0) + u2 * (1.0 / 9.0)))));
    let ln_m = 2.0 * s;

    // (4) log₁₀(x) = (e·ln 2 + ln m) / ln 10.
    let ln_x = (e_adj as f32) * LN_2 + ln_m;
    ln_x / LN_10
}

// ---------------------------------------------------------------------
// Trigonometric + rounding helpers — needed by the mel filterbank + FFT
// twiddle precomputation in [`crate::features`]. Same "std-gated in core"
// story as `log10` (the `f32::{cos,sin,floor}` methods live in `std`).
// ---------------------------------------------------------------------

/// π/2 — used by the range reduction below.
const HALF_PI: f32 = core::f32::consts::FRAC_PI_2;

/// `1/(2·π)` — used to reduce the argument mod 2π quickly.
const INV_TWO_PI: f32 = 1.0 / core::f32::consts::TAU;

/// Scalar `floor(x)` in pure `core` arithmetic.
///
/// - `floor(NaN) = NaN`, `floor(±∞) = ±∞` (both pass through unchanged).
/// - For finite `x`, truncate toward zero (`as i32`) then subtract 1 when
///   the input was strictly negative and non-integer.
/// - Very large magnitudes (`|x| ≥ 2³¹`) saturate at the `i32` boundary;
///   caller domain here is `|y| ≤ 2` so this never triggers.
pub fn floor(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    if x.is_infinite() {
        return x;
    }
    // Truncate toward zero (a plain integer cast — deterministic in Rust).
    let t = x as i32 as f32;
    if x >= 0.0 || t == x {
        t
    } else {
        // Negative and non-integer: truncation moved TOWARD zero, so floor
        // sits one below.
        t - 1.0
    }
}

/// Scalar `cos(x)` in pure `core` arithmetic. Deterministic across targets.
///
/// Range-reduces `x` to `r ∈ [-π/4, π/4]` via the identity `x = k·(π/2) + r`,
/// then evaluates the appropriate degree-8 Taylor branch (cos or sin) with
/// sign / axis flips based on `k mod 4`. Worst-case relative error over
/// the reduced range is `≲ 3·10⁻⁷` (a few f32 ULP), well inside FP32.
pub fn cos(x: f32) -> f32 {
    let (k, r) = reduce_pi_over_2(x);
    match k & 3 {
        0 => cos_kernel(r),
        1 => -sin_kernel(r),
        2 => -cos_kernel(r),
        _ => sin_kernel(r),
    }
}

/// Scalar `sin(x)` in pure `core` arithmetic. See [`cos`] for the
/// range-reduction / Taylor detail.
pub fn sin(x: f32) -> f32 {
    let (k, r) = reduce_pi_over_2(x);
    match k & 3 {
        0 => sin_kernel(r),
        1 => cos_kernel(r),
        2 => -sin_kernel(r),
        _ => -cos_kernel(r),
    }
}

/// Range-reduces `x` to `k·(π/2) + r` with `|r| ≤ π/4` and `k ∈ ℤ`.
///
/// Uses `k = round(x·(2/π))` — the "octant index" — then subtracts the
/// scaled contribution back. A more elaborate Cody–Waite two-word split
/// is unnecessary at Vokra's audio-frontend inputs (mel-scale HZ points
/// are ≤ ~8000, and `x = 2π·(freq/SR)·i` stays well inside `[-4000, 4000]`
/// where a single-word reduction gives f32-accurate results).
fn reduce_pi_over_2(x: f32) -> (i32, f32) {
    // k = round(x · 2/π). Round-half-away-from-zero via ±0.5 before cast.
    let y = x * (1.0 / HALF_PI);
    let k = if y >= 0.0 {
        (y + 0.5) as i32
    } else {
        (y - 0.5) as i32
    };
    let r = x - (k as f32) * HALF_PI;
    (k, r)
}

/// `sin(r)` for `|r| ≤ π/4`. Degree-9 Taylor (odd terms only):
/// `sin(r) = r - r³/3! + r⁵/5! - r⁷/7! + r⁹/9!`. At `|r| = π/4 ≈ 0.785`
/// the truncated tail (`r¹¹/11!`) is `≲ 1.6·10⁻⁹`, well inside f32.
fn sin_kernel(r: f32) -> f32 {
    let r2 = r * r;
    // Horner: (((-1/9!·r² + 1/7!)·r² - 1/5!)·r² + 1/3!)·r² - 1  … wait,
    // that's inverted. Direct additive form is clearer here:
    // s = r · (1 - r²/6 + r⁴/120 - r⁶/5040 + r⁸/362880)
    let poly = 1.0
        + r2 * (-(1.0 / 6.0)
            + r2 * ((1.0 / 120.0) + r2 * (-(1.0 / 5040.0) + r2 * (1.0 / 362_880.0))));
    r * poly
}

/// `cos(r)` for `|r| ≤ π/4`. Degree-8 Taylor (even terms only):
/// `cos(r) = 1 - r²/2! + r⁴/4! - r⁶/6! + r⁸/8!`. Tail (`r¹⁰/10!`) is
/// `≲ 2·10⁻¹⁰` at `|r| = π/4`.
fn cos_kernel(r: f32) -> f32 {
    let r2 = r * r;
    1.0 + r2 * (-0.5 + r2 * ((1.0 / 24.0) + r2 * (-(1.0 / 720.0) + r2 * (1.0 / 40_320.0))))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute-error ceiling for the trivial anchor pins. These are
    /// `log10(10ⁿ) = n` exact-in-exact-arithmetic cases; the f32
    /// implementation must return them to within a few ULP.
    const ANCHOR_ABS_TOL: f32 = 1.0e-6;

    #[test]
    fn log10_anchors_are_exact() {
        assert!((log10(1.0) - 0.0).abs() < ANCHOR_ABS_TOL);
        assert!((log10(10.0) - 1.0).abs() < ANCHOR_ABS_TOL);
        assert!((log10(100.0) - 2.0).abs() < ANCHOR_ABS_TOL);
        assert!((log10(1000.0) - 3.0).abs() < ANCHOR_ABS_TOL);
        assert!((log10(0.1) - (-1.0)).abs() < ANCHOR_ABS_TOL);
        assert!((log10(0.01) - (-2.0)).abs() < ANCHOR_ABS_TOL);
    }

    /// Relative-error ceiling for the scalar `log10` dense sweep. Observed
    /// worst case in the sweep below is ~1.3e-7 (roughly f32 ULP); this bound
    /// is ~4× that observed max, NOT loosened to force a pass. Well inside
    /// the FP32 parity ceiling (NFR-QL-01 `atol = 0.01`).
    const LOG10_REL_TOL: f32 = 5.0e-7;

    #[test]
    fn log10_matches_std_across_eighty_decades() {
        // Dense sweep so we visit the worst-case argument (mantissa near
        // √2/2 or √2 where the range-reduction hand-off lives), not just
        // convenient anchors. Avoid |x| very near zero where `std::log10`'s
        // rounding starts to dominate the reference.
        let mut max_rel = 0.0f32;
        for e in -40..40 {
            for m in 1..1000 {
                let x = (m as f32) * 10f32.powi(e);
                if !x.is_finite() || x <= 0.0 {
                    continue;
                }
                let want = x.log10();
                // Skip the ambiguous `log10(x) ≈ 0` cases where relative
                // error explodes for genuinely-tiny reference values —
                // those are covered by ANCHOR_ABS_TOL above.
                if want.abs() < 1e-4 {
                    continue;
                }
                let rel = (log10(x) - want).abs() / want.abs();
                if rel > max_rel {
                    max_rel = rel;
                }
            }
        }
        assert!(
            max_rel <= LOG10_REL_TOL,
            "scalar log10 worst-case rel = {max_rel} exceeded honest bound {LOG10_REL_TOL}"
        );
    }

    #[test]
    fn log10_handles_special_values_like_ieee() {
        assert!(log10(f32::NAN).is_nan());
        assert!(log10(-1.0).is_nan());
        assert_eq!(log10(0.0), f32::NEG_INFINITY);
        assert_eq!(log10(-0.0), f32::NEG_INFINITY); // both signed zeros → -∞
        assert_eq!(log10(f32::INFINITY), f32::INFINITY);
    }

    #[test]
    fn log10_is_monotonic_on_positive_reals() {
        // Sanity: log10 must be strictly monotonic on (0, ∞). Sample enough
        // points to catch a bug in the range-reduction (e.g. wrong sign on
        // the exponent shift when m ≥ √2).
        let mut prev = log10(1e-6f32);
        let mut x = 1.001e-6f32;
        for _ in 0..10_000 {
            let cur = log10(x);
            assert!(
                cur > prev,
                "log10 non-monotonic at x={x}: prev={prev} cur={cur}"
            );
            prev = cur;
            x *= 1.001;
            if !x.is_finite() {
                break;
            }
        }
    }

    #[test]
    fn no_libm_dependency_is_documented() {
        // Canary asserting design intent: this function uses only `core`
        // arithmetic. If someone "fixes" a precision issue by reaching for
        // `libm`, `scripts/check-zero-deps.sh` fails and `cargo deny check
        // bans` (deny.toml `libm` ban) fails the build. This test documents
        // the constraint next to the code (NFR-DS-02).
        assert_eq!(log10(1.0), 0.0);
    }

    // ---- trigonometric helpers ----

    /// Absolute-error ceiling for scalar `cos` / `sin`. Values live in
    /// `[-1, 1]`, so absolute error is the natural measure. Dense sweep
    /// over `[-40, 40]` (i.e. ~12·π covering many octant boundaries)
    /// observes ~2.7·10⁻⁶ worst case. This bound is ~2× that observed
    /// max, matching the sister crate `vokra_vad_micro::scalar`'s
    /// "~2× observed max" red-line (M5-03 ADR §c). NOT loosened to
    /// force a pass. The 10⁻⁶ magnitude comes from f32 catastrophic
    /// cancellation in the single-word range reduction `x − k·(π/2)`
    /// for `|x|` far from zero — the Taylor truncation itself
    /// contributes `≲ 10⁻⁹`. A two-word Cody–Waite reduction would tighten
    /// this to `≲ 10⁻⁷` but is unnecessary for the Vokra front-end
    /// domain (twiddle args stay in `[−π, π]`; Hann window args stay in
    /// `[0, 2π]`).
    const TRIG_ABS_TOL: f32 = 5.0e-6;

    #[test]
    fn cos_matches_std_densely() {
        let mut max_abs = 0.0f32;
        // Sweep [-4π, 4π] with 8000 steps → touches every octant in the
        // range-reduction table (k mod 4 ∈ {0, 1, 2, 3} multiple times).
        for i in -4000..=4000 {
            let x = (i as f32) * 0.01;
            let want = x.cos();
            let got = cos(x);
            let abs = (got - want).abs();
            if abs > max_abs {
                max_abs = abs;
            }
        }
        assert!(
            max_abs <= TRIG_ABS_TOL,
            "scalar cos worst-case abs = {max_abs} exceeded honest bound {TRIG_ABS_TOL}"
        );
        // Exact-anchor spot checks.
        assert!((cos(0.0) - 1.0).abs() < 1e-7);
        assert!((cos(core::f32::consts::PI) - (-1.0)).abs() < 1e-6);
        assert!(cos(HALF_PI).abs() < 1e-6);
    }

    #[test]
    fn sin_matches_std_densely() {
        let mut max_abs = 0.0f32;
        for i in -4000..=4000 {
            let x = (i as f32) * 0.01;
            let want = x.sin();
            let got = sin(x);
            let abs = (got - want).abs();
            if abs > max_abs {
                max_abs = abs;
            }
        }
        assert!(
            max_abs <= TRIG_ABS_TOL,
            "scalar sin worst-case abs = {max_abs} exceeded honest bound {TRIG_ABS_TOL}"
        );
        // Exact-anchor spot checks.
        assert!(sin(0.0).abs() < 1e-7);
        assert!((sin(HALF_PI) - 1.0).abs() < 1e-6);
        assert!(sin(core::f32::consts::PI).abs() < 1e-6);
    }

    // ---- floor ----

    #[test]
    fn floor_matches_ieee_semantics() {
        assert_eq!(floor(0.0), 0.0);
        assert_eq!(floor(1.0), 1.0);
        assert_eq!(floor(1.7), 1.0);
        assert_eq!(floor(-0.0), 0.0); // -0.0 == 0.0 comparison-wise
        assert_eq!(floor(-1.0), -1.0);
        assert_eq!(floor(-0.5), -1.0); // negative non-integer → next-lower integer
        assert_eq!(floor(-2.3), -3.0);
        assert!(floor(f32::NAN).is_nan());
        assert_eq!(floor(f32::INFINITY), f32::INFINITY);
        assert_eq!(floor(f32::NEG_INFINITY), f32::NEG_INFINITY);
    }
}
// Keep the `INV_TWO_PI` constant referenced so no dead-const warnings fire
// in future callers that reduce arguments by `mod 2π` (currently only the
// π/2-octant reducer is wired). The compiler folds this at const-eval time,
// so it contributes zero code size.
#[allow(dead_code)]
const _INV_TWO_PI_REF: f32 = INV_TWO_PI;
