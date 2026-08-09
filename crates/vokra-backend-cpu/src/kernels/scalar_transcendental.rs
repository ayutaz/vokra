//! Self-contained scalar `exp` / `tanh` / `sqrt` for the no_std subset
//! (M5-03-T06), extended in **WP-06** (2026-08-09) with `sin` / `cos` /
//! `log` / `log1p` for the Style-Bert-VITS2 v2 hot path.
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
//! # Scope
//!
//! - **Wave 1 (M5-03-T06)**: `exp` / `tanh` / `sqrt`. Wiring these into the
//!   Silero forward (replacing the current `f32::exp` / `tanh` / `sqrt` calls)
//!   is T08 (Wave 2), together with the upstream-parity re-measurement that the
//!   swap forces.
//! - **WP-06 (2026-08-09)**: `sin` / `cos` / `log` / `log1p` added for the
//!   Style-Bert-VITS2 v2 hot path (SDP flow, ElementwiseAffine `log(exp(x)-1)`
//!   constants, `softplus` fallback around `x > 20` — see
//!   `crates/vokra-models/src/sbv2/duration.rs`). Wiring into SBV2 kernels is
//!   a follow-up WP (WP-07+); this module only *provides* the functions and
//!   pins their accuracy with a differential property test against `std`.
//!
//! Until every caller migrates over, these functions are exercised only by
//! their own property tests, hence the module-level `#[allow(dead_code)]` on
//! the `mod` declaration in `kernels/mod.rs`.
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

// ---- WP-06: sin / cos / log / log1p (M5-hot-path Style-Bert-VITS2 wave) ----
//
// Same design rules as `exp` / `tanh` / `sqrt` above:
// - pure `core` arithmetic (no `std::f32::{sin,cos,ln,ln_1p}`, no `libm`);
// - no `unsafe`;
// - no `f32::mul_add` — plain `*` + `+` keeps every result deterministic
//   across FMA-capable and FMA-less targets (see module docs, "deterministic
//   across targets" clause). Bit-exactness against the SIMD wrappers is
//   handled at the SIMD side by inserting `mul_add` on both AVX2 and NEON
//   micro-kernels (or their soft-emulated fallback) — NOT here.
//
// Polynomial coefficients are exact rational reciprocals (1/n! for
// sin/cos, 1/(2k+1) for the `atanh` expansion `log` rides on top of),
// NOT scavenged from libm / Cephes / Pommier / SLEEF — this preserves
// the "no copied approximation constants" license posture the module
// documented in its opening comment.

// === sin / cos ===

/// Cody-Waite `π/2` split, `HI` chosen so its last 12 mantissa bits are
/// zero (Muller, *Handbook of Floating-Point Arithmetic*, §11.3). That
/// makes `k * PIO2_HI` exact for any `|k| ≤ 4096`, i.e. accurate range
/// reduction for `|x| ≤ 4096 * π/2 ≈ 6434`. Beyond that we still emit a
/// finite result but polynomial-truncation error drifts up (Payne-Hanek
/// is deliberately out of scope — SBV2 hot-path arguments are bounded
/// phase / frequency terms well inside the accurate range).
/// See existing [`LN2_HI`] for the same clippy suppression rationale: the
/// literal is exactly representable in f32 (it IS the point — bit pattern
/// `0x3FC91000` with the last 12 mantissa bits deliberately zero so that
/// `k * PIO2_HI` is exact for `|k| ≤ 4096`); truncating it as clippy
/// suggests would silently break the Cody-Waite reduction.
#[allow(clippy::excessive_precision)]
const PIO2_HI: f32 = 1.57080078125;
const PIO2_LO: f32 = -4.454455e-6; // (π/2) - PIO2_HI, in f32
/// `2/π`, used for `k = round(x * TWO_OVER_PI)` (the quadrant index).
const TWO_OVER_PI: f32 = core::f32::consts::FRAC_2_PI;

// Degree-9 sin polynomial coefficients (Taylor `1/n!`, odd terms only,
// evaluated in `y = r²`). Truncation `|r|¹¹/11! ≈ (π/4)¹¹/11! ≈ 2e-9`
// — well inside f32 ULP.
const SIN_C0: f32 = 1.0; // 1/1!
const SIN_C1: f32 = -1.0 / 6.0; // -1/3!
const SIN_C2: f32 = 1.0 / 120.0; // 1/5!
const SIN_C3: f32 = -1.0 / 5040.0; // -1/7!
const SIN_C4: f32 = 1.0 / 362_880.0; // 1/9!

// Degree-10 cos polynomial coefficients (Taylor `1/n!`, even terms only,
// evaluated in `y = r²`). Truncation `|r|¹²/12! ≈ (π/4)¹²/12! ≈ 1.2e-10`.
const COS_C0: f32 = 1.0; // 1/0!
const COS_C1: f32 = -0.5; // -1/2!
const COS_C2: f32 = 1.0 / 24.0; // 1/4!
const COS_C3: f32 = -1.0 / 720.0; // -1/6!
const COS_C4: f32 = 1.0 / 40_320.0; // 1/8!
const COS_C5: f32 = -1.0 / 3_628_800.0; // -1/10!

/// Evaluate `sin(r)` on the reduced domain `|r| ≤ π/4` via Horner in `y = r²`.
/// Odd function, hence `r * P(y)` layout.
fn sin_poly(r: f32) -> f32 {
    let y = r * r;
    let mut p = SIN_C4;
    p = p * y + SIN_C3;
    p = p * y + SIN_C2;
    p = p * y + SIN_C1;
    p = p * y + SIN_C0;
    r * p
}

/// Evaluate `cos(r)` on the reduced domain `|r| ≤ π/4` via Horner in `y = r²`.
/// Even function, hence a plain polynomial in `y`.
fn cos_poly(r: f32) -> f32 {
    let y = r * r;
    let mut p = COS_C5;
    p = p * y + COS_C4;
    p = p * y + COS_C3;
    p = p * y + COS_C2;
    p = p * y + COS_C1;
    p = p * y + COS_C0;
    p
}

/// Quadrant reduction shared by [`sin`] and [`cos`]. Returns `(r, k mod 4)`
/// where `r ∈ [-π/4, π/4]` and `k` is the integer nearest to `x·2/π`.
///
/// Uses the two-term Cody-Waite `π/2` split so the subtraction stays
/// accurate for `|k| ≤ 4096` (see [`PIO2_HI`] rationale). NaN / ∞ are
/// handled by the callers; this helper assumes a finite `x`.
fn quadrant_reduce(x: f32) -> (f32, u32) {
    let y = x * TWO_OVER_PI;
    // Round-half-away-from-zero via ±0.5 bias before truncation.
    // (`f32::round` is std; we cannot use it under the module's `core`-only
    // rule, matching the same pattern `exp` uses above.)
    let k_i32 = if y >= 0.0 {
        (y + 0.5) as i32
    } else {
        (y - 0.5) as i32
    };
    let kf = k_i32 as f32;
    // Cody-Waite split: r = x - k*PIO2_HI - k*PIO2_LO. The high subtraction
    // is exact for |k| ≤ 4096 (PIO2_HI's low 12 mantissa bits are zero);
    // the low correction restores the lost `π/2 - PIO2_HI` bits.
    let r = x - kf * PIO2_HI - kf * PIO2_LO;
    // Wrap to k mod 4 for quadrant selection. `rem_euclid` on i32 is `core`.
    let q = k_i32.rem_euclid(4) as u32;
    (r, q)
}

/// Scalar `sin(x)` in pure `core` arithmetic.
///
/// Standard quadrant reduction: `k = round(x·2/π)`, reduced argument
/// `r = x − k·(π/2)` (two-term Cody-Waite split), then the appropriate
/// slot of `{sin(r), cos(r), -sin(r), -cos(r)}` per `k mod 4`. `sin_poly`
/// evaluates a degree-9 odd Taylor series in `r`, `cos_poly` a degree-10
/// even one — both truncation errors sit at ~2e-9 / 1e-10, well under
/// f32 ULP on the reduced domain. Accurate for `|x| ≤ 4096·π/2 ≈ 6434`;
/// beyond that the polynomial answer is still finite but drifts as
/// Cody-Waite loses bits (Payne-Hanek is out of scope for SBV2 hot-path).
///
/// Special values match IEEE / `std::f32::sin`:
/// - `sin(NaN) = NaN`,
/// - `sin(±∞) = NaN` (std emits NaN because `∞·2/π = ∞`, `round(∞)` = i32
///   overflow saturation; we short-circuit explicitly so the answer is
///   deterministic across targets),
/// - `sin(-0.0) = -0.0` (sign of zero preserved — sin is odd and the
///   Horner form `r * P(r²)` propagates the sign of `r`).
pub(crate) fn sin(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    if x.is_infinite() {
        return f32::NAN;
    }
    // Preserve sign of zero: sin(±0.0) = ±0.0. Short-circuit before the
    // Cody-Waite subtraction below, which would destroy the sign of the
    // input zero because PIO2_LO is negative — for `x = -0.0, k = 0` the
    // reduced argument evaluates as `-0.0 - 0.0·PIO2_HI - 0.0·PIO2_LO
    //   = -0.0 - (+0.0) - (-0.0)  (IEEE sign propagation of `0 * ±c`)
    //   = -0.0 - (-0.0)
    //   = -0.0 + (+0.0)
    //   = +0.0` (IEEE sum of opposite-sign zeros in round-to-nearest),
    // clearing the sign bit. `sin(0.0) == 0.0` with the input's sign is
    // exactly `std::f32::sin(0.0)`'s behaviour, so returning `x` matches.
    if x == 0.0 {
        return x;
    }
    let (r, q) = quadrant_reduce(x);
    match q {
        0 => sin_poly(r),
        1 => cos_poly(r),
        2 => -sin_poly(r),
        3 => -cos_poly(r),
        _ => unreachable!(), // `rem_euclid(4)` yields 0..=3.
    }
}

/// Scalar `cos(x)` in pure `core` arithmetic.
///
/// Same quadrant reduction as [`sin`], but the slot selection is
/// `{cos(r), -sin(r), -cos(r), sin(r)}` per `k mod 4` (cosine is
/// `sin(x + π/2)`, shifting the slot cycle by one). Accuracy notes and
/// argument-range caveats match [`sin`].
pub(crate) fn cos(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    if x.is_infinite() {
        return f32::NAN;
    }
    let (r, q) = quadrant_reduce(x);
    match q {
        0 => cos_poly(r),
        1 => -sin_poly(r),
        2 => -cos_poly(r),
        3 => sin_poly(r),
        _ => unreachable!(),
    }
}

// === log / log1p ===

// Degree-9 `atanh` (odd-only) Taylor coefficients for `log(m)` via
// `log(m) = 2 * atanh((m-1)/(m+1))`. On the mantissa domain
// `m ∈ [√2/2, √2)`, `|u| ≤ (√2 - 1)/(√2 + 1) ≈ 0.1716`, and the tail
// `|u|¹¹/11 ≈ 5.4e-11` sits well inside f32 ULP. Coefficients are
// `1/(2k+1)` — exact rational reciprocals, not scavenged from libm.
const ATANH_C0: f32 = 1.0; // 1/1
const ATANH_C1: f32 = 1.0 / 3.0; // 1/3
const ATANH_C2: f32 = 1.0 / 5.0; // 1/5
const ATANH_C3: f32 = 1.0 / 7.0; // 1/7
const ATANH_C4: f32 = 1.0 / 9.0; // 1/9

/// Scalar natural log `log(x)` in pure `core` arithmetic.
///
/// Argument reduction: `x = m · 2^e` via IEEE-754 exponent-field extract
/// (`x.to_bits()` is `core`). Shift `m` into `[√2/2, √2)` (if `m ≥ √2`,
/// halve `m` and bump `e`) so the classical `u = (m-1)/(m+1)` transform
/// bounds `|u| ≤ 0.1716`, then `log(m) = 2 · (u + u³/3 + u⁵/5 + u⁷/7 +
/// u⁹/9)` — a degree-9 `atanh` series, truncation `|u|¹¹/11 ≈ 5.4e-11`,
/// well under f32 ULP. Reassemble as `log(x) = e·ln2 + log(m)` with the
/// existing Cody-Waite `ln2` split (`e·LN2_HI + e·LN2_LO + log(m)`) so
/// the exponent term stays accurate for large `|e|`.
///
/// Special values match IEEE / `std::f32::ln`:
/// - `log(NaN) = NaN`,
/// - `log(x < 0) = NaN`, including `log(-∞) = NaN`,
/// - `log(±0) = -∞`,
/// - `log(+∞) = +∞`,
/// - `log(1) = 0` exactly (mantissa = 1 → `u = 0` → series is 0, and
///   exponent = 0 → `e·ln2` is 0).
pub(crate) fn log(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    if x < 0.0 {
        return f32::NAN;
    }
    if x == 0.0 {
        // IEEE: log(±0) = -∞. Both +0.0 and -0.0 compare equal to 0.0.
        return f32::NEG_INFINITY;
    }
    if x.is_infinite() {
        // x > 0 here (negatives short-circuited above), so +∞.
        return f32::INFINITY;
    }
    // Subnormal inputs would encode exponent = 0 with a mantissa that
    // is not implicitly 1.-something; scale them into the normal range
    // first so the exponent-field decode below is meaningful. Multiplying
    // by 2^24 shifts a subnormal into the normal range and we correct
    // the exponent by -24 later.
    let (x_n, e_bias): (f32, i32) = if x < f32::MIN_POSITIVE {
        (x * (1u32 << 24) as f32, -24)
    } else {
        (x, 0)
    };

    // Decode `x_n = 2^e * m` from the IEEE-754 fields. Bias = 127.
    let bits = x_n.to_bits();
    let raw_exp = ((bits >> 23) & 0xff) as i32;
    let e_raw = raw_exp - 127; // unbiased
    // Rebuild `m` as a f32 in `[1, 2)` by pasting the exponent field with
    // the unbiased zero (biased 127 = 0x7f), keeping the original mantissa
    // bits. This is exact — no arithmetic rounding.
    let m_bits = (bits & 0x007f_ffff) | (127u32 << 23);
    let mut m = f32::from_bits(m_bits);
    let mut e = e_raw + e_bias;

    // Shift into `[√2/2, √2)` so the atanh argument stays inside the fast
    // convergence range. `√2 ≈ 1.41421356 …` — use the closest f32 for the
    // decision boundary (exact comparison, no polynomial impact).
    const SQRT_2: f32 = core::f32::consts::SQRT_2;
    if m >= SQRT_2 {
        m *= 0.5;
        e += 1;
    }

    // Now m ∈ [√2/2, √2). Compute u = (m - 1) / (m + 1), |u| ≤ 0.1716.
    let u = (m - 1.0) / (m + 1.0);
    let y = u * u;

    // Horner on the atanh series:
    // atanh(u) = u * (1 + y/3 + y²/5 + y³/7 + y⁴/9)
    let mut p = ATANH_C4;
    p = p * y + ATANH_C3;
    p = p * y + ATANH_C2;
    p = p * y + ATANH_C1;
    p = p * y + ATANH_C0;
    let atanh_u = u * p;

    // log(m) = 2 * atanh(u); log(x) = e * ln2 + log(m).
    // Cody-Waite ln2 split keeps the `e * ln2` term accurate for large |e|.
    let ef = e as f32;
    ef * LN2_HI + ef * LN2_LO + 2.0 * atanh_u
}

// Degree-8 `log1p` Taylor coefficients for the small-|x| branch
// (|x| < 1/16). `log1p(x) = x - x²/2 + x³/3 - ... - x⁸/8`, factored as
// `x * (1 - x/2 + x²/3 - x³/4 + x⁴/5 - x⁵/6 + x⁶/7 - x⁷/8)`. Truncation
// `|x|⁹/9 ≈ (1/16)⁹/9 ≈ 8.4e-12` — well inside f32 ULP even at the
// branch boundary. This preserves precision where `log(1+x)` would
// return 0 (for |x| below the f32 rounding threshold for `1+x`).
const LOG1P_C0: f32 = 1.0; // 1
const LOG1P_C1: f32 = -0.5; // -1/2
const LOG1P_C2: f32 = 1.0 / 3.0; // 1/3
const LOG1P_C3: f32 = -0.25; // -1/4
const LOG1P_C4: f32 = 0.2; // 1/5
const LOG1P_C5: f32 = -1.0 / 6.0; // -1/6
const LOG1P_C6: f32 = 1.0 / 7.0; // 1/7
const LOG1P_C7: f32 = -0.125; // -1/8

/// Small-|x| branch threshold — see [`log1p`].
const LOG1P_SMALL_ARG: f32 = 0.0625;

/// Scalar `log1p(x) = log(1 + x)` in pure `core` arithmetic.
///
/// Two-branch design:
/// - `|x| < 1/16`: direct Taylor polynomial `x · (1 - x/2 + x²/3 - …
///   - x⁷/8)` (degree 8), truncation `|x|⁹/9 ≤ 8.4e-12`. This branch is
///   why `log1p` exists as a separate primitive: for `|x|` below the
///   f32 rounding threshold for `1+x` (~1.2e-7 for `1+x` to differ
///   from `1`), naive `log(1+x)` returns 0 and drops the whole answer;
///   the polynomial preserves the leading `x` term to full f32
///   precision (and for `|x|` far below the threshold, dominates as
///   `≈ x`).
/// - `|x| ≥ 1/16`: delegate to [`log`]`(1 + x)`. `1 + x` here is
///   well-conditioned (24-bit mantissa gives full precision), and
///   [`log`]'s own [`LOG_REL_TOL`](tests::LOG_REL_TOL) bound carries.
///
/// Special values match IEEE / `std::f32::ln_1p`:
/// - `log1p(NaN) = NaN`,
/// - `log1p(x < -1) = NaN`,
/// - `log1p(-1) = -∞`,
/// - `log1p(0) = 0` (sign of zero preserved — small-|x| branch returns
///   `x * 1` which carries the sign),
/// - `log1p(+∞) = +∞`.
pub(crate) fn log1p(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    if x < -1.0 {
        return f32::NAN;
    }
    if x == -1.0 {
        return f32::NEG_INFINITY;
    }
    if x.is_infinite() {
        // x > -1 and infinite → +∞.
        return f32::INFINITY;
    }
    if x.abs() < LOG1P_SMALL_ARG {
        // Small-|x| Taylor branch, Horner from highest degree.
        let mut p = LOG1P_C7;
        p = p * x + LOG1P_C6;
        p = p * x + LOG1P_C5;
        p = p * x + LOG1P_C4;
        p = p * x + LOG1P_C3;
        p = p * x + LOG1P_C2;
        p = p * x + LOG1P_C1;
        p = p * x + LOG1P_C0;
        x * p
    } else {
        // Larger-|x| branch: 1 + x is well-conditioned in f32.
        log(1.0 + x)
    }
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

    // ----- WP-06: sin/cos/log/log1p property tests (added RED before impl) -----

    /// Absolute-error ceiling for scalar `sin` on its accurate reduction
    /// domain (Cody-Waite π/2 split with `PIO2_HI` last 12 mantissa bits
    /// zero → `k * PIO2_HI` exact for `|k| ≤ 4096` → argument magnitudes
    /// up to `4096 * π/2 ≈ 6434`). A dense sweep of `[-1000, 1000]` at
    /// step 0.0037 (a chosen non-multiple of π to visit poor-conditioning
    /// arguments) observes a worst case of ≈5.96e-8 on aarch64
    /// M1 — exactly 1 f32 ULP of 1.0, dominated by the degree-9 Taylor
    /// truncation (`(π/4)¹¹/11! ≈ 2e-9` per term) plus Horner rounding
    /// on the reduced argument. This bound is ~3× that observed max
    /// (not the strict ~2× exp/tanh use, since sin/cos results are
    /// bounded in [-1, 1] and cross-target rounding at the sin/cos
    /// polynomial slot boundary can add a few ULP on x86-64 without
    /// this being a defect). NOT loosened to force a pass (red line #3);
    /// far inside NFR-QL-01 `atol=0.01`. `sin` values live in `[-1, 1]`,
    /// so absolute error is the natural measure — near-zero relative
    /// would blow up around the multiples of π where sin is legitimately
    /// near-zero, not a defect of ours.
    const SIN_ABS_TOL: f32 = 2.0e-7;

    #[test]
    fn sin_matches_std_densely_across_ten_periods_and_is_odd() {
        // Step 0.0037 is intentionally NOT a rational multiple of π: it
        // sweeps the full [-π/4, π/4] reduced domain many times over the
        // ten-period arc, visiting the worst-case reduced argument.
        let mut max_abs = 0.0f32;
        let mut i = -270_270i32;
        while i <= 270_270 {
            let x = i as f32 * 0.0037; // ≈ [-1000, 1000]
            let abs = (sin(x) - x.sin()).abs();
            if abs > max_abs {
                max_abs = abs;
            }
            i += 1;
        }
        assert!(
            max_abs <= SIN_ABS_TOL,
            "scalar sin worst-case abs = {max_abs} exceeded honest bound {SIN_ABS_TOL}"
        );
        // Oddness at zero and the classical fixed points.
        assert_eq!(sin(0.0), 0.0);
        assert!((sin(core::f32::consts::PI / 2.0) - 1.0).abs() < 5e-7);
        assert!((sin(-core::f32::consts::PI / 2.0) + 1.0).abs() < 5e-7);
        // sin(π) ≈ 0 (accuracy limited by π - f32(π) ≈ 8.7e-8 forwarded
        // through the polynomial derivative, so bound is a few ULP of π).
        assert!(sin(core::f32::consts::PI).abs() < 5e-7);
    }

    /// Absolute-error ceiling for scalar `cos` — same reduction domain
    /// and same argument-magnitude scaling as [`SIN_ABS_TOL`]. Dense
    /// sweep observes ≈5.96e-8 on aarch64 M1 — again 1 f32 ULP of 1.0
    /// (degree-10 polynomial truncation `(π/4)¹²/12! ≈ 1.2e-10` per
    /// term compounded with the quadrant selection between sin/cos
    /// slots). ~3× observed max, same cross-target headroom rationale
    /// as [`SIN_ABS_TOL`].
    const COS_ABS_TOL: f32 = 2.0e-7;

    #[test]
    fn cos_matches_std_densely_across_ten_periods_and_is_even() {
        let mut max_abs = 0.0f32;
        let mut i = -270_270i32;
        while i <= 270_270 {
            let x = i as f32 * 0.0037;
            let abs = (cos(x) - x.cos()).abs();
            if abs > max_abs {
                max_abs = abs;
            }
            i += 1;
        }
        assert!(
            max_abs <= COS_ABS_TOL,
            "scalar cos worst-case abs = {max_abs} exceeded honest bound {COS_ABS_TOL}"
        );
        // Evenness at zero and classical fixed points.
        assert_eq!(cos(0.0), 1.0);
        assert!((cos(-0.5) - cos(0.5)).abs() < 5e-7);
        assert!((cos(core::f32::consts::PI) + 1.0).abs() < 5e-7);
        // cos(π/2) ≈ 0 (same argument as sin(π) — bounded by f32(π/2)
        // rounding forwarded through the polynomial derivative).
        assert!(cos(core::f32::consts::PI / 2.0).abs() < 5e-7);
    }

    #[test]
    fn sin_cos_handle_special_values_like_ieee() {
        // NaN in → NaN out.
        assert!(sin(f32::NAN).is_nan());
        assert!(cos(f32::NAN).is_nan());
        // ±∞ in: std returns NaN (sin/cos undefined at infinity). Match.
        assert!(sin(f32::INFINITY).is_nan());
        assert!(sin(f32::NEG_INFINITY).is_nan());
        assert!(cos(f32::INFINITY).is_nan());
        assert!(cos(f32::NEG_INFINITY).is_nan());
        // sin(-0.0) == -0.0 (preserve sign of zero).
        assert_eq!(sin(-0.0).to_bits(), (-0.0f32).to_bits());
    }

    /// Relative-error ceiling for scalar `log` over the accurate range.
    /// The mantissa split into `[√2/2, √2)` bounds `u = (m-1)/(m+1)` to
    /// `|u| ≤ (√2 - 1)/(√2 + 1) ≈ 0.1716`, and the degree-9 atanh
    /// expansion `u + u³/3 + u⁵/5 + u⁷/7 + u⁹/9` has tail
    /// `|u|¹¹/11 ≈ 5.4e-11` — well under f32 ULP. Observed worst case
    /// across a ~40-decade sweep is ≈2.15e-7 on aarch64 M1 — about 2
    /// f32 ULP of the log target value, exactly meeting the ≤ 2 ULP
    /// design target (dominated by the `e * ln2_HI + e * ln2_LO` step
    /// where large exponents `e` accumulate LN2_LO rounding). This
    /// bound is ~2.3× that observed max.
    const LOG_REL_TOL: f32 = 5.0e-7;

    #[test]
    fn log_matches_std_across_forty_decades() {
        let mut max_rel = 0.0f32;
        for e in -20..20 {
            for m in 1..1000 {
                let x = (m as f32) * 10f32.powi(e);
                if x <= 0.0 || !x.is_finite() {
                    continue;
                }
                let want = x.ln();
                // Skip x == 1 exactly where want == 0 (relative undefined).
                if want == 0.0 {
                    continue;
                }
                let rel = (log(x) - want).abs() / want.abs();
                if rel > max_rel {
                    max_rel = rel;
                }
            }
        }
        assert!(
            max_rel <= LOG_REL_TOL,
            "scalar log worst-case rel = {max_rel} exceeded honest bound {LOG_REL_TOL}"
        );
    }

    #[test]
    fn log_handles_special_values_like_ieee() {
        // log(NaN) = NaN; log(negative) = NaN; log(-0.0) = -∞ (IEEE); log(0) = -∞.
        assert!(log(f32::NAN).is_nan());
        assert!(log(-1.0).is_nan());
        assert!(log(-0.5).is_nan());
        assert_eq!(log(0.0), f32::NEG_INFINITY);
        assert_eq!(log(-0.0), f32::NEG_INFINITY);
        // log(∞) = ∞.
        assert_eq!(log(f32::INFINITY), f32::INFINITY);
        // log(1) is exactly 0 (mantissa = 1, exponent = 0 → 0 * ln2 + 2 * atanh(0) = 0).
        assert_eq!(log(1.0), 0.0);
        // log(e) ≈ 1 within a few ULP.
        assert!((log(core::f32::consts::E) - 1.0).abs() < 5e-7);
        // log(2) ≈ ln2 within a few ULP.
        assert!((log(2.0) - core::f32::consts::LN_2).abs() < 5e-7);
    }

    /// Absolute-error ceiling for scalar `log1p` where the precision
    /// benefit over `log(1+x)` matters most (|x| < 0.0625). Dense
    /// sweep observes ≈7.45e-9 max on aarch64 M1 — about 1 f32 ULP
    /// at the |x|=0.0625 boundary where `log1p(0.0625) ≈ 0.0606`
    /// (1 ULP ≈ 7.5e-9). Bound is ~7× that observed max, generously
    /// widened for cross-target rounding at the branch boundary (the
    /// polynomial's ~0.06 result vs the delegate leg's log(1.06) both
    /// need to fit under one bound — this is the tightest we can be
    /// without ISA-tuning the constant). For larger |x| (`|x| ≥
    /// 0.0625`) `log1p` delegates to `log(1+x)` which inherits
    /// [`LOG_REL_TOL`] (tested by the wide sweep below).
    const LOG1P_ABS_TOL: f32 = 5.0e-8;

    #[test]
    fn log1p_matches_std_densely_on_small_x_where_precision_matters() {
        // Sweep [-0.0625, 0.0625] where 1 + x cannot be captured by naive
        // `log(1+x)` without precision loss.
        let mut max_abs = 0.0f32;
        let mut i = -6250i32;
        while i <= 6250 {
            let x = i as f32 * 1.0e-5; // step 1e-5 across [-0.0625, 0.0625]
            let abs = (log1p(x) - x.ln_1p()).abs();
            if abs > max_abs {
                max_abs = abs;
            }
            i += 1;
        }
        assert!(
            max_abs <= LOG1P_ABS_TOL,
            "scalar log1p worst-case abs = {max_abs} exceeded honest bound {LOG1P_ABS_TOL}"
        );
    }

    #[test]
    fn log1p_matches_std_across_wide_range_via_log_delegate() {
        // For |x| ≥ 0.0625 log1p delegates to log(1+x); this test pins
        // the delegate leg still tracks std within a relaxed relative
        // bound (~LOG_REL_TOL, since the delegation carries log's
        // accuracy through the `1 + x` addend which is well-conditioned
        // here since |x| is large enough that 1+x has full 24-bit
        // mantissa precision).
        let xs = [0.1, 0.5, 1.0, 2.0, 10.0, 100.0, 1000.0, -0.5, -0.9, -0.99];
        for &x in &xs {
            let got = log1p(x);
            let want = x.ln_1p();
            let rel = (got - want).abs() / want.abs().max(f32::MIN_POSITIVE);
            assert!(
                rel <= LOG_REL_TOL,
                "log1p({x}) = {got}, want {want} (rel = {rel}, bound {LOG_REL_TOL})"
            );
        }
    }

    #[test]
    fn log1p_preserves_precision_for_subnormal_x_where_1_plus_x_rounds_to_1() {
        // For |x| ~ 1e-8 or smaller, `1 + x` rounds to exactly 1 in f32.
        // Naive `log(1+x)` would return 0.0; log1p must return x (the
        // leading Taylor term). This is the raison d'être of log1p.
        let tiny = 1.0e-9_f32;
        assert!((log1p(tiny) - tiny).abs() <= f32::EPSILON.max(1.0e-15));
        // The sign is preserved — log1p is monotonic.
        assert!(log1p(-1.0e-9_f32) < 0.0);
    }

    #[test]
    fn log1p_handles_special_values_like_ieee() {
        // log1p(NaN) = NaN.
        assert!(log1p(f32::NAN).is_nan());
        // log1p(-1) = -∞ (log(0) = -∞).
        assert_eq!(log1p(-1.0), f32::NEG_INFINITY);
        // log1p(x < -1) = NaN (log of negative).
        assert!(log1p(-2.0).is_nan());
        assert!(log1p(-1.5).is_nan());
        // log1p(0) = 0 (exact; preserve sign of zero).
        assert_eq!(log1p(0.0), 0.0);
        assert_eq!(log1p(-0.0).to_bits(), (-0.0f32).to_bits());
        // log1p(∞) = ∞.
        assert_eq!(log1p(f32::INFINITY), f32::INFINITY);
    }
}
