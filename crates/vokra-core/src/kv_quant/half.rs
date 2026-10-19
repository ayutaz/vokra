//! Safe, zero-dep IEEE-754 binary16 (`f16`) helper for KV cache quantization
//! blocks.
//!
//! # Placement
//!
//! `vokra-core` is `unsafe_code = "deny"`, and we cannot pull the `half` crate
//! (NFR-DS-02, zero-dep invariant). This file duplicates the F16 semantics of
//! `crates/vokra-core/src/gguf/quant/mod.rs::f16_to_f32` intentionally: the two
//! call sites live in **different layers** (weight-quant K-quants vs KV-cache
//! Q_0 formats — M3-04 ADR D1) and re-exporting the weight-side helper would
//! erase that boundary.
//!
//! # Contract
//!
//! - [`F16Bits`] is a `#[repr(transparent)]` `u16` newtype whose in-memory
//!   representation is IEEE-754 binary16 in little-endian byte order.
//! - [`f32_to_f16_bits`] applies **round-to-nearest-even** with correct
//!   handling of subnormals, `±inf`, and `NaN`. Overflow → `±inf`; underflow →
//!   `±0` via subnormals.
//! - `f16 → f32` is exact for every representable f16 value.

/// IEEE-754 binary16, stored as its raw `u16` bit pattern.
///
/// Kept as a `repr(transparent)` newtype so a `[F16Bits; N]` field in a Q_0
/// block struct lays out the same as a `[u16; N]` — matches llama.cpp's
/// `ggml_fp16_t` on-wire representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(transparent)]
pub struct F16Bits(pub u16);

impl F16Bits {
    /// Decodes to `f32`. Exact for every representable value.
    #[inline]
    #[must_use]
    pub fn to_f32(self) -> f32 {
        f16_bits_to_f32(self.0)
    }

    /// Encodes an `f32` with round-to-nearest-even.
    #[inline]
    #[must_use]
    pub fn from_f32(x: f32) -> Self {
        Self(f32_to_f16_bits(x))
    }
}

/// IEEE-754 half → single precision (exact; handles subnormals, ±inf and NaN).
///
/// Matches the semantics of `crates/vokra-core/src/gguf/quant/mod.rs::f16_to_f32`
/// verbatim; the duplication is deliberate (see module docs).
#[inline]
#[must_use]
pub fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 1;
    let exp = (h >> 10) & 0x1f;
    let mant = h & 0x3ff;
    let sign_f = if sign == 1 { -1.0f32 } else { 1.0f32 };
    match exp {
        0 => sign_f * (mant as f32) * exp2i(-24), // subnormal / zero
        0x1f => {
            if mant == 0 {
                sign_f * f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => sign_f * (1.0 + (mant as f32) / 1024.0) * exp2i(exp as i32 - 15),
    }
}

/// Computes an exact f32 power of two for the exponent range used by binary16.
///
/// Every normal binary16 exponent maps to `[-14, 15]`, and the binary16
/// subnormal scale is `2^-24`; all of those powers are normal, exactly
/// representable f32 values. Constructing the IEEE exponent bits avoids a
/// platform/libm-dependent transcendental call.
#[inline]
fn exp2i(n: i32) -> f32 {
    debug_assert!((-126..=127).contains(&n));
    f32::from_bits(((n + 127) as u32) << 23)
}

/// IEEE-754 single → half precision with round-to-nearest-even.
///
/// - Overflow: `|x| > 65504` → `±inf`.
/// - Underflow: `|x| < 2^-24` → `±0` (via subnormals).
/// - NaN payload is not preserved; a NaN input returns a canonical quiet NaN.
///
/// The reference is IEEE-754 §5.4 rounding rules; the sanity tests below pin
/// the corners (1.0, subnormal edge, overflow, NaN, ±0).
#[must_use]
pub fn f32_to_f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 31) & 0x1) as u16;
    let exp_f32 = ((bits >> 23) & 0xff) as i32;
    let mant_f32 = bits & 0x007f_ffff;

    // NaN → canonical quiet NaN (preserve sign bit, mantissa = 0x200).
    if exp_f32 == 0xff {
        if mant_f32 == 0 {
            // ±inf
            return (sign << 15) | (0x1f << 10);
        }
        return (sign << 15) | (0x1f << 10) | 0x200;
    }

    // ±0 in f32 → ±0 in f16.
    if exp_f32 == 0 && mant_f32 == 0 {
        return sign << 15;
    }

    // Rebias exponent: f32 uses bias 127, f16 uses bias 15.
    let unbiased = exp_f32 - 127;
    let f16_exp = unbiased + 15;

    // Overflow → ±inf.
    if f16_exp >= 0x1f {
        return (sign << 15) | (0x1f << 10);
    }

    // Normal range for f16 (1 ≤ f16_exp ≤ 30).
    if f16_exp >= 1 {
        // Truncate 23-bit mantissa to 10-bit + round-to-nearest-even.
        let shift = 13;
        let round_bit = (mant_f32 >> (shift - 1)) & 1;
        let sticky = (mant_f32 & ((1u32 << (shift - 1)) - 1)) != 0;
        let truncated = mant_f32 >> shift;
        let round_up = round_bit == 1 && (sticky || (truncated & 1) == 1);
        let mut rounded = truncated + if round_up { 1 } else { 0 };
        let mut exp = f16_exp as u16;
        // Rounding may bump mantissa past 10-bit range → carry into exponent.
        if rounded == 0x400 {
            rounded = 0;
            exp += 1;
            if exp >= 0x1f {
                return (sign << 15) | (0x1f << 10);
            }
        }
        return (sign << 15) | (exp << 10) | (rounded as u16 & 0x3ff);
    }

    // Subnormal / underflow range.
    // f16 subnormal encodes `sign * mant * 2^-24` for `mant ∈ [1, 1023]`. If
    // f16_exp < -10 → underflow to ±0 (`2^-24` is the smallest positive f16).
    if f16_exp < -10 {
        return sign << 15;
    }
    // Include the implicit leading 1 in the f32 mantissa, then shift right so
    // the result fits the subnormal encoding (exp field = 0).
    let mant_with_implicit = mant_f32 | 0x0080_0000;
    let shift = (1 - f16_exp) as u32 + 13;
    let round_bit = (mant_with_implicit >> (shift - 1)) & 1;
    let sticky = (mant_with_implicit & ((1u32 << (shift - 1)) - 1)) != 0;
    let truncated = mant_with_implicit >> shift;
    let round_up = round_bit == 1 && (sticky || (truncated & 1) == 1);
    let rounded = truncated + if round_up { 1 } else { 0 };
    // If rounding lifted the subnormal into the normal range (rounded == 0x400)
    // it becomes the smallest normal (exp = 1, mantissa = 0).
    if rounded >= 0x400 {
        return (sign << 15) | (1 << 10);
    }
    (sign << 15) | (rounded as u16 & 0x3ff)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_positive_values_round_trip() {
        for x in [1.0f32, 2.0, -2.0, 0.5, 0.25, 3.25, 65504.0, -65504.0] {
            let bits = f32_to_f16_bits(x);
            let back = f16_bits_to_f32(bits);
            assert_eq!(back, x, "round trip failed for {x}");
        }
    }

    #[test]
    fn zero_signs_are_preserved() {
        assert_eq!(f32_to_f16_bits(0.0), 0);
        assert_eq!(f32_to_f16_bits(-0.0), 0x8000);
    }

    #[test]
    fn overflow_maps_to_inf() {
        assert_eq!(f32_to_f16_bits(1.0e5), 0x7C00);
        assert_eq!(f32_to_f16_bits(-1.0e5), 0xFC00);
        assert_eq!(f32_to_f16_bits(f32::INFINITY), 0x7C00);
    }

    #[test]
    fn underflow_maps_to_zero() {
        assert_eq!(f32_to_f16_bits(1.0e-10), 0);
        assert_eq!(f32_to_f16_bits(-1.0e-10), 0x8000);
    }

    #[test]
    fn nan_maps_to_canonical_nan() {
        let n = f32_to_f16_bits(f32::NAN);
        // exp = 0x1F, mantissa != 0
        assert_eq!((n >> 10) & 0x1F, 0x1F);
        assert_ne!(n & 0x3FF, 0);
        assert!(f16_bits_to_f32(n).is_nan());
    }

    #[test]
    fn subnormal_edge_is_representable() {
        // 2^-24 is the smallest positive f16 subnormal, exactly representable.
        let smallest_sub = f32::from_bits(0x3380_0000);
        let bits = f32_to_f16_bits(smallest_sub);
        assert_eq!(f16_bits_to_f32(bits), smallest_sub);
    }

    #[test]
    fn f16_to_f32_ieee_boundaries_are_exact() {
        // Independent raw IEEE-754 f32 oracles for representative binary16
        // boundaries. NaN payloads remain intentionally classification-only.
        let cases = [
            (0x0000, 0x0000_0000),
            (0x8000, 0x8000_0000),
            (0x0001, 0x3380_0000),
            (0x8001, 0xB380_0000),
            (0x03FF, 0x387F_C000),
            (0x83FF, 0xB87F_C000),
            (0x0400, 0x3880_0000),
            (0x8400, 0xB880_0000),
            (0x3800, 0x3F00_0000),
            (0x3C00, 0x3F80_0000),
            (0x4000, 0x4000_0000),
            (0xC000, 0xC000_0000),
            (0x7BFF, 0x477F_E000),
            (0xFBFF, 0xC77F_E000),
            (0x7C00, 0x7F80_0000),
            (0xFC00, 0xFF80_0000),
        ];
        for (input, expected) in cases {
            assert_eq!(
                f16_bits_to_f32(input).to_bits(),
                expected,
                "binary16 0x{input:04x}"
            );
        }
    }

    #[test]
    fn f16_bits_newtype_is_transparent_u16() {
        // repr(transparent) requirement: F16Bits and u16 have the same layout.
        assert_eq!(std::mem::size_of::<F16Bits>(), std::mem::size_of::<u16>());
        assert_eq!(std::mem::align_of::<F16Bits>(), std::mem::align_of::<u16>());
    }

    #[test]
    fn round_to_nearest_even_direction() {
        // 1.0 + 2^-11 lands exactly between two f16 values; round-to-even picks
        // the one with even mantissa. Exact f32 encoding: bits = 0x3F800800.
        let x = f32::from_bits(0x3F80_0800);
        let h = f32_to_f16_bits(x);
        // Expected: exp=15, mantissa=0 (i.e. 1.0 exactly), not mantissa=1.
        assert_eq!(h & 0x3FF, 0);
    }
}
