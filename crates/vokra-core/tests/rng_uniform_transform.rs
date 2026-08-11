//! Integration tests for [`u32_to_uniform_f32_pytorch`] — the u32 → f32
//! uniform bridge that feeds Box-Muller inside torch's
//! `PhiloxRNGEngine.h::randn` (BSD-3-Clause).
//!
//! `excessive_precision` allowed at the file level: the test literals
//! reproduce torch's exact `4.6566127342e-10f` C++ decimal so a future
//! maintainer diffing against upstream sees a byte-identical constant,
//! even though the trailing digits round to the same f32 (see
//! `SCALE`'s doc in `normal_kernel.rs`).
#![allow(clippy::excessive_precision)]
//!
//! The mapping is `((v & 0x7FFF_FFFF) as f32) * SCALE` with `SCALE =
//! 4.6566127342e-10f`. Torch takes the low 31 bits (masking off the sign
//! bit) and multiplies by that literal; the result lies in `[0, 1)`, and
//! bit-exact matching torch requires (a) the same 31-bit mask, (b) the
//! same SCALE literal (which must parse to `0x2FFF_FFFF` as an f32), and
//! (c) the same multiplication order (mask, cast, multiply — casting the
//! product would double-round).
//!
//! # Bit-pattern derivation
//!
//! `4.6566127342e-10f` in C++ is a `constexpr float` — the compiler
//! evaluates the decimal literal at parse time and rounds to the nearest
//! f32. The nearest f32 to `4.6566127342e-10` has the bit pattern
//! `0x2FFF_FFFF` (verified by running `println!("{:#010x}",
//! 4.6566127342e-10_f32.to_bits())` — see scratchpad/probe_scale.rs).
//! Rust's literal parser goes through f64 then narrows to f32 with the
//! same round-to-nearest-even rule, so both languages land on the same
//! bit pattern.

use vokra_core::rng::u32_to_uniform_f32_pytorch;

/// The all-zero u32 must map to 0.0 exactly — the boundary of the closed
/// half-open `[0, 1)` interval.
#[test]
fn uniform_boundary_zero() {
    assert_eq!(u32_to_uniform_f32_pytorch(0), 0.0_f32);
}

/// The SCALE literal `4.6566127342e-10f` must parse to the f32 with bit
/// pattern `0x2FFF_FFFF`. If a future Rust version silently promoted the
/// literal to f64 (say via an implicit widening in the multiplication),
/// this test would fail even though the surface API kept working.
///
/// Bit-exact pin against a `constexpr float` in C++ — mirrors torch's
/// `PhiloxRNGEngine.h`:
///
/// ```cpp
/// static constexpr float scale = 4.6566127342e-10f;
/// ```
#[test]
fn uniform_scale_is_bit_exact_c_constexpr() {
    let scale: f32 = 4.6566127342e-10;
    assert_eq!(
        scale.to_bits(),
        0x2FFF_FFFFu32,
        "SCALE literal must have the same f32 bit pattern as torch's \
         constexpr float; a mismatch here means Rust's literal parser \
         has silently produced a different value"
    );
}

/// The sign bit (`0x8000_0000`) must be dropped by the mask so that
/// negative-looking u32s map to the same value as the corresponding
/// low-31-bits u32. Isolates the "mask before cast" failure mode from
/// the SCALE-literal failure mode above: if the mask were missing, this
/// test would fail with a value near 1.0 (the top half of u32-as-f32
/// times SCALE lands near 1.0, well distinguishable from 0.0).
#[test]
fn uniform_high_bit_masked_off() {
    assert_eq!(
        u32_to_uniform_f32_pytorch(0x8000_0000),
        u32_to_uniform_f32_pytorch(0),
        "the sign bit must be masked off before the cast — \
         0x80000000 and 0 differ only in the sign bit"
    );
    // Same for the u32::MAX case (all-ones).
    assert_eq!(
        u32_to_uniform_f32_pytorch(0xFFFF_FFFF),
        u32_to_uniform_f32_pytorch(0x7FFF_FFFF),
        "0xFFFFFFFF and 0x7FFFFFFF differ only in the sign bit"
    );
}

/// The maximum value the function can return (u=0x7FFF_FFFF, the largest
/// low-31-bits pattern) is strictly less than 1.0 — the half-open
/// interval's upper boundary. Torch's `randn` then does `1 - u1` to
/// shift into `(0, 1]` so `ln(u1)` is finite; if u1 could equal 0.0
/// exactly at `u=0x7FFF_FFFF`, the shift would leave a 1.0 that
/// `ln(1.0)=0` — silently OK, but the maximum before the shift is what
/// matters for the interval convention.
#[test]
fn uniform_max_31bit_is_below_one() {
    let expected: f32 = (0x7FFF_FFFFu32 as f32) * 4.6566127342e-10_f32;
    let got = u32_to_uniform_f32_pytorch(0xFFFF_FFFF);
    assert_eq!(
        got, expected,
        "explicit mask-then-scale must match hand-rolled"
    );
    assert!(got < 1.0, "u32::MAX must not map to 1.0 exactly");
    // The exact expected bit pattern from scratchpad/probe_scale.rs:
    // SCALE * 0x7FFF_FFFF = 0.99999994 (bits 0x3F7F_FFFF).
    assert_eq!(
        got.to_bits(),
        0x3F7F_FFFFu32,
        "u32::MAX maps to 0.99999994 (bits 0x3F7FFFFF)"
    );
}
