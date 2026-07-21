//! Semiring abstraction for WFST decoding (M5-06 T03).
//!
//! A [`Semiring`] is the algebraic structure `(K, ⊕, ⊗, 0̄, 1̄)` over which a
//! weighted finite-state transducer computes. Best-path (Viterbi) decoding
//! needs only the **tropical** semiring `(ℝ ∪ {+∞}, min, +, +∞, 0)`:
//!
//! - `⊕` = `min` — combine alternative paths, keeping the cheaper one;
//! - `⊗` = `+`   — accumulate the cost along a single path;
//! - `0̄` = `+∞`  — the additive identity / multiplicative annihilator
//!   ("no path"): `min(x, +∞) = x` and `x + (+∞) = +∞`;
//! - `1̄` = `0`   — the multiplicative identity ("free"): `x + 0 = x`.
//!
//! Weights are **costs** (negative log-probabilities): lower is better, and the
//! best path minimises the total cost. This matches OpenFST's `standard`
//! (tropical) arc type, so committed OpenFST fixtures are read verbatim.
//!
//! # Why a trait if only tropical is implemented?
//!
//! The `log` semiring `(ℝ ∪ {±∞}, ⊕_log, +, +∞, 0)` with
//! `x ⊕_log y = −log(e^−x + e^−y)` is needed for posteriors / confidence /
//! forward-backward, but **not** for Viterbi. Per ADR M5-06 it is a deliberate
//! future additive: the trait is carved now so a `LogWeight` variant lands
//! later without reshaping the decoder. This module ships **only**
//! [`TropicalWeight`].
//!
//! # These tests are *algebra*, not numerical parity
//!
//! The unit tests below check the semiring axioms (associativity, identities,
//! annihilation). That is algebraic verification of a closed-form structure —
//! it is **not** a numerical-parity test against an external reference (those
//! live in `tests/parity_wfst.rs`, checked against real OpenFST). Do not
//! conflate the two: loosening an axiom test would break correctness, not a
//! tolerance.

use std::fmt::Debug;

/// A weight semiring `(K, ⊕, ⊗, 0̄, 1̄)` for WFST computation.
///
/// Implementors are the weight type `K` itself (a small `Copy` value). The
/// decoder is generic over this trait so a future `log` semiring is an additive
/// (ADR M5-06 §2).
pub trait Semiring: Copy + Clone + Debug + PartialEq {
    /// The additive identity `0̄` — "no path". `x ⊕ 0̄ = x` and `x ⊗ 0̄ = 0̄`.
    fn zero() -> Self;

    /// The multiplicative identity `1̄` — "free". `x ⊗ 1̄ = x`.
    fn one() -> Self;

    /// The semiring sum `x ⊕ y` (tropical: `min`).
    #[must_use]
    fn plus(self, other: Self) -> Self;

    /// The semiring product `x ⊗ y` (tropical: `+`).
    #[must_use]
    fn times(self, other: Self) -> Self;

    /// `true` when `self` is (bit-for-bit or by the type's own rule) the
    /// additive identity `0̄`. Used by the decoder to prune "no path" tokens
    /// without a magic sentinel comparison at the call site.
    fn is_zero(self) -> bool {
        self == Self::zero()
    }

    /// Tolerance-aware equality for **parity assertions** (not for control
    /// flow). `atol` is an absolute tolerance on the underlying scalar. This
    /// exists so a parity test can compare a decoded path cost against an
    /// OpenFST reference within a documented, honest tolerance
    /// (NFR-QL-01) — it must never be used to make two genuinely different
    /// paths compare equal inside the decoder.
    fn approx_eq(self, other: Self, atol: f64) -> bool;
}

/// The tropical semiring weight `(ℝ ∪ {+∞}, min, +, +∞, 0)`.
///
/// Wraps a single `f32` **cost** (negative log-probability). Matches OpenFST's
/// `standard` arc weight exactly, so a committed `.fst` fixture's arc weights
/// are the raw bit pattern read back here (`f32::from_le_bytes`). `+∞`
/// (`f32::INFINITY`) is [`Semiring::zero`] = "no path"; `0.0` is
/// [`Semiring::one`] = "free".
#[derive(Debug, Clone, Copy)]
pub struct TropicalWeight(pub f32);

impl TropicalWeight {
    /// The raw cost value.
    #[inline]
    pub fn value(self) -> f32 {
        self.0
    }

    /// Constructs a tropical weight from a cost.
    #[inline]
    pub fn new(cost: f32) -> Self {
        Self(cost)
    }
}

impl PartialEq for TropicalWeight {
    /// Bit-insensitive equality that treats every `NaN` cost as unequal to
    /// everything (including itself) — the IEEE-754 rule. Two `+∞` costs are
    /// equal (both are `zero`), which [`Semiring::is_zero`] relies on.
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Semiring for TropicalWeight {
    #[inline]
    fn zero() -> Self {
        Self(f32::INFINITY)
    }

    #[inline]
    fn one() -> Self {
        Self(0.0)
    }

    #[inline]
    fn plus(self, other: Self) -> Self {
        // Tropical ⊕ = min. `f32::min` propagates the non-NaN operand, so a
        // NaN cost (which should never appear in a valid FST) cannot silently
        // win the min; a NaN reaching here is a fixture / reader bug surfaced
        // elsewhere, not swallowed.
        Self(self.0.min(other.0))
    }

    #[inline]
    fn times(self, other: Self) -> Self {
        // Tropical ⊗ = +. `+∞ + finite = +∞` (annihilation) holds for f32.
        Self(self.0 + other.0)
    }

    #[inline]
    fn is_zero(self) -> bool {
        self.0.is_infinite() && self.0 > 0.0
    }

    #[inline]
    fn approx_eq(self, other: Self, atol: f64) -> bool {
        // Both "no path": equal regardless of tolerance.
        if self.is_zero() && other.is_zero() {
            return true;
        }
        // One infinite, the other finite: never approximately equal.
        if self.0.is_infinite() || other.0.is_infinite() {
            return self.0 == other.0;
        }
        (f64::from(self.0) - f64::from(other.0)).abs() <= atol
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // NOTE: these are *algebra* tests (semiring axioms), NOT numerical-parity
    // tests. See the module docs.

    #[test]
    fn identities_hold() {
        let x = TropicalWeight::new(2.5);
        // Additive identity: x ⊕ 0̄ = x.
        assert_eq!(x.plus(TropicalWeight::zero()), x);
        assert_eq!(TropicalWeight::zero().plus(x), x);
        // Multiplicative identity: x ⊗ 1̄ = x.
        assert_eq!(x.times(TropicalWeight::one()), x);
        assert_eq!(TropicalWeight::one().times(x), x);
    }

    #[test]
    fn annihilation_holds() {
        // x ⊗ 0̄ = 0̄ (a "no path" edge kills the whole product).
        let x = TropicalWeight::new(3.0);
        assert!(x.times(TropicalWeight::zero()).is_zero());
        assert!(TropicalWeight::zero().times(x).is_zero());
    }

    #[test]
    fn plus_is_min_and_times_is_add() {
        let a = TropicalWeight::new(1.0);
        let b = TropicalWeight::new(4.0);
        assert_eq!(a.plus(b), TropicalWeight::new(1.0)); // min
        assert_eq!(a.times(b), TropicalWeight::new(5.0)); // +
    }

    #[test]
    fn plus_is_associative_and_commutative() {
        let a = TropicalWeight::new(1.5);
        let b = TropicalWeight::new(0.25);
        let c = TropicalWeight::new(9.0);
        assert_eq!(a.plus(b).plus(c), a.plus(b.plus(c))); // associative
        assert_eq!(a.plus(b), b.plus(a)); // commutative
    }

    #[test]
    fn times_is_associative() {
        let a = TropicalWeight::new(1.5);
        let b = TropicalWeight::new(0.25);
        let c = TropicalWeight::new(9.0);
        assert_eq!(a.times(b).times(c), a.times(b.times(c)));
    }

    #[test]
    fn times_distributes_over_plus() {
        // a ⊗ (b ⊕ c) = (a ⊗ b) ⊕ (a ⊗ c) — the semiring distributive law.
        let a = TropicalWeight::new(2.0);
        let b = TropicalWeight::new(5.0);
        let c = TropicalWeight::new(1.0);
        let lhs = a.times(b.plus(c));
        let rhs = a.times(b).plus(a.times(c));
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn zero_and_one_are_the_expected_constants() {
        assert!(TropicalWeight::zero().value().is_infinite());
        assert!(TropicalWeight::zero().value() > 0.0);
        assert!(TropicalWeight::zero().is_zero());
        assert_eq!(TropicalWeight::one().value(), 0.0);
        assert!(!TropicalWeight::one().is_zero());
    }

    #[test]
    fn approx_eq_respects_tolerance() {
        let a = TropicalWeight::new(1.175);
        let b = TropicalWeight::new(1.175_01);
        assert!(a.approx_eq(b, 1e-3));
        assert!(!a.approx_eq(TropicalWeight::new(1.2), 1e-3));
        // Two "no path" weights are equal; a finite vs infinite is not.
        assert!(TropicalWeight::zero().approx_eq(TropicalWeight::zero(), 0.0));
        assert!(!TropicalWeight::zero().approx_eq(a, 1e9));
    }
}
