//! Single-precision complex number `re + i·im` (a pair of `f32`).
//!
//! This is the host value type behind the [`DType::Complex64`](crate::DType) IR
//! dtype — numpy / torch name `complex64`, i.e. two 32-bit floats, 8 bytes per
//! element (see [`DType::size_in_bytes`](crate::DType::size_in_bytes)).
//!
//! It was promoted out of `vokra-ops` into the core crate in M1-04 so the IR and
//! the audio ops share a single definition (FR-EX-09); `vokra-ops` re-exports it
//! and its from-scratch FFT core builds on it. Storage is `f32` (`re`, `im`) per
//! the original `Complex32` naming; twiddle factors are computed in `f64` and
//! rounded to `f32` inside the FFT, comfortably inside the FP32 parity budget
//! (`atol = 0.01`, NFR-QL-01).
//!
//! The arithmetic here is pure, safe two-`f32` code (`vokra-core` is
//! `unsafe_code = "deny"`). `#[repr(C)]` pins the field order to `[re, im]`, so a
//! contiguous `[Complex32]` has the same layout as an interleaved
//! `[re, im, re, im, …]` `f32` buffer at the C ABI / codec boundary.

use std::ops::{Add, Mul, Sub};

/// A single-precision complex number `re + i·im`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct Complex32 {
    /// Real part.
    pub re: f32,
    /// Imaginary part.
    pub im: f32,
}

impl Complex32 {
    /// The additive identity `0 + 0i`.
    pub const ZERO: Self = Self { re: 0.0, im: 0.0 };

    /// Builds a complex number from its real and imaginary parts.
    #[inline]
    pub const fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }

    /// Builds a real-valued complex number `re + 0i`.
    #[inline]
    pub const fn from_real(re: f32) -> Self {
        Self { re, im: 0.0 }
    }

    /// The complex conjugate `re - i·im`.
    #[inline]
    pub const fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    /// The squared magnitude `re² + im²`.
    #[inline]
    pub fn norm_sqr(self) -> f32 {
        self.re * self.re + self.im * self.im
    }

    /// Scales both components by the real scalar `s`.
    #[inline]
    pub fn scale(self, s: f32) -> Self {
        Self {
            re: self.re * s,
            im: self.im * s,
        }
    }
}

impl Add for Complex32 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl Sub for Complex32 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

impl Mul for Complex32 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arithmetic_matches_hand_values() {
        let a = Complex32::new(1.0, 2.0);
        let b = Complex32::new(-3.0, 4.0);
        assert_eq!(a + b, Complex32::new(-2.0, 6.0));
        assert_eq!(a - b, Complex32::new(4.0, -2.0));
        // (1+2i)(-3+4i) = -3 + 4i - 6i + 8i² = -11 - 2i.
        assert_eq!(a * b, Complex32::new(-11.0, -2.0));
    }

    #[test]
    fn conj_scale_and_norm() {
        let a = Complex32::new(3.0, -4.0);
        assert_eq!(a.conj(), Complex32::new(3.0, 4.0));
        assert_eq!(a.scale(2.0), Complex32::new(6.0, -8.0));
        assert_eq!(a.norm_sqr(), 25.0);
        // z · conj(z) = |z|² (real).
        let p = a * a.conj();
        assert_eq!(p.re, 25.0);
        assert_eq!(p.im, 0.0);
    }

    #[test]
    fn repr_c_layout_is_two_contiguous_f32() {
        // `#[repr(C)]` pins size/align to a bare `[re, im]` f32 pair, which is
        // what lets a `[Complex32]` alias an interleaved f32 buffer.
        assert_eq!(core::mem::size_of::<Complex32>(), 8);
        assert_eq!(core::mem::align_of::<Complex32>(), 4);
    }
}
