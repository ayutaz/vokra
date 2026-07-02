//! Analysis / synthesis window functions (M0-04-T09; FR-OP-01).
//!
//! Implements the four windows FR-OP-01 enumerates — Hann, Hamming, 4-term
//! Blackman-Harris and Kaiser. STFT front-ends use the **periodic**
//! ([`WindowSymmetry::Periodic`]) form (matches `torch.*_window(...,
//! periodic=True)` / librosa `sym=False`); the **symmetric** form matches
//! `numpy.hanning` / `numpy.hamming` / `numpy.kaiser` and is the parity
//! reference for those three (numpy has no 4-term Blackman-Harris).
//!
//! Coefficients are evaluated in `f64` and rounded once to `f32`.

use std::f64::consts::PI;

use vokra_core::ir::graph::{Window, WindowSymmetry};

/// Samples the `kind` window of `length` points under `symmetry`.
///
/// Returns an empty vector for `length == 0` and `[1.0]` for `length == 1`.
pub fn window(kind: Window, length: usize, symmetry: WindowSymmetry) -> Vec<f32> {
    if length == 0 {
        return Vec::new();
    }
    if length == 1 {
        return vec![1.0];
    }
    let denom = match symmetry {
        WindowSymmetry::Periodic => length as f64,
        WindowSymmetry::Symmetric => (length - 1) as f64,
    };
    match kind {
        Window::Hann => cosine_sum(length, denom, &[0.5, 0.5]),
        Window::Hamming => cosine_sum(length, denom, &[0.54, 0.46]),
        Window::BlackmanHarris => cosine_sum(length, denom, &[0.35875, 0.48829, 0.14128, 0.01168]),
        Window::Kaiser { beta } => kaiser(length, denom, beta as f64),
    }
}

/// Generalized cosine-sum window `w[n] = Σ_i (−1)^i · a_i · cos(i · 2πn/denom)`.
///
/// `coeffs = [a_0, a_1, …]`; Hann is `[0.5, 0.5]`, Hamming `[0.54, 0.46]`,
/// Blackman-Harris the four-term set.
fn cosine_sum(length: usize, denom: f64, coeffs: &[f64]) -> Vec<f32> {
    (0..length)
        .map(|n| {
            let x = 2.0 * PI * (n as f64) / denom;
            let mut acc = 0.0;
            let mut sign = 1.0;
            for (i, &a) in coeffs.iter().enumerate() {
                acc += sign * a * ((i as f64) * x).cos();
                sign = -sign;
            }
            acc as f32
        })
        .collect()
}

/// Kaiser window `w[n] = I0(β·√(1 − r²)) / I0(β)` with `r = (n − denom/2) /
/// (denom/2)`.
fn kaiser(length: usize, denom: f64, beta: f64) -> Vec<f32> {
    let i0_beta = bessel_i0(beta);
    let half = denom / 2.0;
    (0..length)
        .map(|n| {
            let r = (n as f64 - half) / half;
            let arg = beta * (1.0 - r * r).max(0.0).sqrt();
            (bessel_i0(arg) / i0_beta) as f32
        })
        .collect()
}

/// Modified Bessel function of the first kind, order 0, by its power series
/// `I0(x) = Σ_k ((x/2)^{2k} / (k!)²)`.
fn bessel_i0(x: f64) -> f64 {
    let half_sq = (x * 0.5) * (x * 0.5);
    let mut term = 1.0;
    let mut sum = 1.0;
    let mut k = 1.0;
    // Ratio of successive terms is (x/2)² / k²; stop once it is negligible.
    while k < 1000.0 {
        term *= half_sq / (k * k);
        sum += term;
        if term < sum * 1e-13 {
            break;
        }
        k += 1.0;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn periodic_hann_known_values() {
        // Periodic Hann of length 4: 0.5 - 0.5 cos(2πn/4) = [0, 0.5, 1, 0.5].
        let w = window(Window::Hann, 4, WindowSymmetry::Periodic);
        let expect = [0.0f32, 0.5, 1.0, 0.5];
        for (a, b) in w.iter().zip(expect) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn symmetric_hann_endpoints_are_zero() {
        let w = window(Window::Hann, 8, WindowSymmetry::Symmetric);
        assert!(w[0].abs() < 1e-6 && w[7].abs() < 1e-6);
        // Symmetric window is a palindrome.
        for i in 0..8 {
            assert!((w[i] - w[7 - i]).abs() < 1e-6);
        }
    }

    #[test]
    fn hamming_endpoints_match_alpha_minus_beta() {
        // Symmetric Hamming endpoints equal 0.54 - 0.46 = 0.08.
        let w = window(Window::Hamming, 16, WindowSymmetry::Symmetric);
        assert!((w[0] - 0.08).abs() < 1e-5);
        assert!((w[15] - 0.08).abs() < 1e-5);
    }

    #[test]
    fn blackman_harris_peaks_near_center_at_unity_sum() {
        // Sum of BH coefficients at the peak (cos terms → ±1 alternating with
        // sign) equals a0+a1+a2+a3 = 1.0 at the center of a symmetric window.
        let w = window(Window::BlackmanHarris, 65, WindowSymmetry::Symmetric);
        let peak = w[32];
        assert!((peak - 1.0).abs() < 1e-4, "peak {peak}");
        assert!(w[0] < 0.001 && w[64] < 0.001);
    }

    #[test]
    fn bessel_i0_reference_points() {
        // I0(0)=1; I0(1)≈1.2660658; I0(2)≈2.2795853 (standard tables).
        assert!((bessel_i0(0.0) - 1.0).abs() < 1e-12);
        assert!((bessel_i0(1.0) - 1.266_065_877_75).abs() < 1e-9);
        assert!((bessel_i0(2.0) - 2.279_585_302_34).abs() < 1e-9);
    }

    #[test]
    fn kaiser_is_symmetric_and_peaks_at_one() {
        let w = window(Window::Kaiser { beta: 8.0 }, 33, WindowSymmetry::Symmetric);
        assert!((w[16] - 1.0).abs() < 1e-6);
        for i in 0..33 {
            assert!((w[i] - w[32 - i]).abs() < 1e-6);
        }
    }
}
