//! Precomputed twiddle-factor (root-of-unity) tables (M0-04-T03).
//!
//! Ported in spirit from pocketfft's `cfftp_comp_twiddle`
//! (M. Reinecke, Max-Planck-Society, BSD-3-Clause — see
//! `THIRD_PARTY_LICENSES/pocketfft-LICENSE.txt`). pocketfft stores per-pass
//! twiddle rows; Vokra instead keeps a single length-`n` root table for the
//! *top* transform length and indexes into it with a stride at every recursion
//! level (all sub-lengths divide `n`), which is simpler and equally reusable.
//!
//! Angles are evaluated in `f64` and rounded to `f32` so the stored roots carry
//! full `f64` accuracy of the sine/cosine before the single rounding step.

use crate::complex::Complex32;

use std::f64::consts::PI;

/// Builds the forward root table `w[t] = e^{-2πi·t / n}` for `t in 0..n`.
///
/// The inverse transform reuses the same table via [`Complex32::conj`] rather
/// than allocating a second table.
pub(crate) fn forward_roots(n: usize) -> Vec<Complex32> {
    (0..n)
        .map(|t| {
            let angle = -2.0 * PI * (t as f64) / (n as f64);
            Complex32::new(angle.cos() as f32, angle.sin() as f32)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roots_are_on_the_unit_circle_at_known_points() {
        let w = forward_roots(4);
        // e^{0} = 1, e^{-iπ/2} = -i, e^{-iπ} = -1, e^{-i3π/2} = +i.
        assert!((w[0].re - 1.0).abs() < 1e-6 && w[0].im.abs() < 1e-6);
        assert!(w[1].re.abs() < 1e-6 && (w[1].im + 1.0).abs() < 1e-6);
        assert!((w[2].re + 1.0).abs() < 1e-6 && w[2].im.abs() < 1e-6);
        assert!(w[3].re.abs() < 1e-6 && (w[3].im - 1.0).abs() < 1e-6);
    }

    #[test]
    fn single_point_table_is_unity() {
        assert_eq!(forward_roots(1), vec![Complex32::new(1.0, 0.0)]);
    }
}
