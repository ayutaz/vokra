//! FFT lowering for the CPU backend (FR-OP-05, M0-04-T03..T06).
//!
//! A from-scratch Rust reimplementation of the pocketfft algorithm family
//! (M. Reinecke, Max-Planck-Society, 3-clause BSD — the license text is
//! bundled at `crates/vokra-ops/THIRD_PARTY_LICENSES/pocketfft-LICENSE.txt`,
//! and the port is recorded in the crate NOTICE follow-up). External FFT crates
//! (`rustfft`, `realfft`, FFTW3) are intentionally **not** used: FFTW3 is GPL
//! (NFR-LC-02/03) and the workspace keeps zero third-party dependencies.
//!
//! Layout:
//!
//! - [`Complex32`] — the complex value type, re-exported from `vokra_core`
//!   (moved to the core crate in M1-04, FR-EX-09);
//! - [`FftPlan`] — reusable complex-to-complex plan (mixed-radix + Bluestein);
//! - [`RealFftPlan`] — real-input `r2c` / `c2r` plan;
//! - [`norm_scale`] — maps a [`Normalization`] onto a scalar factor.
//!
//! Only the CPU path is in scope for M0-04; the GPU FFT lowering targets of
//! FR-OP-05 (cuFFT / VkFFT / MPS FFT) are later milestones.

mod bluestein;
mod cfft;
mod plan;
mod real;
mod twiddle;

pub use plan::FftPlan;
pub use real::RealFftPlan;
pub use vokra_core::Complex32;

use vokra_core::ir::graph::Normalization;

/// The scalar applied to a length-`n` transform under `norm`.
///
/// `forward` selects the direction. With [`Normalization::Backward`] (the FFT
/// engineering default) the forward transform is unscaled and the inverse
/// carries `1/n`; [`Normalization::Forward`] swaps that; [`Normalization::Ortho`]
/// applies `1/√n` to both directions (a unitary transform).
///
/// [`FftPlan`] and [`RealFftPlan`] compute the *unnormalized* transforms
/// (`forward_raw` / the `1/n`-including real `inverse`); higher-level ops
/// multiply by this factor so a chosen convention is applied exactly once.
pub fn norm_scale(norm: Normalization, n: usize, forward: bool) -> f32 {
    let n = n as f32;
    match norm {
        Normalization::Forward => {
            if forward {
                1.0 / n
            } else {
                1.0
            }
        }
        Normalization::Backward => {
            if forward {
                1.0
            } else {
                1.0 / n
            }
        }
        Normalization::Ortho => 1.0 / n.sqrt(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn norm_scale_conventions() {
        // Backward: forward unscaled, inverse 1/n.
        assert_eq!(norm_scale(Normalization::Backward, 8, true), 1.0);
        assert_eq!(norm_scale(Normalization::Backward, 8, false), 1.0 / 8.0);
        // Forward: mirror image.
        assert_eq!(norm_scale(Normalization::Forward, 8, true), 1.0 / 8.0);
        assert_eq!(norm_scale(Normalization::Forward, 8, false), 1.0);
        // Ortho: 1/sqrt(n) both directions.
        let s = 1.0 / (8.0f32).sqrt();
        assert_eq!(norm_scale(Normalization::Ortho, 8, true), s);
        assert_eq!(norm_scale(Normalization::Ortho, 8, false), s);
    }

    #[test]
    fn ortho_roundtrip_is_unit_scaled() {
        // Applying forward then inverse ortho scales by 1/n overall.
        let n = 16;
        let f = norm_scale(Normalization::Ortho, n, true);
        let i = norm_scale(Normalization::Ortho, n, false);
        assert!((f * i - 1.0 / n as f32).abs() < 1e-6);
    }
}
