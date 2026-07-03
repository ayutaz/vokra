//! Real-input FFT: `r2c` / `c2r` for the `real_input` (RFFT) path (M0-04-T06).
//!
//! `real_input` in `StftAttrs` exists because a real
//! signal has a Hermitian-symmetric spectrum, so only `n/2 + 1` bins are
//! independent (FR-OP-01, "real_input (RFFT で 2倍高速)"). The forward path
//! ([`RealFftPlan::forward`]) realizes that ~2× speedup by transforming the
//! `n`-sample real signal with a single length-`n/2` complex FFT and unpacking
//! the two interleaved half-spectra — the standard real-FFT-via-half-length
//! technique (pocketfft ships dedicated `rfftp` real passes; Vokra uses the
//! packing method instead, which reuses the complex core unchanged).
//!
//! The inverse ([`RealFftPlan::inverse`]) reconstructs the full Hermitian
//! spectrum and runs one length-`n` complex inverse. Giving the inverse its own
//! half-length packing is a performance follow-up; correctness is covered by the
//! `irfft(rfft(x)) == x` round-trip test and the iSTFT reconstruction tests.

use vokra_core::Complex32;

use super::plan::FftPlan;

/// A reusable real-input FFT plan for one fixed even/odd length `n`.
pub struct RealFftPlan {
    n: usize,
    /// Length-`n/2` complex plan used by the packed forward path (even `n`).
    half: Option<FftPlan>,
    /// Length-`n` complex plan (forward path for odd `n`; inverse path always).
    full: FftPlan,
    /// Recombination twiddles `e^{-2πi k / n}` for `k in 0..n/2` (even path).
    recomb: Vec<Complex32>,
}

impl RealFftPlan {
    /// Builds a real FFT plan for length `n` (`n ≥ 1`).
    ///
    /// # Panics
    ///
    /// Panics if `n == 0`.
    pub fn new(n: usize) -> Self {
        assert!(n > 0, "real FFT length must be non-zero");
        let even = n % 2 == 0;
        let half = if even && n >= 2 {
            Some(FftPlan::new(n / 2))
        } else {
            None
        };
        let recomb = if even {
            (0..n / 2)
                .map(|k| {
                    let angle = -2.0 * std::f64::consts::PI * (k as f64) / (n as f64);
                    Complex32::new(angle.cos() as f32, angle.sin() as f32)
                })
                .collect()
        } else {
            Vec::new()
        };
        Self {
            n,
            half,
            full: FftPlan::new(n),
            recomb,
        }
    }

    /// The signal length `n`.
    pub fn len(&self) -> usize {
        self.n
    }

    /// Always `false` (length is non-zero by construction).
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Number of independent spectrum bins, `n/2 + 1`.
    pub fn num_bins(&self) -> usize {
        self.n / 2 + 1
    }

    /// Forward real→complex transform: the unnormalized `n/2 + 1`-bin spectrum
    /// of the length-`n` real `signal`.
    ///
    /// The result equals the first `n/2 + 1` bins of the full complex
    /// [`FftPlan::forward_raw`] of `signal` (validated to `atol = 0.01`).
    ///
    /// # Panics
    ///
    /// Panics if `signal.len() != self.len()`.
    pub fn forward(&self, signal: &[f32]) -> Vec<Complex32> {
        assert_eq!(signal.len(), self.n, "real FFT input length mismatch");
        match &self.half {
            Some(half_plan) => self.forward_packed(signal, half_plan),
            None => self.forward_via_full(signal),
        }
    }

    /// Packed even-length path: one length-`n/2` complex FFT + unpack.
    fn forward_packed(&self, signal: &[f32], half_plan: &FftPlan) -> Vec<Complex32> {
        let half = self.n / 2;
        // z[j] = signal[2j] + i·signal[2j+1].
        let z: Vec<Complex32> = (0..half)
            .map(|j| Complex32::new(signal[2 * j], signal[2 * j + 1]))
            .collect();
        let spec = half_plan.forward_raw(&z);

        let mut out = vec![Complex32::ZERO; half + 1];
        // DC and Nyquist are purely real, recovered from spec[0].
        out[0] = Complex32::from_real(spec[0].re + spec[0].im);
        out[half] = Complex32::from_real(spec[0].re - spec[0].im);
        for k in 1..half {
            let zk = spec[k];
            let zmk = spec[half - k];
            // Even-index DFT E[k] and odd-index DFT O[k] of the real halves.
            let e = (zk + zmk.conj()).scale(0.5);
            let diff = zk - zmk.conj();
            // O[k] = (-i/2)·diff.
            let o = Complex32::new(0.5 * diff.im, -0.5 * diff.re);
            out[k] = e + self.recomb[k] * o;
        }
        out
    }

    /// Odd-length fallback: full complex FFT sliced to `n/2 + 1` bins.
    fn forward_via_full(&self, signal: &[f32]) -> Vec<Complex32> {
        let input: Vec<Complex32> = signal.iter().map(|&s| Complex32::from_real(s)).collect();
        let mut spec = self.full.forward_raw(&input);
        spec.truncate(self.num_bins());
        spec
    }

    /// Inverse complex→real transform: the length-`n` real signal whose forward
    /// transform is the `n/2 + 1`-bin Hermitian `spectrum`. Includes the `1/n`
    /// factor (a true inverse), so `inverse(forward(x)) == x`.
    ///
    /// # Panics
    ///
    /// Panics if `spectrum.len() != self.num_bins()`.
    pub fn inverse(&self, spectrum: &[Complex32]) -> Vec<f32> {
        assert_eq!(
            spectrum.len(),
            self.num_bins(),
            "real inverse FFT spectrum length mismatch"
        );
        let n = self.n;
        let mut full = vec![Complex32::ZERO; n];
        full[..spectrum.len()].copy_from_slice(spectrum);
        // Rebuild the conjugate-symmetric upper half.
        for k in 1..n.div_ceil(2) {
            full[n - k] = full[k].conj();
        }
        let raw = self.full.inverse_raw(&full);
        let inv_n = 1.0 / n as f32;
        raw.iter().map(|c| c.re * inv_n).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn signal(n: usize) -> Vec<f32> {
        (0..n)
            .map(|k| (k as f32 * 0.3).sin() + 0.5 * (k as f32 * 0.07 - 0.2).cos())
            .collect()
    }

    fn full_reference(signal: &[f32]) -> Vec<Complex32> {
        let input: Vec<Complex32> = signal.iter().map(|&s| Complex32::from_real(s)).collect();
        FftPlan::new(signal.len()).forward_raw(&input)
    }

    #[test]
    fn rfft_matches_first_half_of_complex_fft() {
        for &n in &[2usize, 8, 16, 64, 400, 512] {
            let x = signal(n);
            let r = RealFftPlan::new(n).forward(&x);
            let full = full_reference(&x);
            assert_eq!(r.len(), n / 2 + 1);
            for (k, bin) in r.iter().enumerate() {
                assert!(
                    (bin.re - full[k].re).abs() < 1e-2 && (bin.im - full[k].im).abs() < 1e-2,
                    "n={n} bin {k}: {bin:?} vs {:?}",
                    full[k]
                );
            }
        }
    }

    #[test]
    fn rfft_handles_odd_length() {
        let n = 7;
        let x = signal(n);
        let r = RealFftPlan::new(n).forward(&x);
        let full = full_reference(&x);
        assert_eq!(r.len(), n / 2 + 1);
        for (k, bin) in r.iter().enumerate() {
            assert!((bin.re - full[k].re).abs() < 1e-2 && (bin.im - full[k].im).abs() < 1e-2);
        }
    }

    #[test]
    fn irfft_roundtrip_recovers_signal() {
        for &n in &[2usize, 8, 16, 64, 400, 7, 15] {
            let x = signal(n);
            let plan = RealFftPlan::new(n);
            let back = plan.inverse(&plan.forward(&x));
            assert_eq!(back.len(), n);
            for (a, b) in x.iter().zip(&back) {
                assert!((a - b).abs() < 1e-3, "n={n}: {a} vs {b}");
            }
        }
    }
}
