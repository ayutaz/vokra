//! Bluestein (chirp-z) transform for arbitrary / large-prime lengths
//! (M0-04-T05).
//!
//! Mirrors the role of pocketfft's `fftblue` (M. Reinecke,
//! Max-Planck-Society, BSD-3-Clause — see
//! `THIRD_PARTY_LICENSES/pocketfft-LICENSE.txt`): a length-`n` DFT is rewritten
//! as a length-`m` circular convolution, where `m` is the next power of two
//! `≥ 2n − 1`, and the convolution is evaluated with the power-of-two
//! [`cfft`](super::cfft) engine. This keeps the transform `O(n log n)` even
//! when `n` has a large prime factor, for which the direct mixed-radix combine
//! would be `O(n · p)`.
//!
//! The chirp identity used is `jk = (j² + k² − (k − j)²) / 2`, giving
//! `X[k] = w[k] · Σ_j (x[j]·w[j]) · conj(w[k−j])` with `w[t] = e^{-iπ t² / n}`.

use std::f64::consts::PI;

use crate::complex::Complex32;

use super::cfft::{FftCtx, factorize, fft_rec};

/// Precomputed plan for the Bluestein transform of a fixed length `n`.
pub(crate) struct BluesteinPlan {
    n: usize,
    m: usize,
    /// Forward roots of the inner power-of-two length `m`.
    roots_m: Vec<Complex32>,
    /// Factorization of `m` (all `4`/`2`).
    factors_m: Vec<usize>,
    /// Chirp `w[k] = e^{-iπ k² / n}`, `k in 0..n`.
    chirp: Vec<Complex32>,
    /// Forward transform of the symmetric filter `b`, length `m`.
    filter_fft: Vec<Complex32>,
}

impl BluesteinPlan {
    /// Builds the plan for length `n` (`n ≥ 1`).
    pub(crate) fn new(n: usize) -> Self {
        let m = next_pow2(2 * n - 1).max(1);
        let roots_m = super::twiddle::forward_roots(m);
        let factors_m = factorize(m);

        // Chirp w[k] = e^{-iπ k²/n}; reduce k² modulo 2n before scaling so the
        // sin/cos argument stays small and accurate for large k.
        let chirp: Vec<Complex32> = (0..n)
            .map(|k| {
                let km = (k as u64 * k as u64) % (2 * n as u64);
                let angle = -PI * (km as f64) / (n as f64);
                Complex32::new(angle.cos() as f32, angle.sin() as f32)
            })
            .collect();

        // Filter b: b[0] = conj(w[0]); b[k] = b[m-k] = conj(w[k]) for k in 1..n.
        let mut filter = vec![Complex32::ZERO; m];
        filter[0] = chirp[0].conj();
        for k in 1..n {
            let v = chirp[k].conj();
            filter[k] = v;
            filter[m - k] = v;
        }
        let filter_fft = raw_dft(&filter, &roots_m, &factors_m, m, false);

        Self {
            n,
            m,
            roots_m,
            factors_m,
            chirp,
            filter_fft,
        }
    }

    /// Unnormalized forward DFT `X[k] = Σ_j x[j] e^{-2πi jk/n}`.
    pub(crate) fn forward_raw(&self, x: &[Complex32]) -> Vec<Complex32> {
        self.transform(x, false)
    }

    /// Unnormalized inverse DFT `x[j] = Σ_k X[k] e^{+2πi jk/n}` (no `1/n`).
    pub(crate) fn inverse_raw(&self, x: &[Complex32]) -> Vec<Complex32> {
        // inverse_raw(x) = conj(forward_raw(conj(x))).
        self.transform(x, true)
    }

    fn transform(&self, x: &[Complex32], inverse: bool) -> Vec<Complex32> {
        debug_assert_eq!(x.len(), self.n);
        // a[k] = x[k] · w[k], zero-padded to m (conjugating x when inverting).
        let mut a = vec![Complex32::ZERO; self.m];
        for (k, slot) in a.iter_mut().take(self.n).enumerate() {
            let xk = if inverse { x[k].conj() } else { x[k] };
            *slot = xk * self.chirp[k];
        }

        let spec = raw_dft(&a, &self.roots_m, &self.factors_m, self.m, false);
        let prod: Vec<Complex32> = spec
            .iter()
            .zip(&self.filter_fft)
            .map(|(s, f)| *s * *f)
            .collect();
        // Circular convolution: true inverse (includes 1/m).
        let conv = raw_dft(&prod, &self.roots_m, &self.factors_m, self.m, true);
        let inv_m = 1.0 / self.m as f32;

        (0..self.n)
            .map(|k| {
                let c = conv[k].scale(inv_m) * self.chirp[k];
                if inverse { c.conj() } else { c }
            })
            .collect()
    }
}

/// Runs the power-of-two mixed-radix DFT of `input` (length `m`).
fn raw_dft(
    input: &[Complex32],
    roots_m: &[Complex32],
    factors_m: &[usize],
    m: usize,
    inverse: bool,
) -> Vec<Complex32> {
    let ctx = FftCtx {
        input,
        tw: roots_m,
        big_n: m,
        inverse,
    };
    let mut out = vec![Complex32::ZERO; m];
    fft_rec(&ctx, 0, 1, &mut out, factors_m);
    out
}

/// Smallest power of two `≥ v` (`1` for `v == 0`).
fn next_pow2(v: usize) -> usize {
    let mut p = 1usize;
    while p < v {
        p <<= 1;
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn next_pow2_rounds_up() {
        assert_eq!(next_pow2(1), 1);
        assert_eq!(next_pow2(201), 256);
        assert_eq!(next_pow2(256), 256);
        assert_eq!(next_pow2(257), 512);
    }

    #[test]
    fn dc_input_transforms_to_bin0_sum() {
        // Prime length exercised through Bluestein.
        let n = 7;
        let plan = BluesteinPlan::new(n);
        let x: Vec<Complex32> = (0..n).map(|_| Complex32::new(1.0, 0.0)).collect();
        let out = plan.forward_raw(&x);
        assert!((out[0].re - n as f32).abs() < 1e-3 && out[0].im.abs() < 1e-3);
        for bin in &out[1..] {
            assert!(bin.re.abs() < 1e-3 && bin.im.abs() < 1e-3);
        }
    }

    #[test]
    fn roundtrip_recovers_input() {
        let n = 11;
        let plan = BluesteinPlan::new(n);
        let x: Vec<Complex32> = (0..n)
            .map(|k| Complex32::new(k as f32 * 0.3 - 1.0, 0.5 - k as f32 * 0.2))
            .collect();
        let fwd = plan.forward_raw(&x);
        let back = plan.inverse_raw(&fwd);
        for (a, b) in x.iter().zip(&back) {
            assert!((a.re - b.re / n as f32).abs() < 1e-3);
            assert!((a.im - b.im / n as f32).abs() < 1e-3);
        }
    }
}
