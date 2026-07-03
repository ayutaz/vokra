//! FFT planning: length factorization and strategy selection (M0-04-T03..T05).
//!
//! A [`FftPlan`] precomputes everything a fixed length needs (factorization and
//! twiddle table, or a [`BluesteinPlan`]) so repeated transforms of the same
//! length reuse it — the hot-path-allocation-free structure FR-EX-05 (M1) will
//! build on. The plan exposes only the *unnormalized* forward/inverse DFTs;
//! [`Normalization`](super::Normalization) scaling is applied by the wrappers in
//! [`super`].

use vokra_core::Complex32;

use super::bluestein::BluesteinPlan;
use super::cfft::{FftCtx, factorize, fft_rec, largest_factor};
use super::twiddle::forward_roots;

/// Above this largest-prime-factor, a length is routed through Bluestein rather
/// than the direct mixed-radix combine (which would be `O(n · p)` for prime
/// `p`). Small primes (3, 5, 7, …, 61) stay on the cheaper direct path.
const BLUESTEIN_THRESHOLD: usize = 61;

enum Kind {
    Direct {
        factors: Vec<usize>,
        tw: Vec<Complex32>,
    },
    Bluestein(BluesteinPlan),
}

/// A reusable plan for complex FFTs of one fixed length.
pub struct FftPlan {
    n: usize,
    kind: Kind,
}

impl FftPlan {
    /// Builds a plan for length `n`.
    ///
    /// # Panics
    ///
    /// Panics if `n == 0`.
    pub fn new(n: usize) -> Self {
        assert!(n > 0, "FFT length must be non-zero");
        let kind = if largest_factor(n) > BLUESTEIN_THRESHOLD {
            Kind::Bluestein(BluesteinPlan::new(n))
        } else {
            Kind::Direct {
                factors: factorize(n),
                tw: forward_roots(n),
            }
        };
        Self { n, kind }
    }

    /// The transform length.
    pub fn len(&self) -> usize {
        self.n
    }

    /// Always `false` (an [`FftPlan`] has a non-zero length by construction);
    /// present to satisfy the `len`/`is_empty` convention.
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Unnormalized forward DFT `X[k] = Σ_j x[j] e^{-2πi jk/n}`.
    ///
    /// # Panics
    ///
    /// Panics if `x.len() != self.len()`.
    pub fn forward_raw(&self, x: &[Complex32]) -> Vec<Complex32> {
        self.run(x, false)
    }

    /// Unnormalized inverse DFT `x[j] = Σ_k X[k] e^{+2πi jk/n}` (no `1/n`
    /// factor — apply normalization separately).
    ///
    /// # Panics
    ///
    /// Panics if `x.len() != self.len()`.
    pub fn inverse_raw(&self, x: &[Complex32]) -> Vec<Complex32> {
        self.run(x, true)
    }

    fn run(&self, x: &[Complex32], inverse: bool) -> Vec<Complex32> {
        assert_eq!(x.len(), self.n, "FFT input length mismatch");
        match &self.kind {
            Kind::Direct { factors, tw } => {
                let ctx = FftCtx {
                    input: x,
                    tw,
                    big_n: self.n,
                    inverse,
                };
                let mut out = vec![Complex32::ZERO; self.n];
                fft_rec(&ctx, 0, 1, &mut out, factors);
                out
            }
            Kind::Bluestein(plan) => {
                if inverse {
                    plan.inverse_raw(x)
                } else {
                    plan.forward_raw(x)
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Independent naive `O(n²)` DFT reference used to validate both plan paths.
    fn naive_dft(x: &[Complex32], inverse: bool) -> Vec<Complex32> {
        let n = x.len();
        let sign = if inverse { 1.0 } else { -1.0 };
        (0..n)
            .map(|k| {
                let mut acc = Complex32::ZERO;
                for (j, &xj) in x.iter().enumerate() {
                    let angle = sign * 2.0 * PI * (j as f64) * (k as f64) / (n as f64);
                    let w = Complex32::new(angle.cos() as f32, angle.sin() as f32);
                    acc = acc + xj * w;
                }
                acc
            })
            .collect()
    }

    fn sample(n: usize) -> Vec<Complex32> {
        (0..n)
            .map(|k| {
                let a = (k as f32 * 0.37).sin();
                let b = (k as f32 * 0.11 - 0.4).cos();
                Complex32::new(a, b)
            })
            .collect()
    }

    fn assert_close(a: &[Complex32], b: &[Complex32], atol: f32) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            assert!(
                (x.re - y.re).abs() < atol && (x.im - y.im).abs() < atol,
                "mismatch: {x:?} vs {y:?} (atol {atol})"
            );
        }
    }

    #[test]
    fn forward_matches_naive_over_many_lengths() {
        // powers of two, composites (2/3/5) and small primes (direct path),
        // plus a large-prime length that forces Bluestein.
        for &n in &[2usize, 8, 64, 512, 1024, 12, 60, 400, 1000, 7, 97, 101] {
            let x = sample(n);
            let plan = FftPlan::new(n);
            assert_close(&plan.forward_raw(&x), &naive_dft(&x, false), 1e-2);
            assert_close(&plan.inverse_raw(&x), &naive_dft(&x, true), 1e-2);
        }
    }

    #[test]
    fn bluestein_path_is_selected_for_large_prime() {
        let plan = FftPlan::new(101);
        assert!(matches!(plan.kind, Kind::Bluestein(_)));
        let plan = FftPlan::new(1024);
        assert!(matches!(plan.kind, Kind::Direct { .. }));
    }

    #[test]
    fn roundtrip_backward_normalized() {
        for &n in &[8usize, 60, 97, 128] {
            let x = sample(n);
            let plan = FftPlan::new(n);
            let back = plan.inverse_raw(&plan.forward_raw(&x));
            let scaled: Vec<Complex32> = back.iter().map(|c| c.scale(1.0 / n as f32)).collect();
            assert_close(&scaled, &x, 1e-3);
        }
    }
}
