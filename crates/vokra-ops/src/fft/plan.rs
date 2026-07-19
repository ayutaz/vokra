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

    /// Whether this plan runs the Direct mixed-radix path. Only Direct plans
    /// support the allocation-free [`forward_raw_into`](Self::forward_raw_into)
    /// / [`inverse_raw_into`](Self::inverse_raw_into) variants — the Bluestein
    /// strategy (a largest prime factor above the threshold) needs internal
    /// buffers. Hot-loop callers (FR-EX-05, M4-03) gate on this at setup time
    /// and reject the length explicitly rather than falling back to an
    /// allocating path (FR-EX-08 spirit).
    pub fn is_direct(&self) -> bool {
        matches!(self.kind, Kind::Direct { .. })
    }

    /// Length of the caller-provided scratch buffer the `_into` variants
    /// require (= the transform length).
    pub fn scratch_len(&self) -> usize {
        self.n
    }

    /// Allocation-free [`forward_raw`](Self::forward_raw): writes the
    /// unnormalized forward DFT of `x` into `out`, using `scratch` for the
    /// combine levels. Bit-identical to `forward_raw` (same code path);
    /// `scratch` may hold arbitrary values (every element is written before
    /// any read).
    ///
    /// # Panics
    ///
    /// Panics on a Bluestein plan (gate with [`is_direct`](Self::is_direct)),
    /// or if `x.len() != self.len()`, `out.len() != self.len()`, or
    /// `scratch.len() < self.scratch_len()`.
    pub fn forward_raw_into(
        &self,
        x: &[Complex32],
        out: &mut [Complex32],
        scratch: &mut [Complex32],
    ) {
        self.run_into(x, false, out, scratch);
    }

    /// Allocation-free [`inverse_raw`](Self::inverse_raw) (unnormalized; no
    /// `1/n`). Same contract as [`forward_raw_into`](Self::forward_raw_into).
    ///
    /// # Panics
    ///
    /// Panics on a Bluestein plan, or on any length mismatch (see
    /// [`forward_raw_into`](Self::forward_raw_into)).
    pub fn inverse_raw_into(
        &self,
        x: &[Complex32],
        out: &mut [Complex32],
        scratch: &mut [Complex32],
    ) {
        self.run_into(x, true, out, scratch);
    }

    fn run_into(
        &self,
        x: &[Complex32],
        inverse: bool,
        out: &mut [Complex32],
        scratch: &mut [Complex32],
    ) {
        assert_eq!(x.len(), self.n, "FFT input length mismatch");
        assert_eq!(out.len(), self.n, "FFT output length mismatch");
        assert!(
            scratch.len() >= self.n,
            "FFT scratch shorter than scratch_len()"
        );
        match &self.kind {
            Kind::Direct { factors, tw } => {
                let ctx = FftCtx {
                    input: x,
                    tw,
                    big_n: self.n,
                    inverse,
                };
                fft_rec(&ctx, 0, 1, out, factors, scratch);
            }
            Kind::Bluestein(_) => panic!(
                "FftPlan::{}_raw_into is alloc-free and unsupported on Bluestein plans \
                 (length {} has a large prime factor); gate with is_direct()",
                if inverse { "inverse" } else { "forward" },
                self.n
            ),
        }
    }

    fn run(&self, x: &[Complex32], inverse: bool) -> Vec<Complex32> {
        assert_eq!(x.len(), self.n, "FFT input length mismatch");
        match &self.kind {
            Kind::Direct { .. } => {
                let mut out = vec![Complex32::ZERO; self.n];
                let mut scratch = vec![Complex32::ZERO; self.n];
                self.run_into(x, inverse, &mut out, &mut scratch);
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
    fn is_direct_mirrors_the_strategy() {
        assert!(FftPlan::new(512).is_direct());
        assert!(FftPlan::new(60).is_direct());
        assert!(!FftPlan::new(101).is_direct(), "large prime → Bluestein");
    }

    /// The alloc-free `_into` variants are bit-identical to the allocating
    /// API (same code path, same combine order), and the caller scratch may
    /// hold arbitrary garbage — combine writes every scratch element before
    /// any read (M4-03-T11 pre-allocation seam).
    #[test]
    fn into_variants_match_allocating_api_bit_exactly() {
        for &n in &[2usize, 8, 64, 512, 12, 60, 400] {
            let x = sample(n);
            let plan = FftPlan::new(n);
            let mut out = vec![Complex32::ZERO; n];
            // Deliberate garbage so a read-before-write in the scratch shows.
            let mut scratch = vec![Complex32::new(f32::NAN, -7.5); plan.scratch_len()];

            plan.forward_raw_into(&x, &mut out, &mut scratch);
            let want = plan.forward_raw(&x);
            for (a, b) in out.iter().zip(&want) {
                assert_eq!(a.re.to_bits(), b.re.to_bits(), "forward n={n}");
                assert_eq!(a.im.to_bits(), b.im.to_bits(), "forward n={n}");
            }

            plan.inverse_raw_into(&x, &mut out, &mut scratch);
            let want = plan.inverse_raw(&x);
            for (a, b) in out.iter().zip(&want) {
                assert_eq!(a.re.to_bits(), b.re.to_bits(), "inverse n={n}");
                assert_eq!(a.im.to_bits(), b.im.to_bits(), "inverse n={n}");
            }
        }
    }

    #[test]
    #[should_panic(expected = "Bluestein")]
    fn into_variant_panics_on_bluestein_plans() {
        // The `_into` contract is alloc-free; the Bluestein strategy needs
        // internal buffers, so callers must gate on `is_direct()`.
        let plan = FftPlan::new(101);
        let x = sample(101);
        let mut out = vec![Complex32::ZERO; 101];
        let mut scratch = vec![Complex32::ZERO; plan.scratch_len()];
        plan.forward_raw_into(&x, &mut out, &mut scratch);
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
