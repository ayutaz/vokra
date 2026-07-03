//! Small, dependency-free pseudo-random generators shared across the workspace.
//!
//! The zero-external-dependency red line (NFR-DS-02) forbids the `rand` crate,
//! so the native models each grew a tiny hand-rolled PRNG. This module hosts the
//! canonical copies so those ad-hoc duplicates collapse onto one implementation
//! **without changing any model's numeric stream**:
//!
//! - [`SplitMix64`] — the splitmix64 generator (Steele, Lea & Flood 2014)
//!   behind piper-plus's synthesis noise, with a [`GaussianSplitMix64`]
//!   Box–Muller wrapper for Gaussian draws;
//! - [`Xorshift64Star`] — the xorshift64\* generator (Vigna 2016) the model
//!   test suites use to pick pseudo-random chunk boundaries / spline params.
//!
//! Both are trivially reproducible (fixed seed ⇒ fixed sequence) and are **not**
//! cryptographically secure; they exist purely for reproducible test / synthesis
//! noise, never for anything security-sensitive.

/// The splitmix64 generator (Steele, Lea & Flood, *Fast Splittable Pseudorandom
/// Number Generators*, 2014).
///
/// A single 64-bit state advanced by a fixed odd increment (the golden-ratio
/// constant) and finalized with the standard bit-mixing avalanche. Seeding is
/// the identity — every `u64` seed is valid, including zero — so it doubles as
/// the canonical way to expand one seed into a stream.
#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Creates a generator seeded with `seed` (any value, including `0`).
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Returns the next 64-bit output and advances the state.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Returns a uniform `f32` in the **open** interval `(0, 1)` — the top 24
    /// bits of one output, shifted half a bin off zero so `ln(x)` is always
    /// finite (used by [`GaussianSplitMix64`]'s Box–Muller transform).
    pub fn next_unit_f32(&mut self) -> f32 {
        // Top 24 bits → [0, 2^24); + 0.5 shifts off the closed endpoints.
        let bits = (self.next_u64() >> 40) as f32;
        (bits + 0.5) / (1u64 << 24) as f32
    }
}

/// A [`SplitMix64`] with a Box–Muller Gaussian transform, the reproducible
/// noise source for piper-plus's stochastic synthesis path (`noise_scale` /
/// `noise_w`).
///
/// Box–Muller produces two independent standard-normal deviates per pair of
/// uniforms; the second is cached in `spare` and returned on the next call, so
/// the output stream is `cos, sin, cos, sin, …` over successive uniform pairs.
/// Fixed-seed and reproducible, but deliberately **not** bit-matched to any
/// external runtime's RNG (only the deterministic, zero-noise path is
/// parity-checked).
#[derive(Debug, Clone)]
pub struct GaussianSplitMix64 {
    rng: SplitMix64,
    spare: Option<f32>,
}

impl GaussianSplitMix64 {
    /// Creates a Gaussian source seeded with `seed`.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self {
            rng: SplitMix64::new(seed),
            spare: None,
        }
    }

    /// Returns the next standard-normal (mean 0, variance 1) deviate.
    pub fn next_gaussian(&mut self) -> f32 {
        if let Some(s) = self.spare.take() {
            return s;
        }
        let u1 = self.rng.next_unit_f32();
        let u2 = self.rng.next_unit_f32();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        self.spare = Some(r * theta.sin());
        r * theta.cos()
    }
}

/// The xorshift64\* generator (Vigna, *An experimental exploration of
/// Marsaglia's xorshift generators, scrambled*, 2016).
///
/// A three-shift xorshift core scrambled by one multiply. The seed is forced
/// odd (`seed | 1`) so the state can never be the all-zero fixed point. Used by
/// the model test suites to draw reproducible pseudo-random chunk boundaries and
/// spline parameters without an external `rand` dependency.
#[derive(Debug, Clone)]
pub struct Xorshift64Star {
    state: u64,
}

impl Xorshift64Star {
    /// Creates a generator seeded with `seed | 1` (forced non-zero).
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self { state: seed | 1 }
    }

    /// Returns the next 64-bit output and advances the state.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Returns a uniform `f32` in `[-1, 1)` — the top 24 bits of one output
    /// mapped onto the symmetric interval.
    pub fn next_signed_f32(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32;
        (bits as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix64_is_deterministic_and_seed_sensitive() {
        let mut a = SplitMix64::new(0xDEAD_BEEF);
        let mut b = SplitMix64::new(0xDEAD_BEEF);
        let seq_a: Vec<u64> = (0..8).map(|_| a.next_u64()).collect();
        let seq_b: Vec<u64> = (0..8).map(|_| b.next_u64()).collect();
        assert_eq!(seq_a, seq_b, "same seed ⇒ same stream");

        let mut c = SplitMix64::new(0xDEAD_BEF0);
        let seq_c: Vec<u64> = (0..8).map(|_| c.next_u64()).collect();
        assert_ne!(seq_a, seq_c, "different seed ⇒ different stream");
    }

    #[test]
    fn splitmix64_first_output_matches_reference_constant() {
        // splitmix64(0) advances the state to the golden-ratio increment and
        // finalizes it; the finalizer applied to 0x9E3779B97F4A7C15 is the
        // published splitmix64 first output. Guards the exact mixing constants.
        let mut r = SplitMix64::new(0);
        assert_eq!(r.next_u64(), 0xE220_A839_7B1D_CDAF);
    }

    #[test]
    fn split_unit_f32_is_in_open_unit_interval() {
        let mut r = SplitMix64::new(1);
        for _ in 0..1000 {
            let u = r.next_unit_f32();
            assert!(u > 0.0 && u < 1.0, "u = {u} must be in (0, 1)");
        }
    }

    #[test]
    fn gaussian_reuses_the_spare_on_alternate_calls() {
        // Two Gaussian draws consume exactly one uniform pair (cos then the
        // cached sin), so a fresh generator's 3rd draw begins a new pair. Verify
        // the wrapper's stream equals the hand-rolled Box–Muller over the same
        // splitmix64 uniforms (the exact piper-plus construction).
        let mut g = GaussianSplitMix64::new(0x5EED_1234_ABCD_0007);
        let got: Vec<f32> = (0..4).map(|_| g.next_gaussian()).collect();

        let mut u = SplitMix64::new(0x5EED_1234_ABCD_0007);
        let mut expect = Vec::new();
        for _ in 0..2 {
            let u1 = u.next_unit_f32();
            let u2 = u.next_unit_f32();
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f32::consts::PI * u2;
            expect.push(r * theta.cos());
            expect.push(r * theta.sin());
        }
        assert_eq!(got, expect);
    }

    #[test]
    fn xorshift64star_is_deterministic_and_avoids_zero_lock() {
        let mut a = Xorshift64Star::new(0);
        let mut b = Xorshift64Star::new(0); // 0 | 1 == 1, never the zero fixed point
        let seq_a: Vec<u64> = (0..8).map(|_| a.next_u64()).collect();
        let seq_b: Vec<u64> = (0..8).map(|_| b.next_u64()).collect();
        assert_eq!(seq_a, seq_b);
        assert!(
            seq_a.iter().all(|&x| x != 0),
            "scrambled output is non-zero"
        );
    }

    #[test]
    fn xorshift_signed_f32_is_in_symmetric_interval() {
        let mut r = Xorshift64Star::new(0x1234_5678_9ABC_DEF0);
        for _ in 0..1000 {
            let v = r.next_signed_f32();
            assert!((-1.0..1.0).contains(&v), "v = {v} must be in [-1, 1)");
        }
    }
}
