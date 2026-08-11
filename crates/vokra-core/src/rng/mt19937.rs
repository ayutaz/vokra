//! Mersenne Twister MT19937 engine — bit-exactly mirrors torch's
//! `aten/src/ATen/core/MT19937RNGEngine.h::mt19937_engine`
//! (BSD-3-Clause).
//!
//! # Why MT19937 (not Philox) on the CPU path
//!
//! `torch.randn(N, device='cpu')` — the default backend — draws its
//! standard-normal samples through `at::normal_distribution<double>`,
//! whose uniform source is `at::mt19937`, not `at::Philox4_32`. The
//! `PhiloxRNGEngine.h` in torch source carries an explicit disclaimer
//! that its own `randn()` "is not used anywhere except for tests in
//! cpu_generator_test.cpp" (lines 39-41 of that header) — so the
//! previous Vokra port of `PhiloxRNGEngine.h::randn`, while
//! algorithmically correct against Random123 KATs (and thus useful for a
//! future CUDA cuRAND parity path with different subsequence packing),
//! matches **no real torch backend** as-is. Bisect report
//! [`wf_20fa0933-53d/…`] documents the empirical proof: `torch.randn(4)`
//! with seed=0 on CPU is `[0x3fc53f5c, 0xbe963c50, 0xc00b7149,
//! 0x3f1184b6]` (u32 bits of the f32 samples) — bit-identical to the
//! output produced by this file's engine plus `TorchRandnStream`'s
//! Box-Muller pair, and nowhere close to the Philox port's output at
//! the same seed. See [`super::normal_kernel`]'s module doc for the
//! full pipeline this engine feeds.
//!
//! # Algorithm reference (MT19937RNGEngine.h)
//!
//! - `init_with_uint32(seed)` (lines 156-165): `state[0] = seed &
//!   0xFFFFFFFF; for j in 1..624 { state[j] = 1812433253 * (state[j-1]
//!   ^ (state[j-1] >> 30)) + j }` — Matsumoto & Nishimura's original
//!   MT init (only the low 32 bits of `seed` participate; the u64
//!   passed to `torch.manual_seed(N)` is truncated at this seam,
//!   matching CPUGeneratorImpl.h's own seed handling).
//! - `next_state()` (lines 175-189): the classic MT19937 twist — two
//!   loops with `MERSENNE_M = 397`, `MATRIX_A = 0x9908B0DF`, `UMASK =
//!   0x80000000`, `LMASK = 0x7FFFFFFF`.
//! - `operator()` tempering: `y ^= y>>11; y ^= (y<<7) & 0x9D2C5680; y
//!   ^= (y<<15) & 0xEFC60000; y ^= y>>18` — the standard MT19937
//!   output tempering.
//!
//! # Scope caveat
//!
//! This engine matches `torch.randn(N, device='cpu')` for **`N < 16` or
//! non-contiguous** tensors. For contiguous `N >= 16`, torch dispatches
//! to `normal_fill` (ATen/native/cpu/DistributionTemplates.h:168-220)
//! which fills the tensor with uniforms first, then applies a
//! *different* Box-Muller formula in 16-wide SIMD blocks with **no pair
//! caching**. The SBV2 SDP noise buffer is small (`2 * text_seq_len`,
//! typically 4-20 elements per test phoneme string), so the small-N
//! path is the one that fires in practice — but a caller filling large
//! contiguous buffers cannot expect byte-parity here.

use core::num::Wrapping;

/// State array length for MT19937 (Matsumoto & Nishimura 1998).
const N: usize = 624;

/// Twist parameter: middle-word offset for the recurrence.
const MERSENNE_M: usize = 397;

/// Twisted-feedback constant from the paper's characteristic polynomial.
const MATRIX_A: u32 = 0x9908_B0DF;

/// Upper mask (bit 31) — selects the "upper" bit of a state word for
/// the twist step.
const UMASK: u32 = 0x8000_0000;

/// Lower mask (bits 30..0) — selects the "lower" 31 bits of a state
/// word for the twist step.
const LMASK: u32 = 0x7FFF_FFFF;

/// Constant used in MT19937's `init_with_uint32(seed)` recurrence
/// (Knuth, TAOCP vol. 2, 3.3.4 Line 26) — bit-exactly matches
/// MT19937RNGEngine.h:161.
const INIT_MULTIPLIER: u32 = 1_812_433_253;

/// Torch's MT19937 engine, bit-exact against
/// `aten/src/ATen/core/MT19937RNGEngine.h::mt19937_engine`.
///
/// Holds a 624-word state array plus a decrement-toward-zero `left`
/// counter that triggers the `next_state()` twist when exhausted, and
/// a `next` cursor into the state array. This exactly mirrors torch's
/// three private fields (`state_`, `left_`, `next_`) — a byte-for-byte
/// port so bugs in future maintenance surface as obvious divergences
/// from that header rather than as subtle semantic drift.
#[derive(Clone, Debug)]
pub struct TorchMt19937Engine {
    /// 624-word MT19937 state array.
    state: [u32; N],
    /// Number of untempered outputs remaining in `state`; when it hits
    /// zero the next `next_u32()` call runs `next_state()` and resets
    /// this to `N`.
    left: i32,
    /// Cursor into `state` for the next output.
    next: usize,
}

impl TorchMt19937Engine {
    /// Constructs a fresh engine seeded with the low 32 bits of `seed`.
    ///
    /// **Only the low 32 bits participate** — matches
    /// `CPUGeneratorImpl.h`'s seed handling for the same reason
    /// MT19937RNGEngine.h's `init_with_uint32` takes a `uint32_t`:
    /// MT19937 is a 32-bit generator, so widening the seed API to
    /// `u64` is purely a caller-ergonomics choice and the upper 32
    /// bits are dropped at this seam. `torch.manual_seed(0x1_0000_0000)`
    /// produces the same stream as `torch.manual_seed(0)` for this
    /// reason.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        let mut state = [0u32; N];
        // Matsumoto & Nishimura 1998 init recurrence, bit-exact per
        // MT19937RNGEngine.h:156-165.
        state[0] = (seed & 0xFFFF_FFFF) as u32;
        for j in 1..N {
            // wrapping_mul / wrapping_add match torch's u32 arithmetic
            // (torch relies on 32-bit unsigned overflow at every
            // multiply — a debug-mode overflow panic would diverge
            // from the header, so wrapping is correct).
            let prev = state[j - 1];
            let mixed = prev ^ (prev >> 30);
            state[j] = (Wrapping(INIT_MULTIPLIER) * Wrapping(mixed) + Wrapping(j as u32)).0;
        }
        Self {
            state,
            left: 1,
            next: 0,
        }
    }

    /// Runs one MT19937 twist over the internal state, refilling the
    /// 624-word buffer with fresh untempered outputs.
    ///
    /// Verbatim port of MT19937RNGEngine.h:175-189 (the three-branch
    /// twist: first `N - MERSENNE_M` words, then the wraparound `N -
    /// MERSENNE_M .. N - 1` block, then the trailing single-word case
    /// at index `N - 1`).
    fn twist(&mut self) {
        // Branch 1: i in 0..N - MERSENNE_M (= 0..227).
        for i in 0..(N - MERSENNE_M) {
            let y = (self.state[i] & UMASK) | (self.state[i + 1] & LMASK);
            self.state[i] =
                self.state[i + MERSENNE_M] ^ (y >> 1) ^ if (y & 1) != 0 { MATRIX_A } else { 0 };
        }
        // Branch 2: i in N - MERSENNE_M..N - 1 (= 227..623).
        for i in (N - MERSENNE_M)..(N - 1) {
            let y = (self.state[i] & UMASK) | (self.state[i + 1] & LMASK);
            // `i + MERSENNE_M - N` wraps into 0..MERSENNE_M-1 range —
            // the wraparound the paper's characteristic polynomial
            // requires.
            self.state[i] =
                self.state[i + MERSENNE_M - N] ^ (y >> 1) ^ if (y & 1) != 0 { MATRIX_A } else { 0 };
        }
        // Branch 3: i = N - 1, using state[0] as the "next" word to
        // close the ring.
        let y = (self.state[N - 1] & UMASK) | (self.state[0] & LMASK);
        self.state[N - 1] =
            self.state[MERSENNE_M - 1] ^ (y >> 1) ^ if (y & 1) != 0 { MATRIX_A } else { 0 };
    }

    /// Returns the next tempered 32-bit output from the engine.
    ///
    /// Decrements the `left` counter; if it would go to zero, runs the
    /// `twist()` step first (bulk refilling all 624 words) and resets
    /// `left = N` and `next = 0`. Then reads `state[next]`, advances
    /// `next`, and applies the MT19937 tempering:
    ///
    /// ```text
    /// y ^= y >> 11
    /// y ^= (y << 7)  & 0x9D2C5680
    /// y ^= (y << 15) & 0xEFC60000
    /// y ^= y >> 18
    /// ```
    ///
    /// Bit-exact match against MT19937RNGEngine.h's `operator()`.
    pub fn next_u32(&mut self) -> u32 {
        self.left -= 1;
        if self.left <= 0 {
            self.twist();
            self.left = N as i32;
            self.next = 0;
        }
        let mut y = self.state[self.next];
        self.next += 1;
        // Tempering — bit-exact per MT19937 paper §4 and torch header.
        y ^= y >> 11;
        y ^= (y << 7) & 0x9D2C_5680;
        y ^= (y << 15) & 0xEFC6_0000;
        y ^= y >> 18;
        y
    }

    /// Returns the next 64-bit output as `(hi << 32) | lo`, with `hi`
    /// drawn **first** and `lo` **second** from the engine.
    ///
    /// The hi-first packing matches ATen's `random64()` helper (see
    /// `ATen/core/DistributionsHelper.h:100-114`); every downstream
    /// distribution — `uniform_real<double>`, `normal_distribution`, …
    /// — takes 53 bits from this packed word for its uniform draw,
    /// so any endianness slip here would break byte-parity for every
    /// distribution simultaneously.
    pub fn random64(&mut self) -> u64 {
        let hi = self.next_u32() as u64;
        let lo = self.next_u32() as u64;
        (hi << 32) | lo
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Bit-exact pin for the first 8 tempered outputs of MT19937(0),
    /// derived by hand-executing `mt19937_engine::init_with_uint32(0)`
    /// plus `operator()` eight times in a Python probe (see the bisect
    /// report captured in `wf_20fa0933-53d`) and cross-checked against
    /// a real `torch.manual_seed(0); torch.Generator().state()`
    /// unpacked-and-tempered run.
    ///
    /// If this test fails, the twist step or the tempering constants
    /// diverged from MT19937RNGEngine.h — every downstream parity
    /// test (`torch.rand`, `torch.randn`) would fail simultaneously,
    /// so this is the isolated failure mode for the engine itself.
    #[test]
    fn mt19937_seed_0_first_8_u32s_match_torch() {
        let mut mt = TorchMt19937Engine::new(0);
        let expected: [u32; 8] = [
            0x8C7F_0AAC,
            0x97C4_AA2F,
            0xB716_A675,
            0xD821_CCC0,
            0x9A4E_B343,
            0xDBA2_52FB,
            0x8B7D_76C3,
            0xD8E5_7D67,
        ];
        for (i, want) in expected.iter().enumerate() {
            let got = mt.next_u32();
            assert_eq!(
                got, *want,
                "MT19937(0) u32 #{i} = {got:#010x}, want {want:#010x}"
            );
        }
    }

    /// Cross-check the uniform-real half of the pipeline: torch's CPU
    /// `uniform_real<float>` on MT19937 masks each u32 to its low 24
    /// bits and divides by 2^24. The bisect report documents
    /// `torch.rand(4)` seed=0 = `[0.4963, 0.7682, 0.0885, 0.1320]`,
    /// which agrees with `((raw_u32 & 0xFFFFFF) as f32) / 2^24` on the
    /// first four MT19937(0) outputs — pin those exact low-24-bit
    /// values here so a future `torch.rand` port has an isolated
    /// oracle that does NOT go through Box-Muller (which would fold
    /// any uniform-mask bug into the normal-sample bit pattern).
    #[test]
    fn mt19937_seed_0_uniform_low_24_bits_match_torch_rand() {
        let mut mt = TorchMt19937Engine::new(0);
        // Low 24 bits of MT19937(0) outputs 0..4:
        //   0x8c7f0aac & 0xffffff = 0x7f0aac → 0.4963
        //   0x97c4aa2f & 0xffffff = 0xc4aa2f → 0.7682
        //   0xb716a675 & 0xffffff = 0x16a675 → 0.0885
        //   0xd821ccc0 & 0xffffff = 0x21ccc0 → 0.1320
        let expected_low24: [u32; 4] = [0x7F_0AAC, 0xC4_AA2F, 0x16_A675, 0x21_CCC0];
        // sanity: bisect-report float values, computed as
        // low24 / 2^24 (rounded to 4 decimals).
        let expected_uniform: [f32; 4] = [0.4963, 0.7682, 0.0885, 0.1320];
        for i in 0..4 {
            let raw = mt.next_u32();
            let low24 = raw & 0x00FF_FFFF;
            assert_eq!(
                low24, expected_low24[i],
                "MT19937(0) low24 #{i} = {low24:#08x}, want {:#08x} (raw = {raw:#010x})",
                expected_low24[i]
            );
            let uniform = (low24 as f32) / ((1u32 << 24) as f32);
            let rounded = (uniform * 10_000.0).round() / 10_000.0;
            assert_eq!(
                rounded, expected_uniform[i],
                "MT19937(0) uniform #{i} = {uniform:.4}, want {}",
                expected_uniform[i]
            );
        }
    }

    /// `random64()` must pack `(hi << 32) | lo` — a byte-swap here
    /// would silently produce a valid-looking but wrong stream, so
    /// this test isolates the endianness contract.
    #[test]
    fn random64_is_hi_shifted_then_lo() {
        let mut mt = TorchMt19937Engine::new(0);
        let hi = mt.next_u32() as u64;
        let lo = mt.next_u32() as u64;
        let expected = (hi << 32) | lo;

        let mut mt2 = TorchMt19937Engine::new(0);
        let got = mt2.random64();
        assert_eq!(got, expected, "random64() must be (hi << 32) | lo");
    }

    /// Determinism: same seed → same stream. A regression here would
    /// mean the twist step accidentally reads external state, breaking
    /// every parity test simultaneously.
    #[test]
    fn same_seed_same_stream() {
        let mut a = TorchMt19937Engine::new(0xDEAD_BEEF);
        let mut b = TorchMt19937Engine::new(0xDEAD_BEEF);
        for i in 0..2000 {
            assert_eq!(a.next_u32(), b.next_u32(), "divergence at index {i}");
        }
    }

    /// Twist is exercised: >= 624 draws forces a `twist()` call. The
    /// engine must continue to produce non-repeating output; this
    /// catches a "twist accidentally re-emits the pre-twist state"
    /// regression that would show as sample 624 == sample 0.
    #[test]
    fn twist_step_produces_fresh_output() {
        let mut mt = TorchMt19937Engine::new(0);
        let first = mt.next_u32();
        // Advance past the first twist boundary.
        for _ in 0..999 {
            let _ = mt.next_u32();
        }
        let later = mt.next_u32();
        assert_ne!(
            first, later,
            "twist step must produce fresh output — a match here means the state was not \
             actually mutated (fixed-point bug in the twist step)"
        );
    }

    /// Low 32 bits of the seed only. Verifies that
    /// `torch.manual_seed(u64::MAX)` produces the same stream as
    /// `torch.manual_seed(u32::MAX as u64)` — the u64 API is caller
    /// ergonomics; MT19937 is a 32-bit generator.
    #[test]
    fn seed_uses_only_low_32_bits() {
        let mut a = TorchMt19937Engine::new(0xFFFF_FFFF);
        let mut b = TorchMt19937Engine::new(0x1234_5678_FFFF_FFFF);
        for i in 0..8 {
            assert_eq!(
                a.next_u32(),
                b.next_u32(),
                "seed high bits must be dropped (i = {i})"
            );
        }
    }
}
