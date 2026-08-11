//! Pure Philox4x32-10 block function (Salmon et al., "Parallel Random
//! Numbers: As Easy as 1, 2, 3", SC'11).
//!
//! Zero state, zero allocations, no floating point — safe under
//! `--no-default-features` (the M5-03 no_std subset). The three KATs in
//! `crates/vokra-core/tests/rng_philox_kat.rs` freeze the algorithm before
//! any counter / seed-init / Box-Muller layer is added on top, so any typo
//! in a constant or an off-by-one round count fails in isolation.
//!
//! # Layout convention
//!
//! Following Random123 v1.14, the 128-bit block is a `[u32; 4]` in the order
//! `[R0, R1, R2, R3]` (little-endian per word). One "block" is one call to
//! [`philox4x32_10`] and yields 128 bits of PRNG output.
//!
//! # Constants
//!
//! - `M0`, `M1` — the two mulhilo32 multipliers, per Random123 v1.14
//!   `philox.h` `PHILOX_M4x32_0` / `PHILOX_M4x32_1`.
//! - `W0`, `W1` — the two Weyl-sequence key-bump increments (Random123's
//!   `PHILOX_W32_0` / `PHILOX_W32_1`), the golden-ratio conjugate
//!   `0x9E3779B9` and the sqrt(3)-1 fractional bits `0xBB67AE85`.
//! - `PHILOX_ROUNDS = 10` — nine iterations that each bump the key by
//!   `(W0, W1)` after mixing, terminated by a tenth round that does NOT bump
//!   the key. Cutting the loop at 9 (forgetting the final round) or 10 (with
//!   a spurious extra key bump) both fail KAT #1.

// Do not use `#![no_std]` inside a submodule — the crate-level attribute in
// `lib.rs` already selects std vs. no_std. This file is intentionally free of
// any std/core dependency beyond u32 primitives so it compiles under either.

/// Philox4x32-10 first multiplier (Random123 v1.14 `PHILOX_M4x32_0`).
pub const PHILOX_M0: u32 = 0xD251_1F53;

/// Philox4x32-10 second multiplier (Random123 v1.14 `PHILOX_M4x32_1`).
pub const PHILOX_M1: u32 = 0xCD9E_8D57;

/// Weyl-sequence key-bump increment for `key[0]` (Random123 v1.14
/// `PHILOX_W32_0` — the golden-ratio conjugate).
pub const PHILOX_W0: u32 = 0x9E37_79B9;

/// Weyl-sequence key-bump increment for `key[1]` (Random123 v1.14
/// `PHILOX_W32_1` — the sqrt(3)-1 fractional bits).
pub const PHILOX_W1: u32 = 0xBB67_AE85;

/// Round count for Philox4x32-10 (the "-10" suffix). Nine key-bumping
/// iterations plus one final mixing round without a key bump.
pub const PHILOX_ROUNDS: usize = 10;

/// Full-width 32×32 → 64-bit multiply, returning `(lo, hi)`. The 64-bit
/// wrapping-multiply in the middle is exact (no overflow up to 2⁶⁴), so the
/// split into two `u32`s is bit-exact against Random123's inline-asm mulhi.
#[inline]
#[must_use]
fn mulhilo32(a: u32, b: u32) -> (u32, u32) {
    let product = (a as u64).wrapping_mul(b as u64);
    (product as u32, (product >> 32) as u32)
}

/// One Philox4x32 round: two independent mulhilos, then a fixed lane
/// permutation that folds each hi word into the "other" pair together with
/// one key word (Random123 v1.14 `philox.h` `_philox4x32round`).
///
/// The lane pairing — `M0` multiplies `ctr[0]` and its `hi` word combines
/// with `ctr[3]` and `key[0]`; `M1` multiplies `ctr[2]` and its `hi` word
/// combines with `ctr[1]` and `key[1]` — is critical to Philox's diffusion
/// property: swapping which multiplier pairs with which counter word fails
/// KAT #2 (all-ones state and key).
#[inline]
#[must_use]
fn single_round(ctr: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let (lo0, hi0) = mulhilo32(PHILOX_M0, ctr[0]);
    let (lo1, hi1) = mulhilo32(PHILOX_M1, ctr[2]);
    [hi1 ^ ctr[1] ^ key[0], lo1, hi0 ^ ctr[3] ^ key[1], lo0]
}

/// The full 10-round Philox4x32 block function. `ctr` is the 128-bit
/// counter, `key` is the 64-bit key; output is the 128-bit encrypted block.
///
/// Deterministic and stateless — the same `(ctr, key)` always produces the
/// same output. The three KATs in `rng_philox_kat.rs` verify this against
/// Random123 v1.14's published vectors.
///
/// # Round schedule
///
/// Loops nine times, each iteration doing `ctr ← single_round(ctr, key)`
/// followed by `key[0] += W0; key[1] += W1`. Terminates with a **tenth**
/// `single_round` that does NOT bump the key (matching Random123's
/// `_philox4x32bumpkey` placement inside the loop, not after it).
#[inline]
#[must_use]
pub fn philox4x32_10(ctr: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let mut c = ctr;
    let mut k = key;
    // Nine key-bumping iterations. `PHILOX_ROUNDS - 1` is the loop bound
    // rather than a hard-coded 9 so a future PHILOX_ROUNDS change (e.g. a
    // "Philox4x32-7" variant for benchmarking) automatically follows.
    for _ in 0..PHILOX_ROUNDS - 1 {
        c = single_round(c, k);
        k[0] = k[0].wrapping_add(PHILOX_W0);
        k[1] = k[1].wrapping_add(PHILOX_W1);
    }
    // Final round without a subsequent key bump — omitting this fails KAT
    // #1; keeping the trailing `k[…] += …` would fail every KAT.
    single_round(c, k)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// mulhilo32 is the arithmetic primitive under the whole algorithm; if
    /// the split into `(lo, hi)` is byte-swapped the whole Philox output
    /// gets rotated 32 bits, which manifests as every KAT failing in the
    /// same way. This unit test isolates that failure mode.
    #[test]
    fn mulhilo32_basic() {
        // 0xFFFF_FFFF * 0xFFFF_FFFF = 0xFFFF_FFFE_0000_0001
        //   -> lo = 0x0000_0001, hi = 0xFFFF_FFFE
        let (lo, hi) = mulhilo32(0xFFFF_FFFF, 0xFFFF_FFFF);
        assert_eq!(lo, 0x0000_0001, "low word of max² must be 1");
        assert_eq!(hi, 0xFFFF_FFFE, "high word of max² must be 2⁶⁴ - 2⁶⁴⁻³²");

        // 0x8000_0000 * 2 = 0x1_0000_0000 -> lo = 0, hi = 1
        let (lo2, hi2) = mulhilo32(0x8000_0000, 2);
        assert_eq!((lo2, hi2), (0, 1));
    }

    /// `single_round` at zero inputs must produce the pure Weyl-only round
    /// value (all mulhilo products are 0, so the output is just the lane
    /// permutation of `ctr[1..=3]` XOR key words XOR zero). Freezes the
    /// exact lane pairing so a future permutation typo fails immediately.
    ///
    /// The `identity_op` lint is allowed here so the `hi1 XOR ctr[1] XOR
    /// key[0]` structure of each expected lane is visible in the test,
    /// rather than pre-collapsed to just `key[0]`.
    #[test]
    #[allow(clippy::identity_op)]
    fn single_round_zeros_is_pure_lane_permutation() {
        assert_eq!(
            single_round([0, 0, 0, 0], [0xdead_beef, 0xcafe_babe]),
            [
                0 ^ 0 ^ 0xdead_beef, // hi1 ^ ctr[1] ^ key[0]
                0,                   // lo1
                0 ^ 0 ^ 0xcafe_babe, // hi0 ^ ctr[3] ^ key[1]
                0,                   // lo0
            ],
        );
    }

    /// Sanity check that the full block-function is not accidentally the
    /// identity — a common failure mode of a mis-copied lane permutation is
    /// that some inputs pass through unchanged.
    #[test]
    fn philox4x32_10_is_not_identity() {
        let ctr = [1, 2, 3, 4];
        let out = philox4x32_10(ctr, [5, 6]);
        assert_ne!(out, ctr, "block function must diffuse the input");
    }
}
