//! PyTorch-compatible seed → Philox state derivation.
//!
//! Mirrors torch's `aten/src/ATen/core/PhiloxRNGEngine.h::PhiloxRNGEngine`
//! constructor layout (BSD-3-Clause):
//!
//! ```text
//! key      = [seed as u32, (seed >> 32) as u32]
//! counter  = [offset_lo, offset_hi, subsequence_lo, subsequence_hi]
//! ```
//!
//! For `torch.cuda.manual_seed(N)` on a single stream the subsequence is 0
//! and the offset starts at 0 — [`TorchPhiloxState::from_manual_seed`].
//! The offset advances by exactly one per `next_block()`, matching torch's
//! `incr()` method (one call = one 128-bit block = 4 u32s).
//!
//! Kept out of `philox_state.rs` so alternative seed derivations
//! (TensorFlow, JAX) can wrap the shared state machine without touching
//! this file.

use super::philox_round::philox4x32_10;
use super::philox_state::PhiloxState;

/// The torch-parity Philox state — a `u64` seed plus a `u64` offset (in
/// 128-bit blocks), matching torch's `PhiloxRNGEngine.h` layout.
///
/// The offset is measured in **4-u32 blocks**, not individual u32s (torch
/// convention). One `next_block()` call = one 128-bit block = 4 u32
/// samples fed to Box-Muller as 2 uniforms. A caller that needs mid-block
/// state must extend this type; the pure-Philox stream never has partial
/// blocks.
#[derive(Clone, Debug)]
pub struct TorchPhiloxState {
    /// The u64 seed passed to `torch.cuda.manual_seed(N)`.
    pub seed: u64,
    /// Number of 128-bit blocks already consumed. Advances by 1 per
    /// `next_block()` call; overflow wraps (see `next_block` for the
    /// rationale — 2⁶⁴ blocks is 2⁶⁶ f32 samples, far beyond any
    /// realistic workload).
    pub offset: u64,
}

impl TorchPhiloxState {
    /// Initial state for `torch.cuda.manual_seed(N)`. Subsequence and
    /// offset both start at 0.
    #[must_use]
    pub const fn from_manual_seed(seed: u64) -> Self {
        Self { seed, offset: 0 }
    }

    /// Constructs a state at an arbitrary `offset`, so a parity fixture
    /// dumper can regenerate the (offset..offset+N) slice of
    /// `torch.randn(seed=…)` output without emulating the whole prefix.
    #[must_use]
    pub const fn with_offset(seed: u64, offset: u64) -> Self {
        Self { seed, offset }
    }

    /// Produces the next 128-bit block: packs seed → key,
    /// (offset_lo, offset_hi, 0, 0) → ctr, calls [`philox4x32_10`], then
    /// increments the offset by 1 (wrapping).
    ///
    /// Wrap-around at `offset == u64::MAX` yields silence rather than a
    /// panic — 2⁶⁴ blocks is 2⁶⁶ f32 samples, so overflow is impossible
    /// in practice, but a debug-mode panic in a production stream would
    /// be worse than the wrap-around silence.
    pub fn next_block(&mut self) -> [u32; 4] {
        let key = [self.seed as u32, (self.seed >> 32) as u32];
        // torch's PhiloxRNGEngine.h places the offset in ctr[0..2] and the
        // subsequence (unused in our single-stream case) in ctr[2..4]. See
        // module doc for the reference to `aten/src/ATen/core/
        // PhiloxRNGEngine.h`'s constructor layout.
        let ctr = [self.offset as u32, (self.offset >> 32) as u32, 0, 0];
        let out = philox4x32_10(ctr, key);
        self.offset = self.offset.wrapping_add(1);
        out
    }

    /// Conversion to the generic [`PhiloxState`] — mainly for interop with
    /// code that already accepts `PhiloxState` and does not need to know
    /// the seed encoding was torch-shaped. The counter starts at
    /// `self.offset` since the torch convention places the offset in the
    /// low 64 bits of the 128-bit counter (subsequence = 0 means the top
    /// 64 bits are 0).
    #[must_use]
    pub fn into_philox_state(self) -> PhiloxState {
        let key = [self.seed as u32, (self.seed >> 32) as u32];
        PhiloxState::new(key, self.offset as u128)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `into_philox_state` must be equivalent to `next_block` in outputs:
    /// both encode the offset as the low 64 bits of a 128-bit counter
    /// with `[subsequence_lo, subsequence_hi] = [0, 0]` on top.
    #[test]
    fn into_philox_state_matches_next_block_output() {
        let mut torch = TorchPhiloxState::with_offset(0xdead_beef, 7);
        let torch_out = torch.next_block();

        let mut generic = TorchPhiloxState::with_offset(0xdead_beef, 7).into_philox_state();
        let generic_out = generic.next_block();

        assert_eq!(torch_out, generic_out);
    }

    /// Seed roundtrip: the two u32 words of `key` must equal (seed_lo,
    /// seed_hi). Isolates the seed packing failure from the block-function
    /// call in `next_block`.
    #[test]
    fn seed_packing_is_little_endian_u64() {
        // 0xAABB_CCDD_1122_3344 splits into (0x1122_3344, 0xAABB_CCDD).
        let seed = 0xAABB_CCDD_1122_3344u64;
        let generic = TorchPhiloxState::from_manual_seed(seed).into_philox_state();
        // PhiloxState.key is pub(crate) so this cross-module test can see
        // it — see philox_state.rs for the field visibility rationale.
        assert_eq!(generic.key, [0x1122_3344, 0xAABB_CCDD]);
    }
}
