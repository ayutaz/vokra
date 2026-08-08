//! Generic block-level Philox4x32-10 state machine — the Vokra-native
//! wrapper over the stateless [`philox4x32_10`](super::philox4x32_10) block
//! function.
//!
//! Holds a 128-bit counter (as `u128` — the widest integer Rust's stable
//! std provides) and a 64-bit key; each [`PhiloxState::next_block`] call
//! feeds the current counter to the block function and advances the counter
//! by one. [`PhiloxState::seek`] jumps to an arbitrary counter in O(1), a
//! primitive future parity fixtures need for regenerating a specific
//! `torch.randn(seed, offset=N)` slice without emulating the whole prefix.
//!
//! Kept separate from `seed_init.rs` so alternative seed derivations
//! (TensorFlow's `weird_tf_constructor` from tract/sonos, JAX's Threefry,
//! …) can wrap this state machine without duplicating the counter /
//! increment / seek logic.

use super::philox_round::philox4x32_10;

/// Block-level Philox4x32-10 state — a 128-bit counter plus a 64-bit key.
///
/// The counter is stored as a `u128` (little-endian split into `[u32; 4]`
/// at each block call) so `seek()` is O(1) and 128-bit arithmetic uses
/// Rust's native `wrapping_add` — no hand-rolled 4×u32 arithmetic and its
/// carry-propagation footguns.
#[derive(Clone, Debug)]
pub struct PhiloxState {
    /// The 64-bit block-function key (two `u32` words).
    pub(crate) key: [u32; 2],
    /// The 128-bit counter, incremented by one after each `next_block()`.
    pub(crate) counter: u128,
}

impl PhiloxState {
    /// Creates a new state with the given `key` and initial `counter`.
    ///
    /// `counter=0` starts at Random123's KAT #1 basis when `key=[0, 0]`;
    /// `counter=N` starts N blocks in (equivalent to N successive
    /// `next_block()` calls from `counter=0`).
    #[must_use]
    pub const fn new(key: [u32; 2], counter: u128) -> Self {
        Self { key, counter }
    }

    /// Advances the state by one block: feeds the current counter to
    /// [`philox4x32_10`], increments the counter (wrapping), and returns
    /// the 128-bit encrypted output.
    ///
    /// Counter overflow is handled by `wrapping_add(1)`: the counter is
    /// 128-bit, so overflow requires 2¹²⁸ ≈ 3.4·10³⁸ blocks, i.e. more
    /// samples than any conceivable Vokra workload — but overflow-panic in
    /// release mode is worse than wrap-around silence, so wrap-around wins.
    pub fn next_block(&mut self) -> [u32; 4] {
        let ctr = Self::counter_words(self.counter);
        let out = philox4x32_10(ctr, self.key);
        self.counter = self.counter.wrapping_add(1);
        out
    }

    /// Seeks the counter to `counter` in O(1). The next `next_block()` will
    /// return the same output N successive `next_block()` calls from
    /// `counter=0` would land on (verified by
    /// `philox_state_seek_matches_serial` in
    /// `tests/rng_philox_state.rs`).
    pub fn seek(&mut self, counter: u128) {
        self.counter = counter;
    }

    /// Splits a 128-bit counter into a `[u32; 4]` block-function argument,
    /// **little-endian** (low 32 bits → `ctr[0]`, next 32 → `ctr[1]`, …).
    ///
    /// Mirrors torch's `PhiloxRNGEngine.h::incr()` layout — the increment
    /// there is `state_[0]++; if state_[0] == 0 { state_[1]++; if state_[1]
    /// == 0 { … } }`, i.e. little-endian carry propagation. Any change to
    /// the split order silently misaligns torch parity.
    #[inline]
    #[must_use]
    fn counter_words(c: u128) -> [u32; 4] {
        [
            c as u32,
            (c >> 32) as u32,
            (c >> 64) as u32,
            (c >> 96) as u32,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Endianness pin: `counter_words(0x…04_03_02_01)` must produce
    /// `[1, 2, 3, 4]`, not `[4, 3, 2, 1]`. Isolates the byte-order failure
    /// mode from the block-function output tests in the integration suite.
    #[test]
    fn counter_words_is_little_endian() {
        assert_eq!(
            PhiloxState::counter_words(0x0000_0004_0000_0003_0000_0002_0000_0001),
            [1u32, 2, 3, 4]
        );
        // All-zero counter is all-zero words.
        assert_eq!(PhiloxState::counter_words(0), [0u32; 4]);
        // Top word only.
        assert_eq!(
            PhiloxState::counter_words(0xdead_beef_u128 << 96),
            [0, 0, 0, 0xdead_beef]
        );
    }

    /// The stored counter after `next_block()` must be old + 1. Guards
    /// against a fencepost where `next_block()` accidentally increments
    /// before feeding the block function (which would skip Random123 KAT
    /// #1 at the initial `counter=0` call).
    #[test]
    fn next_block_increments_after_feeding() {
        let mut s = PhiloxState::new([0, 0], 42);
        let _ = s.next_block();
        assert_eq!(s.counter, 43, "counter must be old + 1 after next_block");
    }
}
