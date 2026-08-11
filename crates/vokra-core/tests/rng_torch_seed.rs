//! Integration tests for [`TorchPhiloxState`] — the PyTorch-compatible seed
//! → PhiloxState derivation.
//!
//! Mirrors `aten/src/ATen/core/PhiloxRNGEngine.h::PhiloxRNGEngine(seed,
//! subsequence, offset)` layout: `key = [seed as u32, (seed >> 32) as u32]`
//! and each `next_block()` uses `ctr = [offset_lo, offset_hi, 0, 0]` then
//! `offset += 1`. For a `torch.cuda.manual_seed(N)` invocation the
//! subsequence is 0 and the offset starts at 0 (this file's
//! `from_manual_seed`).
//!
//! The two tests pin the seed=0 first block to Random123 KAT #1 (proving
//! the key derivation and initial-offset semantics) and check that a
//! different seed produces a different first block (proving the seed
//! actually enters the key). Together with the KATs in `rng_philox_kat.rs`
//! and the state-machine tests in `rng_philox_state.rs`, these three tests
//! freeze the whole raw-u32 stream before Box-Muller is layered on.
//!
//! # Why the seed=0 first block equals Random123 KAT #1
//!
//! `from_manual_seed(0)` sets `key = [0, 0]` and `offset = 0`, and
//! `next_block()` at `offset = 0` uses `ctr = [0, 0, 0, 0]`. That is
//! exactly the input to KAT #1 (`philox4x32_10([0;4], [0;2])`), so the
//! block output must match bit-for-bit.

use vokra_core::rng::TorchPhiloxState;

/// The canonical `torch.cuda.manual_seed(0)` first block equals Random123
/// KAT #1. This one assertion pins three separate things: (a) the seed →
/// key packing (both u32 words are 0 for seed=0), (b) the initial offset =
/// 0 mapping to `ctr = [0, 0, 0, 0]`, and (c) that we call
/// [`philox4x32_10`] with the correct argument order (key second, not
/// first — mixing them up would produce a spurious output that happens to
/// still be non-trivial). If ANY of the three is off, this test fails.
#[test]
fn torch_seed_0_first_block_equals_kat_1() {
    let mut s = TorchPhiloxState::from_manual_seed(0);
    assert_eq!(
        s.next_block(),
        [0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8],
        "seed=0 first block must equal Random123 KAT #1"
    );
    assert_eq!(
        s.offset, 1,
        "offset must advance by exactly 1 per next_block() call"
    );
}

/// A non-zero seed must actually change the output: if the seed → key
/// packing were wrong (e.g. always packed as `[0, 0]` regardless of
/// input), every seed would produce KAT #1 forever. This is the
/// negative-control that catches that silent failure mode.
#[test]
fn torch_seed_42_first_block_differs_from_kat_1() {
    let mut s = TorchPhiloxState::from_manual_seed(42);
    assert_ne!(
        s.next_block(),
        [0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8],
        "seed=42 must produce a different first block from seed=0"
    );
}

/// `with_offset(seed, N).next_block()` must land at the same block that N
/// successive `from_manual_seed(seed).next_block()`s would produce —
/// this is the "start at offset N" fast path for parity-fixture
/// regeneration.
#[test]
fn torch_seed_with_offset_matches_serial_advance() {
    let seed = 12345_u64;
    let n = 250_u64;

    let mut a = TorchPhiloxState::from_manual_seed(seed);
    for _ in 0..n {
        let _ = a.next_block();
    }

    let mut b = TorchPhiloxState::with_offset(seed, n);
    assert_eq!(
        a.next_block(),
        b.next_block(),
        "with_offset(seed, N) must equal N advances of from_manual_seed(seed)"
    );
}
