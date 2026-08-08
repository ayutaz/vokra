//! Integration tests for [`PhiloxState`] — the generic (not torch-specific)
//! block-level Philox state machine.
//!
//! `PhiloxState` holds a 128-bit counter (as `u128`) plus a 64-bit key and
//! calls the pure [`philox4x32_10`] block function once per `next_block()`,
//! advancing the counter by one. Two tests suffice to freeze the semantics:
//!
//! 1. the first `next_block()` at counter 0 reproduces Random123 KAT #1
//!    (`philox_kat_zero_seed_zero_ctr`), then the second block differs from
//!    the first (i.e. the counter actually advanced);
//! 2. `seek(N)` is equivalent to N serial `next_block()` calls — the O(1)
//!    seek needed by future parity dumpers that jump to a specific counter
//!    offset without emulating the whole prefix.
//!
//! Reference: Random123 v1.14 §2.4 (counter-based generator semantics).

use vokra_core::rng::{PhiloxState, philox4x32_10};

/// Freezes the counter-advance semantic: after two `next_block()` calls the
/// state has produced two distinct 128-bit outputs, and the very first
/// output equals what the stateless [`philox4x32_10`] would return at
/// `(ctr=[0;4], key=[0;2])` — i.e. `PhiloxState::new(key=[0;2], counter=0)`
/// starts at counter 0, not at counter 1.
#[test]
fn philox_state_counter_advances() {
    let mut s = PhiloxState::new([0, 0], 0);
    let b0 = s.next_block();
    let b1 = s.next_block();
    assert_eq!(
        b0,
        [0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8],
        "block 0 must equal Random123 KAT #1 (all-zero seed / counter=0)"
    );
    assert_ne!(
        b0, b1,
        "counter must advance between successive next_block() calls"
    );
}

/// `seek(N)` must be equivalent to N serial `next_block()` calls: both leave
/// the state at counter=N, so the next `next_block()` after seeking must
/// equal the (N+1)-th call in the serial path. The 1000-step magnitude is
/// large enough to catch any 32/64-bit wrap-around bug in the counter split
/// (counter is `u128`; splitting into 4× `u32` for the block-function ctr
/// input is where a byte-order or lane-order slip would surface).
#[test]
fn philox_state_seek_matches_serial() {
    // Non-zero key so any key-dependent bug (e.g. seek() forgetting the
    // key) is not masked by the all-zero key of the KATs.
    let key = [0xdead_beef, 0xcafe_babe];

    // Path A: 1000 serial advances, then read.
    let mut a = PhiloxState::new(key, 0);
    for _ in 0..1000 {
        a.next_block();
    }
    let a_out = a.next_block();

    // Path B: O(1) seek to counter=1000, then read.
    let mut b = PhiloxState::new(key, 0);
    b.seek(1000);
    let b_out = b.next_block();

    assert_eq!(
        a_out, b_out,
        "seek(1000) must land on the same block as 1000 serial next_block()s"
    );
}

/// Cross-check against the stateless [`philox4x32_10`]: the state machine
/// must be a thin wrapper around the block function. If `next_block()` at
/// counter N differs from `philox4x32_10(counter_as_[u32;4], key)`, the
/// wrapper's counter encoding is buggy. Also verifies the counter-word
/// endianness convention: counter's low 32 bits go to `ctr[0]`, next 32 to
/// `ctr[1]`, and so on (little-endian split), matching torch's
/// `PhiloxRNGEngine.h::incr()` layout.
#[test]
fn philox_state_matches_stateless_block_function() {
    let key = [0x1234_5678, 0x9abc_def0];
    let counter: u128 = 0x0000_0004_0000_0003_0000_0002_0000_0001;
    // Little-endian split: counter[0] = low 32 bits, counter[3] = top 32.
    let expected_ctr = [1u32, 2u32, 3u32, 4u32];

    let mut s = PhiloxState::new(key, counter);
    assert_eq!(s.next_block(), philox4x32_10(expected_ctr, key));
    // And the state must have advanced by exactly one.
    assert_eq!(s.next_block(), philox4x32_10([2u32, 2u32, 3u32, 4u32], key));
}
