//! Random123 v1.14 Philox4x32-10 Known-Answer Tests (KATs).
//!
//! Freezes the algorithm structure — the four bit-mixing constants (M0, M1,
//! W0, W1), the mulhilo32 lane pairing (M0×ctr[0] cross-pairs with M1×ctr[2]),
//! and the 10-round schedule (9 iterations with a key bump after each,
//! terminated by a 10th round with NO key bump) — before any state, seed
//! initialisation, or Box-Muller layer is added. Any typo in a constant, a
//! swapped lane, or an off-by-one round count fails one of these three tests
//! in isolation, so debugging a Step 4/6/9 fixture divergence never has to
//! guess whether the fault is in the block function itself.
//!
//! Reference: Salmon, Moraes, Dror, Shaw, "Parallel Random Numbers: As Easy
//! as 1, 2, 3", SC'11 §3.1 (algorithm); Random123 v1.14
//! `include/Random123/philox.h` (bit-identical implementation the KATs below
//! are drawn from, published under the BSD-3-Clause DE Shaw Research licence).

use vokra_core::rng::philox4x32_10;

/// KAT #1 — the all-zero seed / all-zero counter is the canonical entry in
/// every Random123 test vector table and doubles as the "seed 0 first block"
/// baseline that `TorchPhiloxState::from_manual_seed(0)` must reproduce
/// (Step 4). If the four constants are typo'd or the round count is off,
/// this test almost certainly catches it before the more exotic seeds do.
#[test]
fn philox_kat_zero_seed_zero_ctr() {
    assert_eq!(
        philox4x32_10([0, 0, 0, 0], [0, 0]),
        [0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8],
        "Random123 KAT #1 (all-zero) — verify M0/M1/W0/W1 constants and \
         10-round schedule"
    );
}

/// KAT #2 — all-ones state and key. Distinguishes "M0 and M1 swapped" from
/// "keys W0 and W1 swapped" that KAT #1 (which uses all zeros for both) does
/// not exercise, because a zero key never triggers a key bump.
#[test]
fn philox_kat_all_ones() {
    assert_eq!(
        philox4x32_10(
            [0xffff_ffff, 0xffff_ffff, 0xffff_ffff, 0xffff_ffff],
            [0xffff_ffff, 0xffff_ffff]
        ),
        [0x408f_276d, 0x41c8_3b0e, 0xa20b_c7c6, 0x6d54_51fd],
        "Random123 KAT #2 (all-ones) — asymmetric across keys, catches \
         W0/W1 or M0/M1 swaps that a zero-key test cannot"
    );
}

/// KAT #3 — Random123 v1.14's canonical "pi digits as key" vector. Third
/// independent check; a bug that survives the first two would need to hit
/// the pi-digits basis coincidentally, which is astronomically unlikely.
#[test]
fn philox_kat_pi_key() {
    assert_eq!(
        philox4x32_10(
            [0x243f_6a88, 0x85a3_08d3, 0x1319_8a2e, 0x0370_7344],
            [0xa409_3822, 0x299f_31d0]
        ),
        [0xd16c_fe09, 0x94fd_cceb, 0x5001_e420, 0x2412_6ea1],
        "Random123 KAT #3 (pi-digits key) — third independent basis, \
         catches any bug that survives KAT #1 and KAT #2 coincidentally"
    );
}
