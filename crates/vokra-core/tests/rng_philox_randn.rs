//! Bit-exact cross-check between Vokra's Rust `philox_randn_sample` and the
//! independent Python port `tools/parity/torch_philox_dump.py`, which itself
//! self-tests against Random123 v1.14 KAT vectors before every fixture
//! generation.
//!
//! # Reference derivation
//!
//! The expected values below were produced by:
//!
//! ```bash
//! cd tools/parity && uv run python torch_philox_dump.py \\
//!     --seed 0 --randn-samples 2 --out /tmp/philox_ref.json
//! ```
//!
//! The dumper (a) reimplements PhiloxRNGEngine.h::randn in pure Python, (b)
//! self-tests against the three Random123 KATs before writing anything, and
//! (c) emits `{blocks, samples, sample_bits}` in the JSON debug format so
//! this test can bake the exact f32 bit pattern (not just a decimal string
//! that might round-trip lossy) as a `u32` `to_bits()` comparison.
//!
//! # Sample-0 sanity check
//!
//! At seed=0 the first Philox block is Random123 KAT #1 (0x6627_E8D5,
//! 0xE169_C58D, 0xBC57_AC4C, 0x9B00_DBD8) — pinned separately in
//! `rng_philox_kat.rs::philox_kat_zero_seed_zero_ctr` and
//! `rng_torch_seed.rs::torch_seed_0_first_block_equals_kat_1`. This file
//! then pins the Box-Muller transform of that block to its resulting
//! standard-normal sample.

use vokra_core::rng::philox_randn_sample;

/// Sample 0 from seed 0: the Box-Muller output of Random123 KAT #1
/// consumed as `(u1=block[0], u2=block[1])`. Bit pattern verified against
/// the Python dumper's `sample_bits[0]`.
#[test]
fn philox_randn_seed_0_sample_0_matches_python_reference() {
    let block = [0x6627_e8d5u32, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8];
    let got = philox_randn_sample(block);
    assert_eq!(
        got.to_bits(),
        0x3DFD_EDAFu32,
        "got={got} ({:#010x}), expected bits 0x3DFDEDAF (~0.1240)",
        got.to_bits()
    );
}

/// Sample 1 from seed 0: the second block, freshly minted from
/// `TorchPhiloxState(seed=0)`'s counter=1 output, consumed the same way.
/// The bit pattern is the Python dumper's `sample_bits[1]` (0xBEC6_D750,
/// ≈ -0.3884).
#[test]
fn philox_randn_seed_0_sample_1_matches_python_reference() {
    // Block[1] from seed=0 — the counter=1 Philox output (see
    // `rng_philox_state.rs::philox_state_counter_advances` for the
    // adjacent block-0 pin). Values transcribed from the Python dumper's
    // `blocks[1]` field.
    let block = [0xf8e4_cca4u32, 0x5cb2_00db, 0xb1a5_74eb, 0x097e_ff67];
    let got = philox_randn_sample(block);
    assert_eq!(
        got.to_bits(),
        0xBEC6_D750u32,
        "got={got} ({:#010x}), expected bits 0xBEC6D750 (~-0.3884)",
        got.to_bits()
    );
}
