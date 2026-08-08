//! End-to-end byte-exact parity between `vokra_core::rng::torch_randn_f32`
//! (Rust) and `tools/parity/torch_philox_dump.py` (Python port of ATen's
//! `PhiloxRNGEngine.h::randn`).
//!
//! Three fixtures cover three (seed, k) points:
//!
//! - seed=0, k=4 — canonical smoke test (16 bytes; the same seed the raw
//!   Philox KATs and one-shot `philox_randn_sample` tests use);
//! - seed=42, k=100 — seed and length diversity (400 bytes);
//! - seed=12345, k=1000 — stress test that spans 1000 blocks =
//!   4000 u32 words, catching any 32-bit wrap-around in the counter split
//!   inside `TorchPhiloxState::next_block` (400 bytes / block × 1000).
//!
//! Fixtures live in `tests/fixtures/rng_torch/` alongside a `README.md`
//! that documents the provenance. If a fixture is missing (never
//! generated or accidentally deleted), the test message points at the
//! regeneration command.

use std::fs;
use std::path::PathBuf;

use vokra_core::rng::torch_randn_f32;

/// Absolute path to a fixture file, resolved against `CARGO_MANIFEST_DIR`
/// (the crate root, which is what `cargo test -p vokra-core` sets).
fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/rng_torch")
        .join(name)
}

/// Reads `path` as bytes and compares against the (seed, k)-generated
/// output byte-for-byte. If the fixture is missing, panics with a message
/// pointing at the regeneration command so a new engineer can rebuild the
/// fixture locally.
fn assert_bytes(seed: u64, k: usize, name: &str) {
    let path = fixture_path(name);
    let expected = fs::read(&path).unwrap_or_else(|_| {
        panic!(
            "fixture {} missing; regenerate via `cd tools/parity && uv run python \
             torch_philox_dump.py --seed {} --randn-samples {} --out {}`",
            path.display(),
            seed,
            k,
            path.display()
        )
    });
    assert_eq!(
        expected.len(),
        k * 4,
        "fixture size mismatch: expected {} bytes ({} f32s * 4), got {} — \
         did the fixture get truncated?",
        k * 4,
        k,
        expected.len(),
    );

    let mut got = vec![0.0_f32; k];
    torch_randn_f32(seed, &mut got);

    // Compare via bytes (not f32 slice equality), because f32 equality
    // makes NaN != NaN and would produce a false negative if a rounding
    // difference sent one sample into a NaN — bytes surface the divergence
    // for what it is (a bit pattern mismatch) rather than swallowing it.
    let got_bytes: Vec<u8> = got.iter().flat_map(|v| v.to_le_bytes()).collect();

    if got_bytes != expected {
        // Emit the first divergent byte offset so the failure message is
        // actionable — hunting a 4000-byte diff by eye is a bad first
        // experience for a maintainer regenerating a fixture.
        let first_diff = got_bytes
            .iter()
            .zip(expected.iter())
            .position(|(a, b)| a != b)
            .unwrap_or(0);
        let sample_idx = first_diff / 4;
        panic!(
            "fixture {} diverged at byte offset {} (sample index {}): \
             got_bytes[{}..{}] = {:?}, expected[{}..{}] = {:?}",
            path.display(),
            first_diff,
            sample_idx,
            first_diff,
            (first_diff + 4).min(got_bytes.len()),
            &got_bytes[first_diff..(first_diff + 4).min(got_bytes.len())],
            first_diff,
            (first_diff + 4).min(expected.len()),
            &expected[first_diff..(first_diff + 4).min(expected.len())],
        );
    }
}

/// Seed 0, k=4 — the canonical smoke test. Sample 0 is the Box-Muller of
/// Random123 KAT #1's block (pinned separately in `rng_philox_randn.rs`).
#[test]
fn torch_randn_seed_0_k_4() {
    assert_bytes(0, 4, "torch_randn_seed0_k4.f32.bin");
}

/// Seed 42, k=100 — seed and length diversity. 100 samples exercise the
/// counter advance 100 times, so a fencepost that skipped an advance
/// would surface as a rotate-by-1 divergence starting at sample 1.
#[test]
fn torch_randn_seed_42_k_100() {
    assert_bytes(42, 100, "torch_randn_seed42_k100.f32.bin");
}

/// Seed 12345, k=1000 — stress test. 1000 samples = 1000 blocks, so any
/// counter-split byte-order slip that only manifests when the counter
/// overflows 32 bits (would happen at counter = 2^32, well beyond 1000)
/// is not exercised here, but this test does exercise the 32-bit ×
/// low-word arithmetic on a wide range of counter values.
#[test]
fn torch_randn_seed_12345_k_1000() {
    assert_bytes(12345, 1000, "torch_randn_seed12345_k1000.f32.bin");
}
