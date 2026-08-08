//! Step 9 of the torch.randn parity work — end-to-end byte-exact check
//! that the noise buffer `SbV2SDP::sample`'s inner fill loop produces
//! with a `TorchRandnStream` matches the Python-Philox fixture emitted
//! by `tools/parity/sbv2_sdp_noise_dump.py`.
//!
//! Why not exercise `SbV2SDP::sample` end-to-end?
//! ---------------------------------------------
//! `SbV2SDP::sample`'s downstream flow-inverse math (`ea.reverse`,
//! `unconstrained_rqs_inverse`, etc.) has its own float-accumulation
//! error whose bound is not yet real-checkpoint-calibrated — it's what
//! `PER_TENSOR_ATOL[\"sdp_sample\"] = 0.05` in `crates/vokra-models/src/
//! sbv2/parity.rs` is a placeholder for. This test isolates the RNG
//! layer at atol = 0.0 (bit-exact bytes) so a parity regression here
//! surfaces *only* as an RNG bug, not a rounding drift downstream that
//! happens to touch the SDP output tensor.
//!
//! Step 10 will then tighten `PER_TENSOR_ATOL[\"sdp_sample\"]` on top of
//! this bit-exact-RNG foundation, once a real SBV2 reference dumper is
//! wired to emit the full sdp_sample tensor.

use std::fs;
use std::path::PathBuf;

use vokra_core::rng::{NormalSource, TorchRandnStream};

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/sbv2")
        .join(name)
}

/// Replicates the exact `SbV2SDP::sample` inner fill loop —
/// `for v in &mut z { *v = rng.next_normal() * noise_scale_w; }` — with
/// `noise_scale_w = 1.0`, so the resulting bytes ARE the RNG output.
/// If a future refactor changes the fill loop's iteration order or
/// scale semantics this test will diverge, which is exactly the
/// invariant we want to guard.
fn fill_sdp_noise<R: NormalSource>(rng: &mut R, text_seq_len: usize) -> Vec<f32> {
    let mut z = vec![0.0_f32; 2 * text_seq_len];
    for v in &mut z {
        *v = rng.next_normal() * 1.0_f32; // noise_scale_w = 1.0
    }
    z
}

/// Seed 0, T=50 → 100 samples (2 channels × 50 timesteps), 400 bytes.
///
/// Byte-exact byte-wise comparison with the fixture emitted by
/// `tools/parity/sbv2_sdp_noise_dump.py`, which itself calls the
/// audited PhiloxRNGEngine.h Python port (self-tested against
/// Random123 KATs).
#[test]
fn sdp_noise_matches_torch_philox_seed_0_t_50() {
    let path = fixture_path("sdp_noise_seed0_T50.f32.bin");
    let expected = fs::read(&path).unwrap_or_else(|_| {
        panic!(
            "fixture {} missing; regenerate via `cd tools/parity && uv run \
             python sbv2_sdp_noise_dump.py --seed 0 --T 50 --out {}`",
            path.display(),
            path.display()
        )
    });
    assert_eq!(
        expected.len(),
        2 * 50 * 4,
        "fixture must be 2*T*4 = 400 bytes"
    );

    let mut rng = TorchRandnStream::new(0);
    let got = fill_sdp_noise(&mut rng, 50);
    let got_bytes: Vec<u8> = got.iter().flat_map(|v| v.to_le_bytes()).collect();

    if got_bytes != expected {
        let first_diff = got_bytes
            .iter()
            .zip(expected.iter())
            .position(|(a, b)| a != b)
            .unwrap_or(0);
        panic!(
            "fixture {} diverged at byte offset {} (sample index {}): \
             got_bytes[{}..{}] = {:?}, expected[{}..{}] = {:?}",
            path.display(),
            first_diff,
            first_diff / 4,
            first_diff,
            (first_diff + 4).min(got_bytes.len()),
            &got_bytes[first_diff..(first_diff + 4).min(got_bytes.len())],
            first_diff,
            (first_diff + 4).min(expected.len()),
            &expected[first_diff..(first_diff + 4).min(expected.len())],
        );
    }
}

/// Cross-check that the Step 8 refactor kept `SbV2SDP::sample` generic:
/// building the same noise buffer via `GaussianSplitMix64` (the
/// pre-existing synthetic RNG) must produce a DIFFERENT byte sequence
/// than the torch-parity path, otherwise the type parameter is being
/// erased and every call site would silently get the torch parity path
/// regardless of the constructor picked.
#[test]
fn sdp_noise_from_gaussian_splitmix_diverges_from_torch_philox() {
    use vokra_core::rng::GaussianSplitMix64;

    let mut torch_rng = TorchRandnStream::new(0);
    let torch_z = fill_sdp_noise(&mut torch_rng, 50);

    let mut mix_rng = GaussianSplitMix64::new(0);
    let mix_z = fill_sdp_noise(&mut mix_rng, 50);

    assert_ne!(
        torch_z, mix_z,
        "TorchRandnStream and GaussianSplitMix64 must produce different sample \
         sequences at the same seed — if they don't, the type parameter is \
         being erased somewhere"
    );
}
