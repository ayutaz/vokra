//! Step 10 — verifies the `SbV2SynthRequest::rng_mode` field routes
//! `SbV2Model::synthesize` to the requested RNG family. Two requests
//! with identical fields except `rng_mode` must produce OBSERVABLY
//! different PCM output; if they didn't, either the dispatch is broken
//! (both paths silently share one RNG) or the two RNGs happen to
//! collide (astronomically unlikely — a 3-phoneme "あいう" utterance
//! consumes ~6 blocks × 4 u32 = 24 u32 words of RNG state per synthesis,
//! and Philox4x32-10 vs. splitmix64 producing identical streams over 24
//! words has probability ~2^-768).
//!
//! Why not compare against a byte-frozen reference?
//! ------------------------------------------------
//! `SbV2Model::synthetic_for_test`'s downstream flow-inverse chain has
//! its own float accumulation whose bound is not yet real-checkpoint-
//! calibrated (Task 28's outstanding work). Any byte-frozen assertion
//! on the full synthesis output would rest on that uncalibrated bound
//! and could break for reasons unrelated to RNG dispatch. The
//! bit-exact RNG parity itself is already frozen at the noise layer by
//! `sbv2_sdp_torch_parity.rs::sdp_noise_matches_torch_philox_seed_0_t_50`.

use vokra_models::sbv2::{Language, RngMode, SbV2Model, SbV2SynthRequest};

fn request_with(rng_mode: RngMode) -> SbV2SynthRequest {
    SbV2SynthRequest {
        text: "あいう".to_string(),
        language: Language::JA,
        speaker_id: 0,
        speaker_embedding: None,
        style_vec: vec![0.0; 4], // matches synthetic_for_test's d_style
        speed: 1.0,
        noise_scale: 0.0,
        // Non-zero noise so the SDP fill loop actually consumes the
        // RNG (short-circuited when noise_scale_w == 0.0).
        noise_scale_w: 0.8,
        seed: 42,
        rng_mode,
    }
}

/// The `rng_mode` field must actually select the RNG: two requests
/// that differ ONLY in `rng_mode` must produce different PCM. If the
/// dispatch in `SbV2Model::synthesize` accidentally ignores the field
/// (e.g. hard-coded `GaussianSplitMix64::new(req.seed)`), the two
/// syntheses would produce identical PCM and this test would fail.
#[test]
fn rng_mode_actually_selects_the_stream() {
    let model = SbV2Model::synthetic_for_test();
    let torch = model
        .synthesize(&request_with(RngMode::PhiloxRngEnginePyTorchParity))
        .expect("synthesize (torch parity) should succeed");
    let legacy = model
        .synthesize(&request_with(RngMode::GaussianSplitMix64Legacy))
        .expect("synthesize (legacy) should succeed");

    // Both syntheses must be non-empty and finite (sanity — a broken
    // dispatch that returns an empty buffer would false-fail the `ne`
    // assertion below without diagnosing the real cause).
    assert!(
        !torch.samples.is_empty(),
        "torch-parity path must produce non-empty PCM"
    );
    assert!(
        !legacy.samples.is_empty(),
        "legacy path must produce non-empty PCM"
    );
    assert!(
        torch.samples.iter().all(|s| s.is_finite()),
        "torch-parity PCM must be all-finite"
    );
    assert!(
        legacy.samples.iter().all(|s| s.is_finite()),
        "legacy PCM must be all-finite"
    );

    // The dispatch assertion: distinct RNG streams must produce
    // distinct outputs. Compare either PCM length OR sample bytes —
    // Philox and splitmix64 producing the same duration sequence
    // AND the same PCM bytes is a 2^-768-probability event.
    assert!(
        torch.samples != legacy.samples || torch.samples.len() != legacy.samples.len(),
        "torch-parity and legacy RNGs must produce observably different \
         PCM (lengths {} vs {}) — if this fails, `SbV2Model::synthesize` \
         is ignoring `req.rng_mode` and both paths silently use one RNG",
        torch.samples.len(),
        legacy.samples.len(),
    );
}

/// Two synthesise calls with the SAME rng_mode and SAME seed must
/// produce identical PCM — reproducibility invariant that both RNG
/// paths must preserve. This test guards against a stateful bug where
/// the RNG state leaks between calls (e.g. a lazy-init that only fires
/// on the second call).
#[test]
fn same_rng_mode_and_seed_is_reproducible() {
    let model = SbV2Model::synthetic_for_test();
    let a = model
        .synthesize(&request_with(RngMode::PhiloxRngEnginePyTorchParity))
        .expect("synthesize should succeed");
    let b = model
        .synthesize(&request_with(RngMode::PhiloxRngEnginePyTorchParity))
        .expect("synthesize should succeed");
    assert_eq!(
        a.samples, b.samples,
        "torch-parity path must be reproducible across calls at the same seed"
    );

    let c = model
        .synthesize(&request_with(RngMode::GaussianSplitMix64Legacy))
        .expect("synthesize should succeed");
    let d = model
        .synthesize(&request_with(RngMode::GaussianSplitMix64Legacy))
        .expect("synthesize should succeed");
    assert_eq!(
        c.samples, d.samples,
        "legacy path must be reproducible across calls at the same seed"
    );
}

/// The `Default` for `RngMode` must be `PhiloxRngEnginePyTorchParity`
/// (torch parity), matching the module doc's stated intent. A future
/// maintainer flipping the default back to legacy would silently break
/// downstream torch-parity tests without any code path in the RNG
/// selection itself failing — this test catches that at the enum
/// boundary.
#[test]
fn rng_mode_default_is_torch_parity() {
    assert_eq!(RngMode::default(), RngMode::PhiloxRngEnginePyTorchParity);
}
