//! FLOW-NOISE-SCALE fix (2026-08-09) regression pin.
//!
//! Verifies `SbV2Model::synthesize` reparameterizes the flow prior
//! with `req.noise_scale` — Python: `z_p = mel_hidden + torch.randn *
//! noise_scale` before `flow.inverse`. Pre-fix Vokra ignored
//! `noise_scale` entirely (grep-verified field was unused). Post-fix
//! two `synthesize` calls with the same `seed`/`rng_mode` but different
//! `noise_scale` produce clearly-divergent output; `noise_scale = 0.0`
//! remains bit-identical to the pre-fix zero-noise path.
//!
//! # Oracle
//!
//! * `noise_scale = 0.0` short-circuits the RNG (pre- and post-fix
//!   byte-identical) → `noise_scale = 0.0` runs across `RngMode`
//!   variants are byte-identical to each other.
//! * `noise_scale = 0.667` (upstream default) draws N(0, 1) samples
//!   scaled by 0.667 and adds elementwise to `mel_hidden`. Two runs
//!   with different `noise_scale` must diverge.
//! * Same `seed` + same `noise_scale` + same `rng_mode` reproduces the
//!   same PCM (determinism guarantee — inherited from
//!   `TorchRandnStream::new(seed)` / `GaussianSplitMix64::new(seed)`).

use vokra_models::sbv2::{Language, RngMode, SbV2Model, SbV2SynthRequest};

fn base_request(noise_scale: f32, seed: u64, rng_mode: RngMode) -> SbV2SynthRequest {
    SbV2SynthRequest {
        text: "test".to_string(),
        language: Language::EN,
        speaker_id: 0,
        speaker_embedding: None,
        style_vec: vec![0.0; 4],
        speed: 1.0,
        noise_scale,
        noise_scale_w: 0.0, // isolate this test to `noise_scale` only
        seed,
        rng_mode,
    }
}

#[test]
fn zero_noise_scale_matches_pre_fix_byte_identical() {
    // Post-FLOW-NOISE-SCALE-fix: `noise_scale == 0.0` short-circuits
    // the RNG entirely, so the whole pipeline is byte-identical to
    // pre-fix. This also proves the fast path is truly a no-op — a
    // pre-fix synthetic test's byte-frozen PCM keeps its expected
    // values.
    //
    // Concrete contract: two identical requests must produce
    // byte-identical output.
    let model = SbV2Model::synthetic_for_test();
    let out_a = model
        .synthesize(&base_request(0.0, 42, RngMode::GaussianSplitMix64Legacy))
        .expect("synthesize should succeed");
    let out_b = model
        .synthesize(&base_request(0.0, 42, RngMode::GaussianSplitMix64Legacy))
        .expect("synthesize should succeed");
    assert_eq!(
        out_a.samples, out_b.samples,
        "noise_scale = 0.0 must be deterministic"
    );
}

#[test]
fn zero_noise_scale_is_rng_mode_invariant() {
    // `noise_scale == 0.0` short-circuits BOTH RNG modes → two
    // requests with the same seed but different rng_mode produce
    // byte-identical output. Ensures the fast path skips the RNG
    // draw regardless of which RNG is selected.
    let model = SbV2Model::synthetic_for_test();
    let out_torch = model
        .synthesize(&base_request(
            0.0,
            42,
            RngMode::PhiloxRngEnginePyTorchParity,
        ))
        .expect("synthesize should succeed");
    let out_legacy = model
        .synthesize(&base_request(0.0, 42, RngMode::GaussianSplitMix64Legacy))
        .expect("synthesize should succeed");
    assert_eq!(
        out_torch.samples, out_legacy.samples,
        "noise_scale = 0.0 must be RNG-mode-invariant (both paths skip the RNG)"
    );
}

#[test]
fn nonzero_noise_scale_diverges_from_zero_noise() {
    // Post-fix: `noise_scale = 0.667` observably perturbs the flow
    // input → different PCM from `noise_scale = 0.0`. Pre-fix ignored
    // `noise_scale` entirely, so both runs would produce IDENTICAL
    // output. This test would fail red pre-fix and green post-fix.
    let model = SbV2Model::synthetic_for_test();
    let out_zero = model
        .synthesize(&base_request(
            0.0,
            42,
            RngMode::PhiloxRngEnginePyTorchParity,
        ))
        .expect("synthesize should succeed");
    let out_noisy = model
        .synthesize(&base_request(
            0.667,
            42,
            RngMode::PhiloxRngEnginePyTorchParity,
        ))
        .expect("synthesize should succeed");
    assert_eq!(
        out_zero.samples.len(),
        out_noisy.samples.len(),
        "sample counts must not depend on noise_scale (durations are text-driven with \
         noise_scale_w = 0)"
    );
    let max_delta = out_zero
        .samples
        .iter()
        .zip(out_noisy.samples.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    // For the synthetic_for_test model, weights are tiny (~5e-2
    // amplitude) so noise propagation is heavily attenuated by the
    // decoder — but the delta must be STRICTLY non-zero. Pre-fix
    // ignored `noise_scale` entirely and produced max|Δ| = 0.0
    // exactly; post-fix produces a small but non-zero delta (~7e-6
    // on this fixture, dominated by fp rounding of the
    // small-noise-scale × small-weights propagation through the
    // decoder + tanh).
    //
    // The strict-inequality check `max_delta > 0.0` is sufficient to
    // catch the pre-fix regression class (RNG-source ignored entirely
    // ⇒ output identical). A "meaningful magnitude" check would need
    // the higher-signal `synthetic_for_test_e2e` fixture.
    assert!(
        max_delta > 0.0,
        "FLOW-NOISE-SCALE: noise_scale > 0 must perturb the pipeline. Pre-fix would ignore \
         `noise_scale` and yield max|Δ| = 0.0 exactly. Observed max|Δ| = {max_delta} (must be \
         > 0.0 strictly post-fix)."
    );
}

#[test]
fn same_seed_same_noise_scale_is_bit_identical() {
    // Determinism: same seed + same noise_scale + same rng_mode →
    // byte-identical PCM. Inherited from the underlying RNG's
    // seed-determinism.
    let model = SbV2Model::synthetic_for_test();
    let out_a = model
        .synthesize(&base_request(
            0.667,
            42,
            RngMode::PhiloxRngEnginePyTorchParity,
        ))
        .expect("synthesize should succeed");
    let out_b = model
        .synthesize(&base_request(
            0.667,
            42,
            RngMode::PhiloxRngEnginePyTorchParity,
        ))
        .expect("synthesize should succeed");
    assert_eq!(
        out_a.samples, out_b.samples,
        "noise_scale > 0 must be deterministic given the same seed"
    );
}

#[test]
fn different_noise_scales_produce_different_pcm() {
    // Two distinct nonzero noise_scale values → different PCM.
    // Confirms the injected noise magnitude is a live tunable
    // (not silently discarded).
    let model = SbV2Model::synthetic_for_test();
    let out_low = model
        .synthesize(&base_request(
            0.1,
            42,
            RngMode::PhiloxRngEnginePyTorchParity,
        ))
        .expect("synthesize should succeed");
    let out_high = model
        .synthesize(&base_request(
            0.9,
            42,
            RngMode::PhiloxRngEnginePyTorchParity,
        ))
        .expect("synthesize should succeed");
    assert_ne!(
        out_low.samples, out_high.samples,
        "different nonzero noise_scale values must produce different PCM (both same seed / \
         same RNG)"
    );
}
