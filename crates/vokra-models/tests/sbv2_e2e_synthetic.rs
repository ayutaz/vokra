//! `SbV2Model::synthesize` e2e-scale synthetic test (Task 42): proves the
//! full pipeline — over `SbV2Model::synthetic_for_test_e2e()`'s
//! e2e-shaped-but-still-synthetic components (no real checkpoint is
//! involved, same caveat as `tests/sbv2_model_synthetic.rs` (Task 23) and
//! `tests/parity_sbv2_synthetic.rs` (Task 27), which both use the
//! *separate*, smaller `synthetic_for_test()` factory instead) —
//! produces PCM that is not just "non-empty" (Task 23's bar) but plausibly
//! audio-shaped: more than one second at 44.1 kHz, every sample finite,
//! and a non-silent peak amplitude, for both JA and EN input text.
//!
//! `synthetic_for_test_e2e()` is a factory wholly separate from
//! `synthetic_for_test()` — see that method's doc for why: touching
//! `synthetic_for_test()` itself (its decoder upsample ladder or SDP flow
//! stack) would change the exact PCM length
//! `tests/parity_sbv2_synthetic.rs`'s `synthetic_shape_invariants_hold`
//! pins (`12` samples for 3-phoneme "あいう"), regressing Task 27. This
//! file's two tests below instead exercise the new factory exclusively,
//! and `synthetic_for_test_e2e_does_not_perturb_original_factory` closes
//! the loop by re-affirming Task 27's exact contract still holds
//! unperturbed.

use vokra_models::sbv2::{Language, SbV2Model, SbV2SynthRequest};

/// Shared e2e-scale request builder, varying only `text`/`language` — see
/// `SbV2Model::synthetic_for_test_e2e`'s doc for why `noise_scale_w: 0.0`
/// is load-bearing here (not just a determinism nicety, as in
/// `tests/parity_sbv2_synthetic.rs`'s `ja_request`): it is what makes the
/// factory's single SDP coupling layer collapse to a *fixed* duration (40)
/// at every phoneme position, which is the whole mechanism this test's
/// `> 44_100` assertion relies on.
fn e2e_request(text: &str, language: Language) -> SbV2SynthRequest {
    SbV2SynthRequest {
        text: text.to_string(),
        language,
        speaker_id: 0,
        style_vec: vec![0.0; 4], // matches synthetic_for_test_e2e's d_style (4)
        speed: 1.0,
        noise_scale: 0.0,
        noise_scale_w: 0.0, // load-bearing — see this file's module doc
        seed: 42,
    }
}

/// Asserts the four e2e bars this task's brief specifies: PCM longer than
/// 1 second at 44.1 kHz, every sample finite, and a non-silent peak
/// amplitude (`> 0.001`).
fn assert_meets_e2e_bar(samples: &[f32], label: &str) {
    assert!(
        samples.len() > 44_100,
        "{label} output must be > 1s at 44.1kHz (44,100 samples), got {} samples",
        samples.len()
    );
    assert!(
        samples.iter().all(|s| s.is_finite()),
        "{label} output must be all-finite"
    );
    let peak = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    assert!(
        peak > 0.001,
        "{label} output peak amplitude must exceed 0.001 (non-silent), got {peak}"
    );
}

#[test]
fn synthesize_ja_meets_e2e_bar() {
    let model = SbV2Model::synthetic_for_test_e2e();
    let audio = model
        .synthesize(&e2e_request("こんにちは", Language::JA))
        .expect("JA synthesize should succeed");

    assert_eq!(audio.sample_rate, 44_100);
    assert_meets_e2e_bar(&audio.samples, "JA \"こんにちは\"");
}

#[test]
fn synthesize_en_meets_e2e_bar() {
    let model = SbV2Model::synthetic_for_test_e2e();
    let audio = model
        .synthesize(&e2e_request("hello world", Language::EN))
        .expect("EN synthesize should succeed");

    assert_eq!(audio.sample_rate, 44_100);
    assert_meets_e2e_bar(&audio.samples, "EN \"hello world\"");
}

/// Bonus regression guard (not itself one of the brief's 4 conditions):
/// same seed still produces byte-identical PCM through the e2e-scale
/// factory too — `SbV2Model::synthesize` holds no internal mutable state
/// (`tests/parity_sbv2_synthetic.rs`'s
/// `synthetic_same_seed_produces_byte_identical_pcm` pins the identical
/// property on the smaller `synthetic_for_test()` factory; this closes the
/// same loop for `synthetic_for_test_e2e()`).
#[test]
fn synthetic_e2e_same_seed_produces_byte_identical_pcm() {
    let model = SbV2Model::synthetic_for_test_e2e();
    let req = e2e_request("こんにちは", Language::JA);

    let audio1 = model.synthesize(&req).expect("first synthesize");
    let audio2 = model.synthesize(&req).expect("second synthesize");

    assert_eq!(
        audio1.samples, audio2.samples,
        "same seed must produce byte-identical PCM"
    );
}

/// Task 27 regression guard: `synthetic_for_test_e2e` is additive-only —
/// `synthetic_for_test()`'s own exact 12-sample contract
/// (`tests/parity_sbv2_synthetic.rs`'s `synthetic_shape_invariants_hold`)
/// must still hold, unperturbed, alongside the new factory. This is a
/// belt-and-suspenders re-check in *this* file (the canonical assertion
/// lives in `tests/parity_sbv2_synthetic.rs` and is unmodified by this
/// task) so a future reader of this file sees the non-regression claim
/// documented next to the change that could have threatened it.
#[test]
fn synthetic_for_test_e2e_does_not_perturb_original_factory() {
    let model = SbV2Model::synthetic_for_test();
    let req = SbV2SynthRequest {
        text: "あいう".to_string(),
        language: Language::JA,
        speaker_id: 0,
        style_vec: vec![0.0; 4],
        speed: 1.0,
        noise_scale: 0.0,
        noise_scale_w: 0.0,
        seed: 42,
    };

    let audio = model.synthesize(&req).expect("synthesize should succeed");

    const EXPECTED_MEL_SEQ_LEN: usize = 3; // "あいう" == 3 phonemes, duration 1 each
    const TOTAL_UPSAMPLE: usize = 2 * 2; // synthetic_for_test's upsample_rates = [2, 2]
    assert_eq!(
        audio.samples.len(),
        EXPECTED_MEL_SEQ_LEN * TOTAL_UPSAMPLE,
        "synthetic_for_test()'s exact-length contract (Task 27) must survive \
         synthetic_for_test_e2e()'s addition unperturbed"
    );
}
