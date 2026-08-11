//! `SbV2Model::synthesize` parity-focused synthetic tests (Task 27):
//! same-seed determinism + output-shape invariants, over
//! `SbV2Model::synthetic_for_test()`'s tiny deterministic components (no
//! real checkpoint is involved — see `tests/sbv2_model_synthetic.rs`'s doc
//! for that same caveat, which applies here too).
//!
//! Complements rather than duplicates `tests/sbv2_model_synthetic.rs`
//! (Task 23's end-to-end JA + EN wiring smoke test, plus the `TtsEngine`
//! adapter check): this file narrows to JA only (EN routing is already
//! proven there) and instead focuses on the two properties a *parity*
//! test cares about — determinism (a numerical-diff comparison is only
//! meaningful once both sides are reproducible) and output shape.
//! `sbv2::parity::tolerance_for` (Task 26) has no reference waveform to
//! diff against yet: `synthetic_for_test`'s weights are procedurally
//! generated, not a real trained checkpoint, so there is no upstream
//! PyTorch forward pass to compare — that lands with Task 28's real,
//! HF-checkpoint-gated parity fixture (`PER_TENSOR_ATOL`'s own "Scaffold
//! caveat" doc). `synthetic_shape_invariants_hold` below instead documents
//! where a real per-tensor assertion will plug in.

use vokra_models::sbv2::{Language, RngMode, SbV2Model, SbV2SynthRequest, tolerance_for};

/// Builds the shared JA request every test below starts from ("あいう" —
/// 3 hiragana chars, each a distinct entry in
/// `SbV2Phonemizer::synthetic_for_test`'s char mapping, so this always
/// produces exactly 3 phonemes — see `synthetic_shape_invariants_hold`),
/// varying only `seed` — keeps each test's input visibly identical bar the
/// one field it actually exercises. `noise_scale_w: 0.0` makes every
/// predicted duration exactly `1` regardless of `seed`
/// (`SbV2SDP::sample`'s doc), which is what makes
/// `synthetic_shape_invariants_hold`'s exact output-length assertion
/// possible; `synthetic_different_seeds_produce_different_pcm` below
/// overrides it back to nonzero specifically because it needs `seed` to
/// matter.
fn ja_request(seed: u64) -> SbV2SynthRequest {
    SbV2SynthRequest {
        text: "あいう".to_string(),
        language: Language::JA,
        speaker_id: 0,
        speaker_embedding: None, // Blocker 3: legacy synthetic lookup path
        style_vec: vec![0.0; 4], // matches synthetic_for_test's d_style (4)
        speed: 1.0,
        noise_scale: 0.0,
        noise_scale_w: 0.0,
        seed,
        // Synthetic test: keep the pre-Step-10 splitmix64 stream so any
        // byte-frozen synthetic assertion continues to hold. The Step 9
        // `sbv2_sdp_torch_parity` test proves the torch-parity path
        // separately; this file only checks the SBV2 wiring around a
        // deterministic stream.
        rng_mode: RngMode::GaussianSplitMix64Legacy,
    }
}

/// Brief (a): synthetic weights still produce non-empty PCM.
#[test]
fn synthetic_synthesize_returns_non_empty_pcm() {
    let model = SbV2Model::synthetic_for_test();
    let audio = model
        .synthesize(&ja_request(42))
        .expect("synthesize should succeed");

    assert!(!audio.samples.is_empty(), "PCM output must be non-empty");
    assert_eq!(audio.sample_rate, 44_100);
}

/// Brief (b): same seed -> byte-identical PCM (determinism). Two
/// independent `synthesize` calls on the same model with the same request
/// must reproduce exactly: the model holds no internal mutable state that
/// would let call order matter, and `GaussianSplitMix64::new(req.seed)` is
/// freshly re-seeded every call (`SbV2Model::synthesize`'s step 6). Uses a
/// nonzero `noise_scale_w` (unlike `ja_request`'s default) so the
/// duration predictor's Gaussian draw actually participates in the
/// output — with `noise_scale_w == 0.0` *every* seed collapses to the
/// same all-`1`s duration vector (see `ja_request`'s doc), which would
/// make this test pass even if `req.seed` were silently never reaching
/// the RNG at all.
#[test]
fn synthetic_same_seed_produces_byte_identical_pcm() {
    let model = SbV2Model::synthetic_for_test();
    let mut req = ja_request(42);
    req.noise_scale_w = 0.8; // SBV2's documented SDP default (mod.rs's TtsEngine adapter doc)

    let audio1 = model.synthesize(&req).expect("first synthesize");
    let audio2 = model.synthesize(&req).expect("second synthesize");

    assert_eq!(
        audio1.samples, audio2.samples,
        "same seed must produce byte-identical PCM"
    );
    assert_eq!(audio1.sample_rate, audio2.sample_rate);
}

/// Bonus regression guard (not itself one of the brief's 3 conditions):
/// distinct seeds produce distinct PCM once the duration predictor's
/// Gaussian draw actually participates (`noise_scale_w > 0.0`) — the
/// converse of `synthetic_same_seed_produces_byte_identical_pcm`, closing
/// the loop on "seed is honored" vs. "seed is ignored" (a bug either test
/// alone cannot distinguish: this one alone could pass by coincidence, the
/// other alone would pass even for an ignored seed).
#[test]
fn synthetic_different_seeds_produce_different_pcm() {
    let model = SbV2Model::synthetic_for_test();
    let mut req_a = ja_request(42);
    req_a.noise_scale_w = 0.8;
    let mut req_b = req_a.clone();
    req_b.seed = 43;

    let audio_a = model.synthesize(&req_a).expect("synthesize seed 42");
    let audio_b = model.synthesize(&req_b).expect("synthesize seed 43");

    assert_ne!(
        audio_a.samples, audio_b.samples,
        "different seeds (noise_scale_w > 0) must not collapse to the same PCM"
    );
}

/// Brief (c): output shape invariants hold pipeline-wide.
/// `ja_request`'s `noise_scale_w == 0.0` makes every predicted duration
/// exactly `1` (`SbV2SDP::sample`'s doc), so for 3-phoneme "あいう" input
/// `mel_seq_len == 3` deterministically; `synthetic_for_test`'s HiFi-GAN
/// attrs (`upsample_rates = [2, 2]`, `upsample_kernel_sizes = [4, 4]`, the
/// `kernel = 2 * stride` convention) then make the decoder's output length
/// exactly `mel_seq_len * (2 * 2)` (`SbV2Decoder::generate`'s doc) — `12`
/// samples, not just "non-empty".
///
/// `tolerance_for` (Task 26) has no reference waveform to compare against
/// yet (see this file's module doc). This test documents the call site a
/// real per-tensor assertion will use once Task 28's real fixture exists,
/// e.g.:
/// ```text
/// let atol = tolerance_for("waveform");
/// assert!(max_abs_diff(&audio.samples, &reference_pcm) <= atol);
/// ```
/// For now it only pins that the lookup itself is reachable and returns
/// the documented default for a tensor name with no per-tensor override
/// (`"waveform"` is not in `PER_TENSOR_ATOL` — see that table's doc).
#[test]
fn synthetic_shape_invariants_hold() {
    let model = SbV2Model::synthetic_for_test();
    let audio = model
        .synthesize(&ja_request(42))
        .expect("synthesize should succeed");

    assert_eq!(audio.sample_rate, 44_100);
    const EXPECTED_MEL_SEQ_LEN: usize = 3; // "あいう" == 3 phonemes, duration 1 each
    const TOTAL_UPSAMPLE: usize = 2 * 2; // synthetic_for_test's upsample_rates = [2, 2]
    assert_eq!(
        audio.samples.len(),
        EXPECTED_MEL_SEQ_LEN * TOTAL_UPSAMPLE,
        "PCM length must equal mel_seq_len * total_upsample_factor exactly"
    );

    // Wave-9 (2026-08-09): `waveform` was promoted from ATOL_DEFAULT (0.01)
    // to Measured override 1.5 after CI run 31303426623 measured max |Δ| =
    // 0.9248 through the HiFi-GAN vocoder stack on real fixtures. See
    // `PER_TENSOR_ATOL`'s `"waveform"` block-doc + ADR sbv2-parity-atol §5
    // for the full derivation (cross-plat libm through ~600k transcendental
    // calls). This pin fires alongside `sbv2_parity_atol_calibration.rs`
    // whenever the override drifts — updating either side alone (breaking
    // the redundant-recording rule in memory `feedback-honest-parity-atol`)
    // now turns this test red on top of the calibration status test.
    assert_eq!(tolerance_for("waveform"), 1.5);
}
