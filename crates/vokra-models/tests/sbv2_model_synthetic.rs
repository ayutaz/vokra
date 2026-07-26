//! `SbV2Model::synthesize` (Task 23) integration tests: the full
//! G2P → text encoder → BERT (+ bridge) → speaker/style → SDP → length
//! regulation → flow → HiFi-GAN decoder pipeline, exercised over
//! `SbV2Model::synthetic_for_test()`'s tiny deterministic components (no
//! real checkpoint is involved — that lands with the Task 24-27 converter).
//!
//! The first two tests prove the wiring end-to-end for each language this
//! scaffold routes ("あいう" for JA hits `SbV2Phonemizer::synthetic_for_test`'s
//! hiragana char mapping; "test" for EN hits its ASCII-letter mapping — see
//! `tests/sbv2_g2p.rs`). The third test proves `SbV2Model`'s
//! `vokra_core::TtsEngine` adapter is a faithful thin wrapper: an equivalent
//! `SynthesisRequest` (routed through the trait) must reproduce the exact
//! same PCM as calling `SbV2Model::synthesize` directly.

use vokra_core::{SynthesisRequest, TtsEngine};
use vokra_models::sbv2::{Language, SbV2Model, SbV2SynthRequest};

#[test]
fn synthesize_ja_returns_non_empty_pcm() {
    let model = SbV2Model::synthetic_for_test();
    let req = SbV2SynthRequest {
        text: "あいう".to_string(),
        language: Language::JA,
        speaker_id: 0,
        style_vec: vec![0.0; 4], // matches synthetic_for_test's d_style (4)
        speed: 1.0,
        noise_scale: 0.0,
        noise_scale_w: 0.0,
        seed: 42,
    };

    let audio = model.synthesize(&req).expect("synthesize should succeed");

    assert!(!audio.samples.is_empty(), "PCM output must be non-empty");
    assert_eq!(audio.sample_rate, 44_100);
}

#[test]
fn synthesize_en_returns_non_empty_pcm() {
    let model = SbV2Model::synthetic_for_test();
    let req = SbV2SynthRequest {
        text: "test".to_string(),
        language: Language::EN,
        speaker_id: 0,
        style_vec: vec![0.0; 4], // matches synthetic_for_test's d_style (4)
        speed: 1.0,
        noise_scale: 0.0,
        noise_scale_w: 0.0,
        seed: 42,
    };

    let audio = model.synthesize(&req).expect("synthesize should succeed");

    assert!(!audio.samples.is_empty(), "PCM output must be non-empty");
    assert_eq!(audio.sample_rate, 44_100);
}

/// `TtsEngine::synthesize` must be a faithful thin adapter: a
/// `SynthesisRequest` with `deterministic()` set and no engine-inapplicable
/// fields (`speaker_embedding`/`prosody_features` both `None`) reproduces
/// the exact same PCM as the equivalent `SbV2SynthRequest` passed directly
/// to `SbV2Model::synthesize` — same text, JA (the adapter's default
/// language and `.with_language("ja")`'s explicit selection agree), speaker
/// 0, the identity (all-zero) style vector, unit speed, and both noise
/// scales zeroed (the adapter's `deterministic` mapping).
#[test]
fn tts_engine_adapter_matches_direct_synthesize() {
    let model = SbV2Model::synthetic_for_test();

    let request = SynthesisRequest::new("test")
        .with_language("ja")
        .deterministic();
    let via_trait =
        TtsEngine::synthesize(&model, &request).expect("trait synthesize should succeed");

    let direct_req = SbV2SynthRequest {
        text: "test".to_string(),
        language: Language::JA,
        speaker_id: 0,
        style_vec: vec![0.0; 4], // matches synthetic_for_test's d_style (4)
        speed: 1.0,
        noise_scale: 0.0,
        noise_scale_w: 0.0,
        seed: 0,
    };
    let direct = model
        .synthesize(&direct_req)
        .expect("direct synthesize should succeed");

    assert_eq!(via_trait.samples, direct.samples);
    assert_eq!(via_trait.sample_rate, direct.sample_rate);
}
