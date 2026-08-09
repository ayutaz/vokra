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

use vokra_core::{SynthesisRequest, TtsEngine, VokraError};
use vokra_models::sbv2::{
    Language, PhonemizeFixture, PhonemizeResult, RngMode, SbV2Model, SbV2SynthRequest,
};

#[test]
fn synthesize_ja_returns_non_empty_pcm() {
    let model = SbV2Model::synthetic_for_test();
    let req = SbV2SynthRequest {
        text: "あいう".to_string(),
        language: Language::JA,
        speaker_id: 0,
        speaker_embedding: None, // Blocker 3: legacy synthetic lookup path
        style_vec: vec![0.0; 4], // matches synthetic_for_test's d_style (4)
        speed: 1.0,
        noise_scale: 0.0,
        noise_scale_w: 0.0,
        seed: 42,
        // Non-empty-PCM smoke test — RNG choice irrelevant because
        // `noise_scale_w = 0.0` short-circuits the RNG (`SbV2SDP::sample`
        // skips the fill loop). Legacy for symmetry with the other
        // synthetic tests in this file.
        rng_mode: RngMode::GaussianSplitMix64Legacy,
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
        speaker_embedding: None, // Blocker 3: legacy synthetic lookup path
        style_vec: vec![0.0; 4], // matches synthetic_for_test's d_style (4)
        speed: 1.0,
        noise_scale: 0.0,
        noise_scale_w: 0.0,
        seed: 42,
        // See `synthesize_ja_returns_non_empty_pcm` for the RNG-choice
        // rationale (noise_scale_w = 0.0 short-circuits the RNG).
        rng_mode: RngMode::GaussianSplitMix64Legacy,
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
///
/// The input text `"あい"` is 2 hiragana chars both present in
/// `SbV2Phonemizer::synthetic_for_test`'s JA char map (see `g2p.rs`).
/// Using ASCII text with `language="ja"` (e.g. `"test"` — the pre-WP-14
/// input) would silently map every char to the default phoneme id under
/// the old Lenient behavior, and reject loudly under the new
/// [`OovPolicy::Strict`] default (WP-14, FR-EX-08). Neither is what this
/// "adapter equivalence" test wants to check — it needs a JA input the
/// synthetic G2P actually covers.
#[test]
fn tts_engine_adapter_matches_direct_synthesize() {
    let model = SbV2Model::synthetic_for_test();

    let request = SynthesisRequest::new("あい")
        .with_language("ja")
        .deterministic();
    let via_trait =
        TtsEngine::synthesize(&model, &request).expect("trait synthesize should succeed");

    let direct_req = SbV2SynthRequest {
        text: "あい".to_string(),
        language: Language::JA,
        speaker_id: 0,
        speaker_embedding: None, // Blocker 3: legacy synthetic lookup path
        style_vec: vec![0.0; 4], // matches synthetic_for_test's d_style (4)
        speed: 1.0,
        noise_scale: 0.0,
        noise_scale_w: 0.0,
        seed: 0,
        // Must match `TtsEngine::synthesize`'s adapter default
        // (`RngMode::default()` = torch parity) exactly — otherwise the
        // byte-equality assertion below diverges purely on RNG choice.
        rng_mode: RngMode::default(),
    };
    let direct = model
        .synthesize(&direct_req)
        .expect("direct synthesize should succeed");

    assert_eq!(via_trait.samples, direct.samples);
    assert_eq!(via_trait.sample_rate, direct.sample_rate);
}

// ---------------------------------------------------------------------------
// WP-19: ZH BERT branch wiring
// ---------------------------------------------------------------------------
//
// The three tests below exercise the three arms of the ZH BERT dispatch
// table added by WP-19:
//
// | model                                    | request.language | expected             |
// |------------------------------------------|------------------|----------------------|
// | `synthetic_for_test` (no ZH)             | `ZH` via fixture | loud `NotImplemented` (BERT arm)  |
// | `synthetic_with_zh_for_test` (with ZH)   | `ZH` via fixture | non-empty PCM        |
// | `synthetic_with_zh_for_test` (with ZH)   | `JA` (default)   | non-empty PCM (backward compat) |
//
// All three go through the same `synthesize` body — the ZH branch must not
// disturb the JA/EN paths, and the fail-closed default (no ZH wired ==
// `NotImplemented`, not a silent JA/EN fallback) must survive FR-EX-08.
//
// The G2P side is bypassed via `PhonemizeFixture` (Task 7) for ZH: the
// production ZH G2P is a piper-plus-side delegation (WP-18), not something
// this crate can exercise without a full 8-language integration crate wired
// in — so the fixture supplies pre-computed phoneme ids matching what a
// real dumper would produce.

/// Build a small `PhonemizeFixture` that maps the given `(language,
/// text)` entries to fixed-length id sequences compatible with
/// `synthetic_for_test`'s `N_VOCAB = 256` / `N_TONES = 3` shape. Every id
/// is ≤ 3 so any `N_VOCAB ≥ 4` phoneme table clears them, and every tone
/// is 0 so any `N_TONES ≥ 1` tone table clears them.
///
/// `bert_input_text` is the same as `text` so the BERT tokenizer runs on
/// the caller-visible input, matching the JA/EN convention (see
/// `PhonemizeResult`'s field doc).
///
/// A single fixture serving multiple languages is legal per
/// `PhonemizeFixture::insert`'s `(Language, String)` map key — used below
/// so a ZH-wired model built with a fixture-backed phonemizer can still
/// serve JA (the fixture holds both language paths, dispatched by
/// `phonemize`'s existing `(language, text)` lookup).
fn multi_lang_fixture(entries: &[(Language, &str)]) -> PhonemizeFixture {
    let mut f = PhonemizeFixture::new();
    for (lang, text) in entries {
        f.insert(
            *lang,
            (*text).to_string(),
            PhonemizeResult {
                phoneme_ids: vec![1, 2, 3, 2],
                tones: vec![0, 0, 0, 0],
                word_boundaries: vec![true, false, false, false],
                bert_input_text: (*text).to_string(),
            },
        );
    }
    f
}

/// ZH-only convenience wrapper for the RED-1 test (the common single-
/// language shape). Delegates to [`multi_lang_fixture`].
fn zh_fixture_for(text: &str) -> PhonemizeFixture {
    multi_lang_fixture(&[(Language::ZH, text)])
}

/// WP-19 RED-1 → GREEN: a model with a wired ZH BERT branch synthesizes
/// non-empty PCM for a ZH request. Fails on any pre-WP-19 build because
/// `SbV2Model::synthetic_with_zh_for_test` does not exist.
#[test]
fn synthesize_zh_with_wired_zh_bert_returns_non_empty_pcm() {
    let model = SbV2Model::synthetic_with_zh_for_test(zh_fixture_for("你好"));
    let req = SbV2SynthRequest {
        text: "你好".to_string(),
        language: Language::ZH,
        speaker_id: 0,
        speaker_embedding: None,
        style_vec: vec![0.0; 4], // matches synthetic_for_test's d_style (4)
        speed: 1.0,
        noise_scale: 0.0,
        noise_scale_w: 0.0,
        seed: 42,
        // noise_scale_w = 0.0 short-circuits the RNG — legacy for symmetry
        // with the JA/EN synthetic tests above.
        rng_mode: RngMode::GaussianSplitMix64Legacy,
    };
    let audio = model
        .synthesize(&req)
        .expect("synthesize should succeed on a model with ZH BERT wired");
    assert!(!audio.samples.is_empty(), "ZH PCM output must be non-empty");
    assert_eq!(audio.sample_rate, 44_100);
}

/// WP-19 RED-2 → GREEN: FR-EX-08 fail-closed. A model **without** a wired
/// ZH BERT branch (i.e. plain `synthetic_for_test`) that receives a ZH
/// request — pushed past the G2P step via a `PhonemizeFixture` — must
/// return a loud `NotImplemented` from the BERT tokenizer arm rather than
/// silently falling through to the JA/EN encoder or panicking.
#[test]
fn synthesize_zh_without_wired_zh_bert_fails_loudly() {
    // `synthetic_for_test` has no ZH tokenizer + no ZH encoder. Rebuild it
    // with a `from_fixture` phonemizer so the G2P step succeeds for ZH
    // and control reaches the BERT dispatch arm this test is targeting.
    let model = SbV2Model::synthetic_for_test_with_phonemizer(
        vokra_models::sbv2::SbV2Phonemizer::from_fixture(zh_fixture_for("你好")),
    );

    let req = SbV2SynthRequest {
        text: "你好".to_string(),
        language: Language::ZH,
        speaker_id: 0,
        speaker_embedding: None,
        style_vec: vec![0.0; 4],
        speed: 1.0,
        noise_scale: 0.0,
        noise_scale_w: 0.0,
        seed: 42,
        rng_mode: RngMode::GaussianSplitMix64Legacy,
    };

    match model.synthesize(&req) {
        Ok(_) => panic!(
            "a ZH request against a model with no ZH BERT wired must fail loudly, \
             never fall through to JA/EN (FR-EX-08)"
        ),
        Err(VokraError::NotImplemented(msg)) => {
            assert!(
                msg.contains("ZH") || msg.to_lowercase().contains("zh"),
                "error message should identify ZH as the unimplemented path; got: {msg}"
            );
        }
        Err(other) => panic!("expected VokraError::NotImplemented(ZH), got: {other:?}"),
    }
}

/// WP-19 RED-3 → GREEN: the ZH-wired model must not disturb JA/EN. Loading
/// a ZH BERT into `SbV2BertContainer`'s new `Option` fields leaves the
/// JA/EN dispatch arms untouched, so a JA request against a ZH-wired
/// model produces non-empty PCM through the JA BERT path (unchanged from
/// pre-WP-19), matching the JA/EN test-shape convention above.
///
/// The fixture below holds a JA entry too so `SbV2Phonemizer`'s
/// fixture-arm dispatch reaches JA — the fixture is the only phonemizer
/// path this synthetic `SbV2Model` exposes (`synthetic_with_zh_for_test`
/// replaces the default synthetic char-map phonemizer with a fixture-
/// backed one, mirroring the production wiring pattern in
/// `parity_sbv2_real.rs`).
#[test]
fn synthesize_ja_on_zh_wired_model_still_works() {
    let fixture = multi_lang_fixture(&[(Language::ZH, "你好"), (Language::JA, "あいう")]);
    let model = SbV2Model::synthetic_with_zh_for_test(fixture);
    let req = SbV2SynthRequest {
        text: "あいう".to_string(),
        language: Language::JA,
        speaker_id: 0,
        speaker_embedding: None,
        style_vec: vec![0.0; 4],
        speed: 1.0,
        noise_scale: 0.0,
        noise_scale_w: 0.0,
        seed: 42,
        rng_mode: RngMode::GaussianSplitMix64Legacy,
    };
    let audio = model
        .synthesize(&req)
        .expect("JA synth on a ZH-wired model should still succeed");
    assert!(!audio.samples.is_empty(), "JA PCM output must be non-empty");
    assert_eq!(audio.sample_rate, 44_100);
}
