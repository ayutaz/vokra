//! Blocker 3 (external 512-d speaker input path) integration tests.
//!
//! The real Style-Bert-VITS2 v2 base checkpoint has **no per-speaker
//! embedding table** (`emb_g`) — speaker conditioning enters as an
//! external `[d_speaker=512]` vector that
//! `enc_p.encoder.spk_emb_linear.{weight,bias}` projects to `[d_model=192]`
//! before the text-encoder broadcast add (scout report §1). These tests
//! exercise the input-plumbing side of that path end-to-end through
//! `SbV2Model::synthesize` and its `TtsEngine` adapter, using a
//! `SbV2Model::synthetic_for_test`-shaped model with a synthetic
//! [`ExternalSpeakerProjection`] attached via
//! [`SbV2Model::with_external_speaker_projection`].
//!
//! **Regression contract**: the pre-Blocker-3 synthetic path (no
//! `speaker_embedding` on the request, no projection on the model —
//! `SbV2Model::synthetic_for_test`'s exact 12-sample output) is
//! independently pinned by `tests/parity_sbv2_synthetic.rs`'s
//! `synthetic_shape_invariants_hold`; the tests here focus on the new
//! external-input surface and the loud-error cases FR-EX-08 requires.

use vokra_core::{SynthesisRequest, TtsEngine, VokraError};
use vokra_models::sbv2::{ExternalSpeakerProjection, Language, SbV2Model, SbV2SynthRequest};

/// Deterministic synthetic external speaker projection sized to
/// `synthetic_for_test`'s tiny dims — d_model=8 (see that fn's doc),
/// d_speaker chosen as 6 so a wrong-length request check has a clear
/// off-by-one target and the input/output widths differ (matching the
/// real ckpt's `d_speaker != d_model` shape, not the synthetic
/// `d_speaker == d_model` legacy shape). The tiny magnitudes mirror the
/// same "smooth-sinusoidal, bounded, nonzero" convention every other
/// synthetic weight in this crate uses.
fn synthetic_external_projection(d_speaker: usize, d_model: usize) -> ExternalSpeakerProjection {
    let weight = (0..d_model * d_speaker)
        .map(|i| ((i as f32) * 0.037).sin() * 0.05)
        .collect();
    let bias = (0..d_model)
        .map(|i| ((i as f32) * 0.11).cos() * 0.01)
        .collect();
    ExternalSpeakerProjection::from_weights(weight, bias, d_speaker, d_model)
}

fn base_request(text: &str, language: Language) -> SbV2SynthRequest {
    SbV2SynthRequest {
        text: text.to_string(),
        language,
        speaker_id: 0,
        speaker_embedding: None,
        style_vec: vec![0.0; 4], // matches synthetic_for_test's d_style (4)
        speed: 1.0,
        noise_scale: 0.0,
        noise_scale_w: 0.0,
        seed: 42,
    }
}

/// Model with an external projection attached succeeds when the request
/// carries an external `[d_speaker]` embedding — the projection maps
/// `[d_speaker] -> [d_model]` and the projected result broadcast-adds
/// into `hidden` in place of the lookup-path contribution.
#[test]
fn synthesize_with_external_speaker_embedding_succeeds() {
    const D_SPEAKER: usize = 6; // != d_model=8, exercises non-square shape
    const D_MODEL: usize = 8;
    let proj = synthetic_external_projection(D_SPEAKER, D_MODEL);
    let model = SbV2Model::synthetic_for_test().with_external_speaker_projection(proj);

    let mut req = base_request("あいう", Language::JA);
    req.speaker_embedding = Some((0..D_SPEAKER).map(|i| (i as f32) * 0.1).collect());

    let audio = model
        .synthesize(&req)
        .expect("external embedding path must succeed");
    assert!(!audio.samples.is_empty(), "PCM output must be non-empty");
    assert!(
        audio.samples.iter().all(|s| s.is_finite()),
        "PCM output must be all-finite"
    );
    assert_eq!(audio.sample_rate, 44_100);
}

/// A request with `speaker_embedding: None` **and** a model that carries
/// an external projection uses the deterministic all-zero external vector
/// (`SynthesisRequest::speaker_embedding`'s documented "None uses the
/// zero vector, the deterministic zero-shot default" contract). The
/// projection still applies its bias to that zero vector, so the audio
/// is deterministic — the bias-only contribution is not zero.
#[test]
fn synthesize_with_none_embedding_and_with_projection_uses_zero_default() {
    const D_SPEAKER: usize = 6;
    const D_MODEL: usize = 8;
    let proj = synthetic_external_projection(D_SPEAKER, D_MODEL);
    let model_a = SbV2Model::synthetic_for_test().with_external_speaker_projection(proj);
    let proj = synthetic_external_projection(D_SPEAKER, D_MODEL);
    let model_b = SbV2Model::synthetic_for_test().with_external_speaker_projection(proj);

    // Request with explicit zero external embedding — has to reproduce the
    // same audio as passing None (the documented default).
    let mut req_zeros = base_request("あいう", Language::JA);
    req_zeros.speaker_embedding = Some(vec![0.0_f32; D_SPEAKER]);

    let mut req_none = base_request("あいう", Language::JA);
    req_none.speaker_embedding = None;

    let audio_zeros = model_a
        .synthesize(&req_zeros)
        .expect("explicit-zero external must succeed");
    let audio_none = model_b
        .synthesize(&req_none)
        .expect("None external must succeed (zero-default path)");
    assert_eq!(
        audio_zeros.samples, audio_none.samples,
        "None (= zero-default) must reproduce the explicit all-zero embedding path bit-for-bit"
    );
}

/// A caller-supplied `speaker_embedding` whose length does not match the
/// projection's `d_in` fails loudly with [`VokraError::InvalidArgument`],
/// never a silent zero-pad / truncate / reshape (FR-EX-08).
#[test]
fn synthesize_with_wrong_length_embedding_is_invalid_argument() {
    const D_SPEAKER: usize = 6;
    const D_MODEL: usize = 8;
    let proj = synthetic_external_projection(D_SPEAKER, D_MODEL);
    let model = SbV2Model::synthetic_for_test().with_external_speaker_projection(proj);

    let mut req = base_request("あいう", Language::JA);
    req.speaker_embedding = Some(vec![0.0_f32; D_SPEAKER - 1]);
    match model.synthesize(&req) {
        Ok(_) => panic!("wrong-length embedding must not succeed silently"),
        Err(VokraError::InvalidArgument(msg)) => {
            assert!(
                msg.contains(&(D_SPEAKER - 1).to_string()),
                "error must name the actual length ({}), got: {msg}",
                D_SPEAKER - 1
            );
            assert!(
                msg.contains(&D_SPEAKER.to_string()),
                "error must name the expected length ({}), got: {msg}",
                D_SPEAKER
            );
        }
        Err(other) => panic!("expected VokraError::InvalidArgument, got {other:?}"),
    }
}

/// A caller-supplied `speaker_embedding` on a model that has **no**
/// external projection loaded (e.g. `SbV2Model::synthetic_for_test`
/// without `.with_external_speaker_projection`) fails loudly rather than
/// silently dropping the caller-supplied data. This is the mirror image
/// of the current SBV2 `TtsEngine` adapter's fail-loud gate — moved from
/// the trait boundary into the pipeline so both entry points (inherent
/// `synthesize` and `TtsEngine::synthesize`) surface the same error.
#[test]
fn synthesize_with_embedding_but_no_projection_is_invalid_argument() {
    let model = SbV2Model::synthetic_for_test(); // no `.with_external_speaker_projection`

    let mut req = base_request("あいう", Language::JA);
    req.speaker_embedding = Some(vec![0.0_f32; 512]);
    match model.synthesize(&req) {
        Ok(_) => panic!("external embedding without projection must not succeed silently"),
        Err(VokraError::InvalidArgument(_)) => { /* expected */ }
        Err(other) => panic!("expected VokraError::InvalidArgument, got {other:?}"),
    }
}

/// Backward-compat regression: a `speaker_embedding: None` request on a
/// model that has no external projection loaded still routes through the
/// pre-Blocker-3 `SpeakerEmbedding::lookup` path — this is what every
/// existing synthetic test (`tests/sbv2_model_synthetic.rs`,
/// `tests/sbv2_e2e_synthetic.rs`, `tests/parity_sbv2_synthetic.rs`)
/// depends on. This test asserts synthesis simply succeeds; the exact
/// output bytes are pinned separately by
/// `tests/parity_sbv2_synthetic.rs`.
#[test]
fn synthesize_without_embedding_and_no_projection_uses_lookup() {
    let model = SbV2Model::synthetic_for_test();

    let req = base_request("あいう", Language::JA);
    let audio = model
        .synthesize(&req)
        .expect("lookup-path synthesize (backward compat) must succeed");
    assert!(
        !audio.samples.is_empty(),
        "backward-compat lookup path must still produce PCM"
    );
}

/// `TtsEngine::synthesize` now **honors** a caller-supplied
/// `SynthesisRequest::speaker_embedding` instead of returning
/// `VokraError::InvalidArgument`. The adapter passes the embedding
/// through to the inherent `SbV2Model::synthesize`; a matching
/// `SbV2SynthRequest` built by hand must reproduce the exact same PCM.
#[test]
fn tts_engine_synthesize_honors_request_speaker_embedding() {
    const D_SPEAKER: usize = 6;
    const D_MODEL: usize = 8;
    let proj = synthetic_external_projection(D_SPEAKER, D_MODEL);
    let model_a = SbV2Model::synthetic_for_test().with_external_speaker_projection(proj);
    let proj = synthetic_external_projection(D_SPEAKER, D_MODEL);
    let model_b = SbV2Model::synthetic_for_test().with_external_speaker_projection(proj);

    let embedding: Vec<f32> = (0..D_SPEAKER).map(|i| (i as f32) * 0.1).collect();

    // Path 1: cross-engine `SynthesisRequest` (through the trait). Match
    // `SbV2Model`'s `TtsEngine::synthesize` deterministic mapping: JA is
    // the default language, `deterministic()` zeroes both noise scales,
    // seed=0 is the adapter's default (see that impl's doc).
    let request = SynthesisRequest::new("あいう")
        .with_language("ja")
        .with_speaker_embedding(embedding.clone())
        .deterministic();
    let via_trait =
        TtsEngine::synthesize(&model_a, &request).expect("trait synthesize must succeed");

    // Path 2: inherent `SbV2Model::synthesize` with a request that
    // mirrors the adapter's field mapping exactly.
    let direct_req = SbV2SynthRequest {
        text: "あいう".to_string(),
        language: Language::JA,
        speaker_id: 0,
        speaker_embedding: Some(embedding),
        style_vec: vec![0.0; 4], // matches synthetic_for_test's d_style
        speed: 1.0,
        noise_scale: 0.0,
        noise_scale_w: 0.0,
        seed: 0, // matches the adapter's default
    };
    let direct = model_b
        .synthesize(&direct_req)
        .expect("direct synthesize must succeed");

    assert_eq!(
        via_trait.samples, direct.samples,
        "TtsEngine::synthesize must reproduce the direct SbV2Model::synthesize output byte-for-byte"
    );
    assert_eq!(via_trait.sample_rate, direct.sample_rate);
}

/// `TtsEngine::synthesize` still rejects `prosody_features` loudly — SBV2
/// derives pitch-accent tones from its own G2P (see the `TtsEngine` impl
/// doc). Blocker 3 does not touch this gate.
#[test]
fn tts_engine_synthesize_still_rejects_prosody_features() {
    let model = SbV2Model::synthetic_for_test();
    let mut request = SynthesisRequest::new("あいう")
        .with_language("ja")
        .deterministic();
    request.prosody_features = Some(vec![[0, 0, 0]]);
    match TtsEngine::synthesize(&model, &request) {
        Ok(_) => panic!("prosody_features must still be rejected loudly"),
        Err(VokraError::InvalidArgument(msg)) => {
            assert!(
                msg.contains("prosody_features"),
                "error message should name the offending field, got: {msg}"
            );
        }
        Err(other) => panic!("expected VokraError::InvalidArgument, got {other:?}"),
    }
}
