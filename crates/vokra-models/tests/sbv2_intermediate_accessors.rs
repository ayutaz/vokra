//! Wave-4 INTERMEDIATE-ACCESSORS: SbV2Model exposes per-stage
//! intermediate tensors alongside the final PCM.
//!
//! Audit rank 16 (major): parity_sbv2_real can only compare waveform
//! via ATOL_DEFAULT because SbV2Model exposes only synthesize(). Without
//! per-stage accessors, every bug-hunt has to instrument via ad-hoc env
//! vars (like the discontinued VOKRA_SBV2_SDP_HIDDEN_OVERRIDE). Sibling
//! parity tests (parity_kokoro, parity_whisper) already have per-tensor
//! tables. This test proves the accessor exists and returns every
//! intermediate the manifest schema names.

use vokra_models::sbv2::{Language, RngMode, SbV2Model, SbV2SynthRequest};

/// Every field of SbV2Intermediates is populated on a JA request; the
/// EN bucket is empty (per per-language convention). The final PCM
/// matches what synthesize() alone returns byte-for-byte — the
/// accessor must not perturb the pipeline output.
#[test]
fn synthesize_with_intermediates_populates_all_ja_fields() {
    let model = SbV2Model::synthetic_for_test();
    let req = SbV2SynthRequest {
        text: "あいう".to_string(),
        language: Language::JA,
        speaker_id: 0,
        speaker_embedding: None,
        style_vec: vec![0.0; 4], // matches synthetic_for_test's d_style (4)
        speed: 1.0,
        noise_scale: 0.0,
        noise_scale_w: 0.0,
        seed: 42,
        rng_mode: RngMode::GaussianSplitMix64Legacy,
    };

    let (audio, inter) = model
        .synthesize_with_intermediates(&req)
        .expect("synthesize_with_intermediates should succeed");

    // Every JA-side intermediate field must be non-empty.
    assert!(
        !inter.phoneme_embed.is_empty(),
        "phoneme_embed must be populated"
    );
    assert!(
        !inter.text_hidden.is_empty(),
        "text_hidden must be populated"
    );
    assert!(
        !inter.bert_hidden_ja.is_empty(),
        "bert_hidden_ja must be populated on a JA request"
    );
    assert!(
        inter.bert_hidden_en.is_empty(),
        "bert_hidden_en must be EMPTY on a JA request (per-language convention)"
    );
    assert!(
        !inter.bert_bridge_out.is_empty(),
        "bert_bridge_out must be populated"
    );
    assert!(
        !inter.speaker_embed.is_empty(),
        "speaker_embed must be populated"
    );
    assert!(
        !inter.style_projected.is_empty(),
        "style_projected must be populated"
    );
    assert!(!inter.sdp_sample.is_empty(), "sdp_sample must be populated");
    assert!(!inter.mel_hidden.is_empty(), "mel_hidden must be populated");
    assert!(!inter.z_latent.is_empty(), "z_latent must be populated");

    // phoneme_embed and text_hidden must have identical shape (both
    // [T_text, d_model], the transformer stack does not change shape).
    assert_eq!(
        inter.phoneme_embed.len(),
        inter.text_hidden.len(),
        "phoneme_embed and text_hidden must share shape [T_text, d_model]"
    );
    // bert_bridge_out is text_hidden + bridge; same shape.
    assert_eq!(
        inter.bert_bridge_out.len(),
        inter.text_hidden.len(),
        "bert_bridge_out and text_hidden must share shape"
    );

    // Cross-check: PCM matches what synthesize() alone returns.
    let audio_only = model.synthesize(&req).expect("synthesize should succeed");
    assert_eq!(
        audio.samples, audio_only.samples,
        "synthesize_with_intermediates must return byte-identical PCM to synthesize()"
    );
    assert_eq!(audio.sample_rate, audio_only.sample_rate);
}

/// EN request: bert_hidden_en populated, bert_hidden_ja empty (mirror).
#[test]
fn synthesize_with_intermediates_populates_en_bucket_only_on_en_request() {
    let model = SbV2Model::synthetic_for_test();
    let req = SbV2SynthRequest {
        text: "test".to_string(),
        language: Language::EN,
        speaker_id: 0,
        speaker_embedding: None,
        style_vec: vec![0.0; 4],
        speed: 1.0,
        noise_scale: 0.0,
        noise_scale_w: 0.0,
        seed: 42,
        rng_mode: RngMode::GaussianSplitMix64Legacy,
    };

    let (_audio, inter) = model
        .synthesize_with_intermediates(&req)
        .expect("synthesize_with_intermediates should succeed");

    assert!(
        !inter.bert_hidden_en.is_empty(),
        "bert_hidden_en must be populated on an EN request"
    );
    assert!(
        inter.bert_hidden_ja.is_empty(),
        "bert_hidden_ja must be EMPTY on an EN request (per-language convention)"
    );
}

/// `to_dumper_map` returns the manifest-order tensor list, skipping the
/// inactive BERT bucket. Every entry name matches the Python dumper's
/// `reference_dump/<name>.bin` filename convention.
#[test]
fn to_dumper_map_lists_all_active_tensors_in_manifest_order() {
    let model = SbV2Model::synthetic_for_test();
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

    let (_audio, inter) = model
        .synthesize_with_intermediates(&req)
        .expect("synthesize_with_intermediates should succeed");

    let map = inter.to_dumper_map();
    let names: Vec<&str> = map.iter().map(|(n, _)| *n).collect();

    // JA request: bert_hidden_en is skipped; every other manifest tensor
    // is present in the dumper's own order (design doc §10).
    assert_eq!(
        names,
        vec![
            "phoneme_embed",
            "text_hidden",
            "bert_hidden_ja",
            "bert_bridge_out",
            "speaker_embed",
            "style_projected",
            "sdp_sample",
            "mel_hidden",
            "z_latent",
        ],
        "to_dumper_map must emit every JA-active tensor in manifest order"
    );

    // Every payload is non-empty and its length is a multiple of 4 (f32
    // byte size — the Python dumper's `arr.tobytes()` invariant).
    for (name, bytes) in &map {
        assert!(!bytes.is_empty(), "{name} payload must be non-empty");
        assert_eq!(
            bytes.len() % 4,
            0,
            "{name} payload must be a multiple of 4 bytes (f32)"
        );
    }
}
