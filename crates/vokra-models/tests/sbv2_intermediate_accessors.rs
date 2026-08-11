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

use std::path::{Path, PathBuf};

use vokra_core::json::{self, JsonValue};
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

/// Repo-root-relative fixture directory (mirror of
/// [`parity_sbv2_real::fixtures_dir`]).
fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("fixtures")
        .join("sbv2")
}

/// Loads every tensor name listed in the committed
/// `reference_dump.manifest.json`.
fn manifest_tensor_names() -> Vec<String> {
    let manifest_path = fixtures_dir().join("reference_dump.manifest.json");
    let bytes = std::fs::read(&manifest_path)
        .unwrap_or_else(|e| panic!("{}: cannot read manifest: {e}", manifest_path.display()));
    let manifest = json::parse(&bytes)
        .unwrap_or_else(|e| panic!("{}: JSON parse error: {e}", manifest_path.display()));
    manifest
        .get("tensors")
        .and_then(JsonValue::as_array)
        .unwrap_or_else(|| panic!("{}: `tensors` missing/not-array", manifest_path.display()))
        .iter()
        .map(|t| {
            t.get("name")
                .and_then(JsonValue::as_str)
                .unwrap_or_else(|| panic!("tensors[] entry missing name"))
                .to_string()
        })
        .collect()
}

/// WP-01 (2026-08-09): every name emitted by [`to_dumper_map`] MUST have
/// a matching manifest `tensors[]` entry so
/// `parity_sbv2_real::diff_intermediates_against_manifest`'s
/// `find_tensor` lookup succeeds for every dumper-map row. Fires without
/// the real fixtures — the manifest JSON alone is a committed schema
/// anchor. Fail-closed rationale: without this pin, a rename of a
/// dumper-map key or a manifest key would surface only when the
/// `#[ignore]`d harness ran with the real fixture bundle — much later
/// than the drift was introduced.
#[test]
fn every_dumper_map_name_is_present_in_manifest() {
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

    let manifest = manifest_tensor_names();
    for (dumper_name, _bytes) in inter.to_dumper_map() {
        assert!(
            manifest.iter().any(|m| m == dumper_name),
            "to_dumper_map emits `{dumper_name}` but the committed manifest \
             does not list it — either rename the map arm to match the \
             manifest, or add `{dumper_name}` to \
             `tests/fixtures/sbv2/reference_dump.manifest.json`"
        );
    }
}

/// WP-01 (2026-08-09): every manifest tensor (except `waveform`, which is
/// returned via [`SynthesizedAudio::samples`] rather than the intermediate
/// map) MUST be emitted by [`to_dumper_map`] on the appropriate
/// per-language request. This test picks JA — so the ONE manifest tensor
/// that legitimately does NOT appear on a JA request is `bert_hidden_en`;
/// everything else is expected.
///
/// A future new manifest tensor added to §10 of the design doc that
/// forgets to grow [`SbV2Intermediates`] + [`to_dumper_map`] trips this
/// assertion at plain `cargo test` time, not `--ignored`-only.
#[test]
fn every_ja_active_manifest_tensor_is_emitted_by_dumper_map() {
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
    let dumper_names: Vec<&str> = inter.to_dumper_map().iter().map(|(n, _)| *n).collect();
    for manifest_name in manifest_tensor_names() {
        // `waveform` is returned via `SynthesizedAudio::samples`, not
        // via the intermediate dumper map — the parity harness has a
        // dedicated tolerance-based length/RMS gate for it.
        if manifest_name == "waveform" {
            continue;
        }
        // `bert_hidden_en` is legitimately skipped on a JA request per
        // `to_dumper_map`'s per-language convention (see that method's
        // doc); an EN request would have `bert_hidden_ja` skipped
        // symmetrically. This test picks JA, so filter EN out here.
        if manifest_name == "bert_hidden_en" {
            continue;
        }
        assert!(
            dumper_names.iter().any(|d| *d == manifest_name),
            "manifest tensor `{manifest_name}` is not emitted by \
             `SbV2Intermediates::to_dumper_map` on a JA request — either \
             extend `SbV2Intermediates` + `to_dumper_map` to carry it, or \
             adjust this test's skip list if the tensor is a documented \
             out-of-map slot (like `waveform`)"
        );
    }
}
