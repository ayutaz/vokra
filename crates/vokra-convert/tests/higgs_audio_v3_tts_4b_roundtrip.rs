//! Higgs-Audio v3 TTS 4B converter integration test
//! (coverage-audit-2026-08-03 Wave B fast-track, post-audit 2026-08-13).
//!
//! Exercises the public [`convert_file`] entry point through the
//! [`ModelKind::HiggsAudioV3Tts4b`] dispatch arm (mirror of the
//! `magpietts_v2602` / `firered_asr_aed_l` / `hibiki` / `frcrn` /
//! `nkf_aec` / `whisper` / `emotion2vec` roundtrip pattern). A
//! synthetic safetensors buffer with a mix of F32 / F16 / BF16 tensors
//! is written to disk, converted via the public API, and the resulting
//! GGUF is loaded back with the runtime loader — every tensor's
//! dtype + payload must survive the pipeline byte-identical (the pass-
//! through contract) and the provenance / category / upstream_hf
//! stamps must land on the artifact so the publish pipeline can gate
//! on the GGUF alone (no side-car lookup).
//!
//! Real-weight parity with the upstream Higgs-Audio v3 reference is
//! deferred to owner (`docs/license-audit.md §3.1` sign-off; ~8 GB
//! weights fetched on vast.ai per memory
//! `[[feedback-large-models-on-vast-ai]]`). This test locks the
//! byte-parallel GGUF surface so a future
//! `HiggsAudioV3Tts4bWeights::from_gguf` can bind against a stable
//! schema.

use std::path::PathBuf;

use vokra_convert::{
    HiggsAudioV3Tts4bReport, ModelKind, convert_file, convert_file_licensed,
    convert_higgs_audio_v3_tts_4b_file,
};
use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufFile, chunks};

/// A unique temp path for this test process (mirror of
/// `roundtrip.rs::tmp_path` — no external `tempfile` dep, preserving
/// zero-dep NFR-DS-02).
fn tmp_path(tag: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "vokra-convert-higgs-audio-v3-tts-4b-it-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_default(),
    ));
    p
}

/// Builds a three-tensor safetensors buffer covering the whole
/// pass-through matrix (F32, F16, BF16 — the three dtypes the
/// converter accepts). Layout mirrors the magpietts_v2602 / hibiki /
/// emotion2vec / wespeaker / frcrn module-level fixtures.
fn synthetic_higgs_audio_v3_tts_4b_safetensors() -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
    // F32 payload: 6 non-zero values so a silent widen would flip a
    // fence instead of trivially round-tripping a zero buffer.
    let f32_vals: [f32; 6] = [1.0, -2.0, 3.5, -0.25, 100.0, 0.001];
    let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(f32_bytes.len(), 24);
    // F16 payload: 4 half-floats with known non-zero bit patterns.
    let f16_patterns: [u16; 4] = [0x3C00, 0xC000, 0x4200, 0x0001];
    let f16_bytes: Vec<u8> = f16_patterns.iter().flat_map(|p| p.to_le_bytes()).collect();
    assert_eq!(f16_bytes.len(), 8);
    // BF16 payload: 6 non-zero values compressed into bf16 (top 16 bits
    // of the f32 bit pattern) — same construction as the module-level
    // fixture so the assertion below can byte-compare.
    let bf16_vals: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
    let bf16_bytes: Vec<u8> = bf16_vals
        .iter()
        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
        .collect();
    assert_eq!(bf16_bytes.len(), 12);

    // Realistic-looking Higgs-Audio v3 TTS 4B tensor names — LM
    // decoder / speaker encoder / audio codec topology. Names are
    // deliberately from a plausible upstream state-dict layout so a
    // future `HiggsAudioV3Tts4bWeights::from_gguf` can walk the same
    // manifest.
    let f32_off_end = f32_bytes.len();
    let f16_off_end = f32_off_end + f16_bytes.len();
    let bf16_off_end = f16_off_end + bf16_bytes.len();
    let header = format!(
        r#"{{"decoder.model.embed_tokens.weight":{{"dtype":"F32","shape":[2,3],"data_offsets":[0,{f32_off_end}]}},"speaker_encoder.linear.bias":{{"dtype":"F16","shape":[4],"data_offsets":[{f32_off_end},{f16_off_end}]}},"audio_codec.decoder.layers.0.self_attn.q_proj.weight":{{"dtype":"BF16","shape":[2,3],"data_offsets":[{f16_off_end},{bf16_off_end}]}}}}"#
    );
    let mut buf = Vec::new();
    buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
    buf.extend_from_slice(header.as_bytes());
    buf.extend_from_slice(&f32_bytes);
    buf.extend_from_slice(&f16_bytes);
    buf.extend_from_slice(&bf16_bytes);
    (buf, f32_bytes, f16_bytes, bf16_bytes)
}

/// End-to-end integration through the public [`convert_file`] surface
/// (the `--model higgs-audio-v3-tts-4b` CLI path): every dtype survives
/// the pipeline byte-identical, tensor names / shapes round-trip, and
/// the provenance / category / upstream stamps land on the artifact.
///
/// This is the "wrong-arm dispatch" fence — if a future refactor wired
/// `ModelKind::HiggsAudioV3Tts4b` to the wrong `convert_*_file`
/// function (e.g. magpietts_v2602 by copy-paste) the arch string would
/// come out as `magpietts_v2602` and this assertion would fire loudly.
#[test]
fn higgs_audio_v3_tts_4b_safetensors_roundtrips_through_convert_file() {
    let (input_bytes, f32_payload, f16_payload, bf16_payload) =
        synthetic_higgs_audio_v3_tts_4b_safetensors();
    let input = tmp_path("higgs-in");
    let output = tmp_path("higgs-out");
    std::fs::write(&input, &input_bytes).expect("write input");

    let summary = convert_file(ModelKind::HiggsAudioV3Tts4b, &input, &output).expect("convert");
    assert_eq!(summary.model, ModelKind::HiggsAudioV3Tts4b);
    assert_eq!(summary.tensor_count, 3, "3 float tensors written");
    assert!(
        summary.output_bytes > 0,
        "output GGUF must have non-empty size"
    );
    assert_eq!(summary.notes.len(), 1, "single summary note emitted");
    assert!(
        summary.notes[0].contains("higgs-audio-v3-tts-4b: 3 float weights"),
        "summary must mention the higgs-audio-v3-tts-4b count: {}",
        summary.notes[0]
    );
    assert!(
        summary.notes[0].contains("1 BF16 passthrough"),
        "summary must call out the BF16 pass-through subset: {}",
        summary.notes[0]
    );

    let file = GgufFile::open(&output).expect("load output gguf");
    assert_eq!(file.tensors().len(), 3, "GGUF has 3 tensors");

    // F32 tensor: dtype + shape + payload byte-identical.
    let f32_info = file
        .tensor_info("decoder.model.embed_tokens.weight")
        .expect("F32 tensor present");
    assert_eq!(f32_info.dtype, GgmlType::F32, "F32 stays F32 (no widen)");
    assert_eq!(f32_info.dimensions, vec![2, 3]);
    assert_eq!(file.tensor_bytes(f32_info), f32_payload.as_slice());

    // F16 tensor: dtype + shape + payload byte-identical.
    let f16_info = file
        .tensor_info("speaker_encoder.linear.bias")
        .expect("F16 tensor present");
    assert_eq!(f16_info.dtype, GgmlType::F16, "F16 stays F16 (no widen)");
    assert_eq!(f16_info.dimensions, vec![4]);
    assert_eq!(file.tensor_bytes(f16_info), f16_payload.as_slice());

    // BF16 tensor: the pass-through pin (a silent widen to F32 would
    // change the on-disk dtype tag AND balloon the payload from 12 B
    // → 24 B, so this assertion double-locks the invariant).
    let bf16_info = file
        .tensor_info("audio_codec.decoder.layers.0.self_attn.q_proj.weight")
        .expect("BF16 tensor present after pass-through");
    assert_eq!(
        bf16_info.dtype,
        GgmlType::BF16,
        "BF16 stays BF16 (GGUF type 30, no convert-time widening)"
    );
    assert_eq!(bf16_info.dimensions, vec![2, 3]);
    assert_eq!(
        file.tensor_bytes(bf16_info),
        bf16_payload.as_slice(),
        "BF16 payload must be byte-identical to input"
    );

    // Provenance + category chunks pinned on the artifact itself.
    assert_eq!(
        file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
        Some("higgs_audio_v3_tts_4b"),
        "arch stamp distinct from every sibling TTS entry — silent alias would misroute"
    );
    assert_eq!(
        file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
        Some("higgs-audio-v3-tts-4b")
    );
    assert_eq!(
        file.get("vokra.model.category").and_then(|v| v.as_str()),
        Some("tts"),
        "category groups Higgs-Audio v3 with the TTS family for the zoo manifest"
    );
    assert_eq!(
        file.get("vokra.provenance.upstream_hf")
            .and_then(|v| v.as_str()),
        Some("bosonai/higgs-audio-v3-tts-4b"),
        "upstream slug pins traceability back to the BosonAI HF release"
    );
    // The audit-ticket default really was `apache-2.0`, and this assertion
    // pinned it. Primary-source verification (3baf317, 2026-08-14) found the
    // actual LICENSE is "BOSON HIGGS TTS 3 RESEARCH AND NON-COMMERCIAL
    // LICENSE AGREEMENT", whose §II-A(c) bans redistribution, hosting and
    // embedding. The converter was corrected; this test was not, so it went
    // on asserting that a redistribution-forbidden model is Permissive —
    // the one direction a stale licence claim must never fail in.
    assert_eq!(
        file.get(chunks::KEY_PROVENANCE_LICENSE)
            .and_then(|v| v.as_str()),
        Some("LicenseRef-Boson-Higgs-TTS-3-Research-Non-Commercial"),
        "licence is the bespoke BOSON HIGGS TTS 3 R&NC agreement, not apache-2.0"
    );
    assert_eq!(
        file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
            .and_then(|v| v.as_str()),
        Some(LicenseClass::RedistributionForbidden.as_str()),
        "R&NC §II-A(c) forbids redistribution — publishing must stay gated"
    );
    assert!(
        file.get(chunks::KEY_SCHEMA_VERSION).is_some(),
        "vokra.schema.version must be stamped"
    );
    assert!(
        file.get(chunks::KEY_SCHEMA_PRODUCER).is_some(),
        "vokra.schema.producer must be stamped"
    );

    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&output);
}

/// Direct `convert_higgs_audio_v3_tts_4b_file` entry-point exercise:
/// the report counters must add up (`read == written + skipped_non_float`)
/// and the subset counters agree with the pass-through matrix — a
/// regression where the F16 arm silently reclassified BF16 as F16
/// would flip both counters, so this asserts them independently.
#[test]
fn higgs_audio_v3_tts_4b_direct_entry_point_returns_matching_report() {
    let (input_bytes, _, _, _) = synthetic_higgs_audio_v3_tts_4b_safetensors();
    let input = tmp_path("higgs-direct-in");
    let output = tmp_path("higgs-direct-out");
    std::fs::write(&input, &input_bytes).expect("write input");

    let report: HiggsAudioV3Tts4bReport = convert_higgs_audio_v3_tts_4b_file(&input, &output, None)
        .expect("convert_higgs_audio_v3_tts_4b_file");

    assert_eq!(
        report.read, 3,
        "3 tensors observed in the safetensors header"
    );
    assert_eq!(report.written, 3, "all 3 must reach the pass-through arm");
    assert_eq!(
        report.skipped_non_float, 0,
        "no synthetic tensor is non-float — the skip counter must stay 0"
    );
    assert_eq!(
        report.bf16_passthrough, 1,
        "exactly one BF16 tensor in the fixture — the counter must \
         match, not silently reclassify BF16 as F16"
    );

    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&output);
}

/// The `convert_file_licensed` boundary threads `Some(spdx)` through
/// to the same converter body — pins that the license override does
/// land on the artifact via the outer dispatch (not just via the
/// direct entry point). Mirrors the `hibiki_roundtrip` license
/// override test.
#[test]
fn higgs_audio_v3_tts_4b_license_override_threads_through_convert_file_licensed() {
    let (input_bytes, _, _, _) = synthetic_higgs_audio_v3_tts_4b_safetensors();
    let input = tmp_path("higgs-override-in");
    let output = tmp_path("higgs-override-out");
    std::fs::write(&input, &input_bytes).expect("write input");

    let summary = convert_file_licensed(
        ModelKind::HiggsAudioV3Tts4b,
        &input,
        &output,
        Some("apache-2.0"),
    )
    .expect("convert_file_licensed with explicit apache-2.0 override");
    assert_eq!(summary.model, ModelKind::HiggsAudioV3Tts4b);

    // This exercises the `--license` override PLUMBING, not a claim about
    // Higgs-Audio: the override exists for "implementation is clean-room but
    // the upstream checkpoint carries something else". `apache-2.0` is the
    // arbitrary probe value. Higgs-Audio's real licence is the R&NC agreement
    // asserted in the roundtrip test above; do not read this as a second,
    // contradictory answer.
    let file = GgufFile::open(&output).expect("load output gguf");
    assert_eq!(
        file.get(chunks::KEY_PROVENANCE_LICENSE)
            .and_then(|v| v.as_str()),
        Some("apache-2.0"),
        "explicit `Some(\"apache-2.0\")` override still lands the SPDX string"
    );
    assert_eq!(
        file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
            .and_then(|v| v.as_str()),
        Some(LicenseClass::Permissive.as_str()),
        "apache-2.0 maps to Permissive on the LicenseClass re-derivation"
    );

    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&output);
}
