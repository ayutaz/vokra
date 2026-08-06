//! Nemotron-Speech-Streaming-v2603 converter integration test
//! (coverage-audit 2026-08-03 Wave B).
//!
//! Exercises the public [`convert_file`] entry point through the
//! [`ModelKind::NemotronSpeechStreamingV2603`] dispatch arm (mirror of
//! the frcrn / rnnoise / emotion2vec / parakeet-tdt-1.1b roundtrip
//! pattern). A synthetic safetensors buffer with a mix of F32 / F16 /
//! BF16 tensors is written to disk, converted via the public API, and
//! the resulting GGUF is loaded back with the runtime loader — every
//! tensor's dtype + payload must survive the pipeline byte-identical
//! (the pass-through contract) and the provenance / category /
//! upstream_url stamps must land on the artifact so the publish
//! pipeline can gate on the GGUF alone (no side-car lookup).
//!
//! Real-weight parity with the upstream Nemotron-Speech-Streaming-v2603
//! `.nemo` release is deferred to owner (§3.1 sign-off + hparam
//! transcription). This test locks the byte-parallel GGUF surface so a
//! future `NemotronSpeechStreamingV2603Weights::from_gguf` can bind
//! against a stable schema.

use std::path::PathBuf;

use vokra_convert::{
    ModelKind, NemotronSpeechStreamingV2603Report, convert_file,
    convert_nemotron_speech_streaming_v2603_file,
};
use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufFile, chunks};

/// A unique temp path for this test process (mirror of the
/// `frcrn_roundtrip::tmp_path` — no external `tempfile` dep, preserving
/// zero-dep NFR-DS-02).
fn tmp_path(tag: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "vokra-convert-nemotron-speech-streaming-v2603-it-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_default(),
    ));
    p
}

/// Builds a three-tensor safetensors buffer covering the whole
/// pass-through matrix (F32, F16, BF16 — the three dtypes the converter
/// accepts). Layout mirrors the frcrn / rnnoise / parakeet-tdt-1.1b
/// module-level fixtures.
fn synthetic_nemotron_safetensors() -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
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

    // Realistic-looking Nemotron FastConformer encoder + streaming head
    // tensor names — the exact topology
    // (`encoder.blocks.N.attn.qkv_proj.weight` /
    // `encoder.blocks.0.mlp.fc1.weight` /
    // `decoder.output_layer.weight`) the future
    // `NemotronSpeechStreamingV2603Weights::from_gguf` walks.
    let f32_off_end = f32_bytes.len();
    let f16_off_end = f32_off_end + f16_bytes.len();
    let bf16_off_end = f16_off_end + bf16_bytes.len();
    let header = format!(
        r#"{{"encoder.blocks.0.mlp.fc1.weight":{{"dtype":"F32","shape":[2,3],"data_offsets":[0,{f32_off_end}]}},"decoder.output_layer.weight":{{"dtype":"F16","shape":[1,4],"data_offsets":[{f32_off_end},{f16_off_end}]}},"encoder.blocks.0.attn.qkv_proj.weight":{{"dtype":"BF16","shape":[2,3],"data_offsets":[{f16_off_end},{bf16_off_end}]}}}}"#
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
/// (the `--model nemotron-speech-streaming-v2603` CLI path): every
/// dtype survives the pipeline byte-identical, tensor names / shapes
/// round-trip, and the provenance / category / upstream stamps land on
/// the artifact.
#[test]
fn nemotron_safetensors_roundtrips_through_convert_file() {
    let (input_bytes, f32_payload, f16_payload, bf16_payload) = synthetic_nemotron_safetensors();
    let input = tmp_path("nemotron-in");
    let output = tmp_path("nemotron-out");
    std::fs::write(&input, &input_bytes).expect("write input");

    let summary =
        convert_file(ModelKind::NemotronSpeechStreamingV2603, &input, &output).expect("convert");
    assert_eq!(summary.model, ModelKind::NemotronSpeechStreamingV2603);
    assert_eq!(summary.tensor_count, 3, "3 float tensors written");
    assert!(
        summary.output_bytes > 0,
        "output GGUF must have non-empty size"
    );
    assert_eq!(summary.notes.len(), 1, "single summary note emitted");
    assert!(
        summary.notes[0].contains("nemotron-speech-streaming-v2603: 3 float weights"),
        "summary must mention the nemotron-speech-streaming-v2603 count: {}",
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
        .tensor_info("encoder.blocks.0.mlp.fc1.weight")
        .expect("F32 tensor present");
    assert_eq!(f32_info.dtype, GgmlType::F32, "F32 stays F32 (no widen)");
    assert_eq!(f32_info.dimensions, vec![2, 3]);
    assert_eq!(file.tensor_bytes(f32_info), f32_payload.as_slice());

    // F16 tensor: dtype + shape + payload byte-identical.
    let f16_info = file
        .tensor_info("decoder.output_layer.weight")
        .expect("F16 tensor present");
    assert_eq!(f16_info.dtype, GgmlType::F16, "F16 stays F16 (no widen)");
    assert_eq!(f16_info.dimensions, vec![1, 4]);
    assert_eq!(file.tensor_bytes(f16_info), f16_payload.as_slice());

    // BF16 tensor: the pass-through pin (a silent widen to F32 would
    // change the on-disk dtype tag AND balloon the payload from 12 B
    // → 24 B, so this assertion double-locks the invariant).
    let bf16_info = file
        .tensor_info("encoder.blocks.0.attn.qkv_proj.weight")
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
        Some("nemotron-speech-streaming-v2603"),
        "arch stamp distinct from any sibling parakeet / canary tag — silent alias would misroute"
    );
    assert_eq!(
        file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
        Some("nemotron-speech-streaming-v2603")
    );
    assert_eq!(
        file.get("vokra.model.category").and_then(|v| v.as_str()),
        Some("asr"),
        "category groups Nemotron-Speech-Streaming-v2603 with the ASR family for the zoo manifest"
    );
    assert_eq!(
        file.get("vokra.provenance.upstream_url")
            .and_then(|v| v.as_str()),
        Some(
            "https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/nemotron_speech_streaming_v2603"
        ),
        "upstream_url pins traceability back to the NGC catalog (no HF mirror at land time)"
    );
    // upstream_hf must NOT be stamped for this model — NGC-only
    // distribution at land time means `upstream_url` is the canonical
    // provenance key and stamping a speculative HF slug would
    // fabricate a mirror that does not exist (the honesty pin).
    assert!(
        file.get("vokra.provenance.upstream_hf").is_none(),
        "upstream_hf MUST NOT be stamped for NGC-only Nemotron — no HF mirror at land time"
    );
    assert_eq!(
        file.get(chunks::KEY_PROVENANCE_LICENSE)
            .and_then(|v| v.as_str()),
        Some("apache-2.0"),
        "default license is apache-2.0 (Permissive)"
    );
    assert_eq!(
        file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
            .and_then(|v| v.as_str()),
        Some(LicenseClass::Permissive.as_str())
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

/// Direct `convert_nemotron_speech_streaming_v2603_file` entry-point
/// exercise: the report counters must add up
/// (`read == written + skipped_non_float`) and the subset counters
/// agree with the pass-through matrix — a regression where the F16 arm
/// silently reclassified BF16 as F16 would flip both counters, so this
/// asserts them independently.
#[test]
fn nemotron_direct_entry_point_returns_matching_report() {
    let (input_bytes, _, _, _) = synthetic_nemotron_safetensors();
    let input = tmp_path("nemotron-direct-in");
    let output = tmp_path("nemotron-direct-out");
    std::fs::write(&input, &input_bytes).expect("write input");

    let report: NemotronSpeechStreamingV2603Report =
        convert_nemotron_speech_streaming_v2603_file(&input, &output, None)
            .expect("convert_nemotron_speech_streaming_v2603_file");

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
        "exactly one BF16 tensor was in the fixture"
    );
    assert_eq!(
        report.read,
        report.written + report.skipped_non_float,
        "read = written + skipped invariant (mirror of qwen3_tts / frcrn pattern)"
    );

    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&output);
}

/// The `--model nemotron-speech-streaming-v2603` alias must dispatch to
/// `ModelKind::NemotronSpeechStreamingV2603` (and the canonical
/// `as_arg` must round-trip). Pinned separately from the alias walk in
/// `lib.rs` so a future dropped `NemotronSpeechStreamingV2603` arm in
/// either direction is caught at the integration surface.
#[test]
fn nemotron_alias_dispatch_round_trips() {
    // Canonical spelling round-trips.
    let kind = ModelKind::from_arg("nemotron-speech-streaming-v2603")
        .expect("`--model nemotron-speech-streaming-v2603` must resolve");
    assert_eq!(kind, ModelKind::NemotronSpeechStreamingV2603);
    assert_eq!(kind.as_arg(), "nemotron-speech-streaming-v2603");

    // Underscore + NGC-path + short-family aliases all dispatch to the
    // same variant (mirror of the parakeet-ctc-1.1b sibling posture —
    // model cards / catalog URLs use `_v2603` while CLI arguments
    // typically use hyphens).
    for alias in [
        "nemotron-speech-streaming",
        "nemotron_speech_streaming_v2603",
        "nemotron_speech_streaming",
        "nvidia/nemotron-speech-streaming-v2603",
        "nvidia/nemotron_speech_streaming_v2603",
        "nvidia/nemo/nemotron_speech_streaming_v2603",
    ] {
        let k = ModelKind::from_arg(alias).unwrap_or_else(|| {
            panic!("alias {alias} must resolve to NemotronSpeechStreamingV2603")
        });
        assert_eq!(
            k,
            ModelKind::NemotronSpeechStreamingV2603,
            "alias {alias} dispatched to the wrong variant"
        );
    }

    // Sibling NeMo family arms must not accept this SKU — silent
    // misroute guard.
    assert_ne!(
        ModelKind::from_arg("parakeet-tdt").unwrap_or(ModelKind::NemotronSpeechStreamingV2603),
        ModelKind::NemotronSpeechStreamingV2603,
        "`parakeet-tdt` must resolve to the Parakeet TDT arm, NOT NemotronSpeechStreamingV2603"
    );
    assert_ne!(
        ModelKind::from_arg("parakeet-ctc").unwrap_or(ModelKind::NemotronSpeechStreamingV2603),
        ModelKind::NemotronSpeechStreamingV2603,
        "`parakeet-ctc` must resolve to the Parakeet CTC arm, NOT NemotronSpeechStreamingV2603"
    );
    assert_ne!(
        ModelKind::from_arg("canary").unwrap_or(ModelKind::NemotronSpeechStreamingV2603),
        ModelKind::NemotronSpeechStreamingV2603,
        "`canary` must resolve to the Canary arm, NOT NemotronSpeechStreamingV2603"
    );
}
