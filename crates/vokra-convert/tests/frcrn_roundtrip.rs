//! FRCRN converter integration test (coverage-audit wave-a, 2026-08-03).
//!
//! Exercises the public [`convert_file`] entry point through the
//! [`ModelKind::Frcrn`] dispatch arm (mirror of the `whisper` /
//! `emotion2vec` roundtrip pattern in `roundtrip.rs`). A synthetic
//! safetensors buffer with a mix of F32 / F16 / BF16 tensors is written
//! to disk, converted via the public API, and the resulting GGUF is
//! loaded back with the runtime loader — every tensor's dtype +
//! payload must survive the pipeline byte-identical (the pass-through
//! contract) and the provenance / category / upstream_hf stamps must
//! land on the artifact so the publish pipeline can gate on the GGUF
//! alone (no side-car lookup).
//!
//! Real-weight parity with the upstream FRCRN reference is deferred to
//! owner (§3.1 sign-off). This test locks the byte-parallel GGUF
//! surface so a future `FrcrnWeights::from_gguf` can bind against a
//! stable schema.

use std::path::PathBuf;

use vokra_convert::{FrcrnReport, ModelKind, convert_file, convert_frcrn_file};
use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufFile, chunks};

/// A unique temp path for this test process (mirror of
/// `roundtrip.rs::tmp_path` — no external `tempfile` dep, preserving
/// zero-dep NFR-DS-02).
fn tmp_path(tag: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "vokra-convert-frcrn-it-{tag}-{}-{}",
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
/// converter accepts). Layout mirrors the emotion2vec / wespeaker
/// module-level fixtures.
fn synthetic_frcrn_safetensors() -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
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

    // Realistic-looking FRCRN tensor names — Complex U-Net + freq-
    // recurrent LSTM topology. Names are deliberately from the
    // author-repo naming scheme so a future `FrcrnWeights::from_gguf`
    // can walk the same manifest.
    let f32_off_end = f32_bytes.len();
    let f16_off_end = f32_off_end + f16_bytes.len();
    let bf16_off_end = f16_off_end + bf16_bytes.len();
    let header = format!(
        r#"{{"encoder.complex_conv0.weight":{{"dtype":"F32","shape":[2,3],"data_offsets":[0,{f32_off_end}]}},"decoder.complex_conv0.weight":{{"dtype":"F16","shape":[1,4],"data_offsets":[{f32_off_end},{f16_off_end}]}},"rnn.weight_ih_l0":{{"dtype":"BF16","shape":[2,3],"data_offsets":[{f16_off_end},{bf16_off_end}]}}}}"#
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
/// (the `--model frcrn` CLI path): every dtype survives the pipeline
/// byte-identical, tensor names / shapes round-trip, and the
/// provenance / category / upstream stamps land on the artifact.
#[test]
fn frcrn_safetensors_roundtrips_through_convert_file() {
    let (input_bytes, f32_payload, f16_payload, bf16_payload) = synthetic_frcrn_safetensors();
    let input = tmp_path("frcrn-in");
    let output = tmp_path("frcrn-out");
    std::fs::write(&input, &input_bytes).expect("write input");

    let summary = convert_file(ModelKind::Frcrn, &input, &output).expect("convert");
    assert_eq!(summary.model, ModelKind::Frcrn);
    assert_eq!(summary.tensor_count, 3, "3 float tensors written");
    assert!(
        summary.output_bytes > 0,
        "output GGUF must have non-empty size"
    );
    assert_eq!(summary.notes.len(), 1, "single summary note emitted");
    assert!(
        summary.notes[0].contains("frcrn: 3 float weights"),
        "summary must mention the frcrn count: {}",
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
        .tensor_info("encoder.complex_conv0.weight")
        .expect("F32 tensor present");
    assert_eq!(f32_info.dtype, GgmlType::F32, "F32 stays F32 (no widen)");
    assert_eq!(f32_info.dimensions, vec![2, 3]);
    assert_eq!(file.tensor_bytes(f32_info), f32_payload.as_slice());

    // F16 tensor: dtype + shape + payload byte-identical.
    let f16_info = file
        .tensor_info("decoder.complex_conv0.weight")
        .expect("F16 tensor present");
    assert_eq!(f16_info.dtype, GgmlType::F16, "F16 stays F16 (no widen)");
    assert_eq!(f16_info.dimensions, vec![1, 4]);
    assert_eq!(file.tensor_bytes(f16_info), f16_payload.as_slice());

    // BF16 tensor: the pass-through pin (a silent widen to F32 would
    // change the on-disk dtype tag AND balloon the payload from 12 B
    // → 24 B, so this assertion double-locks the invariant).
    let bf16_info = file
        .tensor_info("rnn.weight_ih_l0")
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
        Some("frcrn"),
        "arch stamp distinct from `denoise` (DFN3) — silent alias would misroute"
    );
    assert_eq!(
        file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
        Some("frcrn")
    );
    assert_eq!(
        file.get("vokra.model.category").and_then(|v| v.as_str()),
        Some("denoise"),
        "category groups FRCRN with the DFN3 denoise family for the zoo manifest"
    );
    assert_eq!(
        file.get("vokra.provenance.upstream_hf")
            .and_then(|v| v.as_str()),
        Some("alibabasglab/FRCRN"),
        "upstream slug pins traceability back to the author repo"
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

/// Direct `convert_frcrn_file` entry-point exercise: the report
/// counters must add up (`read == written + skipped_non_float`) and the
/// subset counters agree with the pass-through matrix — a regression
/// where the F16 arm silently reclassified BF16 as F16 would flip both
/// counters, so this asserts them independently.
#[test]
fn frcrn_direct_entry_point_returns_matching_report() {
    let (input_bytes, _, _, _) = synthetic_frcrn_safetensors();
    let input = tmp_path("frcrn-direct-in");
    let output = tmp_path("frcrn-direct-out");
    std::fs::write(&input, &input_bytes).expect("write input");

    let report: FrcrnReport =
        convert_frcrn_file(&input, &output, None).expect("convert_frcrn_file");

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
        "read = written + skipped invariant (mirror of qwen3_tts pattern)"
    );

    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&output);
}

/// The `--model frcrn` alias must dispatch to `ModelKind::Frcrn` (and
/// the canonical `as_arg` must round-trip). Pinned separately from the
/// alias walk in `lib.rs` so a future dropped `Frcrn` arm in either
/// direction is caught at the integration surface.
#[test]
fn frcrn_alias_dispatch_round_trips() {
    // Canonical spelling round-trips.
    let kind = ModelKind::from_arg("frcrn").expect("`--model frcrn` must resolve");
    assert_eq!(kind, ModelKind::Frcrn);
    assert_eq!(kind.as_arg(), "frcrn");

    // Repository-slug + ClearerVoice-Studio spellings all dispatch to
    // the same variant.
    for alias in [
        "alibabasglab/frcrn",
        "clearervoice-studio/frcrn",
        "modelscope/clearervoice-studio-frcrn",
    ] {
        assert_eq!(
            ModelKind::from_arg(alias),
            Some(ModelKind::Frcrn),
            "--model {alias} must dispatch to ModelKind::Frcrn"
        );
    }

    // Unrelated FRCRN-shaped strings must NOT resolve (silent misroute
    // guard — FR-EX-08 posture).
    for miss in ["frcrn-v2", "frcrn/large", "frcrn-huge"] {
        assert!(
            ModelKind::from_arg(miss).is_none(),
            "{miss:?} must NOT resolve to any ModelKind"
        );
    }
}
