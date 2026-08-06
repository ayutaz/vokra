//! Integration test for the Sber GigaAM v3 converter
//! (coverage-audit-2026-08-03 Wave B fast-track).
//!
//! Mirrors the in-module unit tests but exercises the outer public
//! entry points (`convert_sber_gigaam_v3_file` re-export + the
//! `ModelKind::SberGigaamV3` dispatch through `convert_file`).
//! No large real checkpoint is committed; real-model E2E is a manual
//! local run of the `vokra-convert` binary.

use std::path::PathBuf;

use vokra_convert::{ModelKind, SberGigaamV3Report, convert_file, convert_sber_gigaam_v3_file};
use vokra_core::gguf::{GgmlType, GgufFile};

/// Per-test unique scratch path (PID + nanos + a tag so concurrent
/// runs cannot collide).
fn scratch_path(tag: &str, ext: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "vokra-sber-gigaam-v3-it-{tag}-{}-{}.{ext}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0),
    ));
    p
}

/// RAII cleanup for temp files (best-effort — a panic mid-cleanup is
/// fine).
struct TempFileGuard(PathBuf);
impl Drop for TempFileGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// A tiny safetensors buffer with two F32 tensors under Sber-style
/// upstream state-dict keys (the runtime name contract is verbatim).
fn synthetic_safetensors_two_f32() -> Vec<u8> {
    let a: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    let b: Vec<u8> = [5.0f32, 6.0].iter().flat_map(|f| f.to_le_bytes()).collect();
    let header = r#"{"encoder.layers.0.self_attn.qkv_proj.weight":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},"encoder.layers.0.ffn.linear1.bias":{"dtype":"F32","shape":[2],"data_offsets":[16,24]}}"#;
    let mut out = Vec::new();
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&a);
    out.extend_from_slice(&b);
    out
}

/// `convert_sber_gigaam_v3_file` end-to-end: two F32 tensors pass
/// through verbatim, provenance + category chunks are stamped, and the
/// file round-trips through the GGUF loader with the tensor names +
/// bytes preserved.
#[test]
fn convert_sber_gigaam_v3_file_roundtrips_two_f32_tensors() {
    let input_path = scratch_path("file-in", "safetensors");
    let output_path = scratch_path("file-out", "gguf");
    std::fs::write(&input_path, synthetic_safetensors_two_f32()).expect("write input");
    let _in_guard = TempFileGuard(input_path.clone());
    let _out_guard = TempFileGuard(output_path.clone());

    let report: SberGigaamV3Report =
        convert_sber_gigaam_v3_file(&input_path, &output_path, None).expect("convert");
    assert_eq!(report.read, 2, "two tensors in the header");
    assert_eq!(report.written, 2, "both F32 tensors pass through");
    assert_eq!(report.skipped_non_float, 0);
    assert_eq!(
        report.bf16_passthrough, 0,
        "F32-only input must leave the BF16 counter at Default 0"
    );

    let file = GgufFile::open(&output_path).expect("load output GGUF");
    // Both tensors are present under their upstream state-dict keys.
    let info_a = file
        .tensor_info("encoder.layers.0.self_attn.qkv_proj.weight")
        .expect("F32 tensor A present");
    assert_eq!(info_a.dtype, GgmlType::F32);
    assert_eq!(info_a.dimensions, vec![2, 2]);
    let info_b = file
        .tensor_info("encoder.layers.0.ffn.linear1.bias")
        .expect("F32 tensor B present");
    assert_eq!(info_b.dtype, GgmlType::F32);
    assert_eq!(info_b.dimensions, vec![2]);
    // First tensor's bytes survive intact.
    let want_a: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    assert_eq!(file.tensor_bytes(info_a), want_a.as_slice());
}

/// `convert_file(ModelKind::SberGigaamV3, ...)` dispatch: the outer
/// entry point (used by the CLI + workflow) routes through the
/// `convert_file_licensed` arm to the file-based converter, and the
/// resulting `ConvertSummary` names the SberGigaamV3 model.
#[test]
fn model_kind_dispatch_roundtrips_through_convert_file() {
    let input_path = scratch_path("dispatch-in", "safetensors");
    let output_path = scratch_path("dispatch-out", "gguf");
    std::fs::write(&input_path, synthetic_safetensors_two_f32()).expect("write input");
    let _in_guard = TempFileGuard(input_path.clone());
    let _out_guard = TempFileGuard(output_path.clone());

    let summary =
        convert_file(ModelKind::SberGigaamV3, &input_path, &output_path).expect("dispatch");
    assert_eq!(summary.model, ModelKind::SberGigaamV3);
    assert_eq!(
        summary.tensor_count, 2,
        "both F32 tensors round-trip via the dispatch entry point"
    );
    // The dispatch arm records `metadata_count = 0` (populated by the
    // inner file-based converter, mirror of every sibling file-based
    // ModelKind arm — DebertaV2 / DebertaV3 / SbV2 / KimiAudio /
    // StepAudio2Mini / ...). This is intentional and not a defect.
    assert_eq!(summary.metadata_count, 0);
    assert!(
        !summary.notes.is_empty(),
        "the dispatch arm surfaces a diagnostic note (float weights + BF16 pass-through + \
         non-float skipped counters)"
    );
    assert!(
        summary
            .notes
            .iter()
            .any(|n| n.contains("sber-gigaam-v3") && n.contains("float weights written verbatim")),
        "the note names the model and reports the pass-through counters; got {:?}",
        summary.notes
    );

    // The output is mmap-loadable and carries the arch chunk under the
    // canonical arch tag.
    let file = GgufFile::open(&output_path).expect("load output GGUF");
    assert_eq!(
        file.get("vokra.model.arch").and_then(|v| v.as_str()),
        Some("sber_gigaam_v3"),
        "arch chunk carries the canonical underscore variant"
    );
    assert_eq!(
        file.get("vokra.model.name").and_then(|v| v.as_str()),
        Some("gigaam-v3"),
        "name chunk carries the canonical hyphenated variant"
    );
    assert_eq!(
        file.get("vokra.model.category").and_then(|v| v.as_str()),
        Some("asr"),
        "category chunk stays at the first-word ASR tag"
    );
    assert_eq!(
        file.get("vokra.provenance.upstream_hf")
            .and_then(|v| v.as_str()),
        Some("ai-sage/GigaAM-v3"),
        "upstream_hf chunk pins the ai-sage HF collection"
    );
}

/// `ModelKind::from_arg` accepts every documented spelling and rejects
/// unknown ones (regression guard for the alias table in `lib.rs`).
#[test]
fn model_kind_from_arg_accepts_all_documented_spellings() {
    for spelling in [
        "sber-gigaam-v3",
        "sber_gigaam_v3",
        "gigaam-v3",
        "gigaam_v3",
        "ai-sage/gigaam-v3",
        "ai-sage/GigaAM-v3",
    ] {
        assert_eq!(
            ModelKind::from_arg(spelling),
            Some(ModelKind::SberGigaamV3),
            "spelling `{spelling}` must resolve to SberGigaamV3"
        );
    }
    // Unrelated slugs stay unknown.
    assert_ne!(
        ModelKind::from_arg("sber-gigaam-v4"),
        Some(ModelKind::SberGigaamV3),
        "future v4 must not silently alias to v3"
    );
    assert_ne!(
        ModelKind::from_arg("gigaam"),
        Some(ModelKind::SberGigaamV3),
        "bare `gigaam` (no version) must not silently alias to v3"
    );
}

/// `ModelKind::SberGigaamV3::as_arg()` returns the canonical hyphenated
/// slug, and the round-trip through `from_arg` recovers the same
/// variant.
#[test]
fn model_kind_as_arg_roundtrips_through_from_arg() {
    let canonical = ModelKind::SberGigaamV3.as_arg();
    assert_eq!(canonical, "sber-gigaam-v3");
    assert_eq!(
        ModelKind::from_arg(canonical),
        Some(ModelKind::SberGigaamV3),
        "as_arg -> from_arg round-trip must recover the same variant"
    );
}
