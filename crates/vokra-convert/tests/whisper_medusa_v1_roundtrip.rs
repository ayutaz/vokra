//! External roundtrip test for the aiola Whisper-Medusa-v1 converter
//! (coverage-audit 2026-08-03 Wave B ticket).
//!
//! Exercises the [`convert_file`] / [`convert_file_licensed`] dispatch
//! (i.e. the outward `ModelKind::WhisperMedusaV1` arm — not the module-
//! internal `convert_whisper_medusa_v1_file`) with a synthetic BF16
//! safetensors, so the wire-up between the CLI-facing enum and the
//! file-based converter is held under the same regression watch as the
//! sibling neucodec / emotion2vec / hibiki skeletons. The **distinct
//! arch tag** (`"whisper-medusa-v1"` — not `"whisper"`) is pinned
//! here so a silent aliasing regression that would drop the Medusa
//! heads at load fails loudly at conversion time.

use std::path::PathBuf;

use vokra_convert::{
    ModelKind, convert_file, convert_file_licensed, convert_whisper_medusa_v1_file,
};
use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufFile};

/// A unique temp path for this test process. Nanosecond suffix keeps
/// parallel `cargo test` runs from colliding (mirror of the hibiki /
/// canary_1b_flash test fixture).
fn tmp_path(tag: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "vokra-convert-whisper-medusa-v1-it-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0)
    ));
    p
}

/// Builds a single-BF16-tensor safetensors buffer with the caller-
/// supplied name / shape / bit pattern. Mirror of the module-internal
/// test fixture in `models::whisper_medusa_v1`, kept private to this
/// file so the external test remains self-contained (the sibling
/// hibiki / canary_1b_flash tests use the same posture).
fn safetensors_one_bf16(name: &str, shape: &[u64], bf16_bytes: &[u8]) -> Vec<u8> {
    let elems: u64 = shape.iter().product();
    let expected = elems as usize * 2;
    assert_eq!(bf16_bytes.len(), expected, "shape × 2 BF16 payload");
    let shape_str = shape
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let header = format!(
        r#"{{"{name}":{{"dtype":"BF16","shape":[{shape_str}],"data_offsets":[0,{}]}}}}"#,
        bf16_bytes.len()
    );
    let mut out = Vec::new();
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(bf16_bytes);
    out
}

/// Non-zero BF16 payload so a silent-widen regression cannot hide
/// behind a trivial zero round-trip.
fn synthetic_bf16_payload() -> ([f32; 4], Vec<u8>) {
    let values: [f32; 4] = [1.0, -2.5, 0.15625, 42.0];
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
        .collect();
    (values, bytes)
}

#[test]
fn convert_file_dispatch_lands_whisper_medusa_v1_metadata_and_bf16_passthrough() {
    let (_values, bf16) = synthetic_bf16_payload();
    // Tensor name from the Medusa-head family — the raison d'être of
    // this converter is that these tensors travel with the base
    // Whisper checkpoint, and vanilla `models::whisper` would drop
    // them on the floor.
    let input_bytes = safetensors_one_bf16("medusa_head.0.linear.weight", &[2, 2], &bf16);

    let input = tmp_path("dispatch-in");
    let output = tmp_path("dispatch-out");
    std::fs::write(&input, &input_bytes).expect("write input");

    // Route through the outward `convert_file` -> `convert_file_licensed`
    // arm so the ModelKind::WhisperMedusaV1 dispatch is exercised end-to-end.
    let summary = convert_file(ModelKind::WhisperMedusaV1, &input, &output).expect("convert");
    assert_eq!(summary.model, ModelKind::WhisperMedusaV1);
    assert_eq!(summary.tensor_count, 1, "one float tensor written");
    assert!(
        summary
            .notes
            .iter()
            .any(|n| n.starts_with("whisper-medusa-v1:") && n.contains("BF16 passthrough")),
        "notes must surface the whisper-medusa-v1 pass-through counter, got {:?}",
        summary.notes
    );

    let file = GgufFile::open(&output).expect("load output gguf");
    let info = file
        .tensor_info("medusa_head.0.linear.weight")
        .expect("Medusa-head BF16 tensor present");
    assert_eq!(
        info.dtype,
        GgmlType::BF16,
        "BF16 must not be widened at convert time (GGUF type 30 verbatim)"
    );
    assert_eq!(file.tensor_bytes(info), bf16.as_slice());

    // Provenance defaults — apache-2.0 Permissive per the ticket
    // header (the aiola-lab precedent; primary-source sign-off pending
    // in docs/license-audit.md §3.1).
    assert_eq!(
        file.get("vokra.model.arch").and_then(|v| v.as_str()),
        Some("whisper-medusa-v1"),
        "arch tag must NOT collide with vanilla `whisper` — a silent \
         alias would drop the Medusa-head tensors on load"
    );
    assert_eq!(
        file.get("vokra.model.name").and_then(|v| v.as_str()),
        Some("whisper-medusa-v1")
    );
    assert_eq!(
        file.get("vokra.model.category").and_then(|v| v.as_str()),
        Some("asr")
    );
    assert_eq!(
        file.get("vokra.provenance.upstream_hf")
            .and_then(|v| v.as_str()),
        Some("aiola/whisper-medusa-v1")
    );
    assert_eq!(
        file.get("vokra.provenance.license")
            .and_then(|v| v.as_str()),
        Some("apache-2.0")
    );
    assert_eq!(
        file.get("vokra.provenance.weight_license")
            .and_then(|v| v.as_str()),
        Some(LicenseClass::Permissive.as_str())
    );

    // The runtime research-flag gate resolves Permissive and passes
    // the strict (commercial) policy without a research flag —
    // apache-2.0 is commercial-OK. If this assertion ever fires it
    // means a downstream re-classifier confused Permissive with
    // NonCommercial.
    let res = vokra_core::resolve_license_class(&file);
    assert_eq!(res.class, LicenseClass::Permissive);
    assert!(!res.is_research_only());
    vokra_core::check_weight_license(&file, &vokra_core::CompliancePolicy::strict())
        .expect("apache-2.0 passes the strict gate");

    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&output);
}

#[test]
fn convert_file_licensed_override_swaps_the_stamped_licence() {
    let (_values, bf16) = synthetic_bf16_payload();
    // Also test with a base Whisper-namespace tensor name to make sure
    // the pass-through walk survives both tensor families.
    let input_bytes = safetensors_one_bf16(
        "model.encoder.layers.0.self_attn.q_proj.weight",
        &[2, 2],
        &bf16,
    );

    let input = tmp_path("override-in");
    let output = tmp_path("override-out");
    std::fs::write(&input, &input_bytes).expect("write input");

    // Override with a plain MIT SPDX id. The default path stamps
    // apache-2.0 + Permissive; the override must re-stamp both (MIT is
    // also Permissive, but the raw SPDX string must change so the
    // downstream can distribute under MIT terms).
    let summary = convert_file_licensed(ModelKind::WhisperMedusaV1, &input, &output, Some("MIT"))
        .expect("convert_file_licensed with SPDX override");
    assert_eq!(summary.tensor_count, 1);

    let file = GgufFile::open(&output).expect("load output gguf");
    assert_eq!(
        file.get("vokra.provenance.license")
            .and_then(|v| v.as_str()),
        Some("MIT"),
        "override SPDX must land in `vokra.provenance.license`"
    );
    assert_eq!(
        file.get("vokra.provenance.weight_license")
            .and_then(|v| v.as_str()),
        Some(LicenseClass::Permissive.as_str()),
        "override must reclassify the weight-class alongside the SPDX"
    );

    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&output);
}

#[test]
fn direct_convert_whisper_medusa_v1_file_equivalent_to_dispatch() {
    // Confirms the file-based re-export and the `ModelKind::WhisperMedusaV1`
    // dispatch arm land the same bytes over the same input — a
    // regression fence against the two entry points drifting apart
    // (they must share `models::whisper_medusa_v1::convert_whisper_medusa_v1_file`).
    let (_values, bf16) = synthetic_bf16_payload();
    let input_bytes = safetensors_one_bf16("medusa_head.2.linear.weight", &[1, 4], &bf16);

    let input_a = tmp_path("direct-in-a");
    let output_a = tmp_path("direct-out-a");
    let input_b = tmp_path("direct-in-b");
    let output_b = tmp_path("direct-out-b");
    std::fs::write(&input_a, &input_bytes).expect("write A");
    std::fs::write(&input_b, &input_bytes).expect("write B");

    let report = convert_whisper_medusa_v1_file(&input_a, &output_a, None).expect("direct convert");
    assert_eq!(report.written, 1);
    assert_eq!(report.bf16_passthrough, 1);

    let summary =
        convert_file(ModelKind::WhisperMedusaV1, &input_b, &output_b).expect("dispatch convert");
    assert_eq!(summary.tensor_count, 1);

    let bytes_a = std::fs::read(&output_a).expect("read A");
    let bytes_b = std::fs::read(&output_b).expect("read B");
    assert_eq!(
        bytes_a, bytes_b,
        "direct convert_whisper_medusa_v1_file and \
         ModelKind::WhisperMedusaV1 dispatch must produce byte-identical \
         GGUFs for the same input"
    );

    let _ = std::fs::remove_file(&input_a);
    let _ = std::fs::remove_file(&output_a);
    let _ = std::fs::remove_file(&input_b);
    let _ = std::fs::remove_file(&output_b);
}
