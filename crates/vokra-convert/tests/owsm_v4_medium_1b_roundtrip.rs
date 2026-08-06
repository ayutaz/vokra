//! External roundtrip test for the ESPnet OWSM v4 medium 1B converter
//! (coverage-audit 2026-08-03 Wave B ticket).
//!
//! Exercises the [`convert_file`] / [`convert_file_licensed`] dispatch
//! (i.e. the outward `ModelKind::OwsmV4Medium1b` arm — not the module-
//! internal `convert_owsm_v4_medium_1b_file`) with a synthetic BF16
//! safetensors, so the wire-up between the CLI-facing enum and the
//! file-based converter is held under the same regression watch as the
//! sibling neucodec / emotion2vec / hibiki skeletons.

use std::path::PathBuf;

use vokra_convert::{
    ModelKind, convert_file, convert_file_licensed, convert_owsm_v4_medium_1b_file,
};
use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufFile};

/// A unique temp path for this test process. Nanosecond suffix keeps
/// parallel `cargo test` runs from colliding.
fn tmp_path(tag: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "vokra-convert-owsm-v4-medium-1b-it-{tag}-{}-{}",
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
/// test fixture in `models::owsm_v4_medium_1b`, kept private to this
/// file so the external test remains self-contained.
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
fn convert_file_dispatch_lands_owsm_metadata_and_bf16_passthrough() {
    let (_values, bf16) = synthetic_bf16_payload();
    let input_bytes = safetensors_one_bf16("encoder.embed.weight", &[2, 2], &bf16);

    let input = tmp_path("dispatch-in");
    let output = tmp_path("dispatch-out");
    std::fs::write(&input, &input_bytes).expect("write input");

    // Route through the outward `convert_file` -> `convert_file_licensed`
    // arm so the ModelKind::OwsmV4Medium1b dispatch is exercised end-to-end.
    let summary = convert_file(ModelKind::OwsmV4Medium1b, &input, &output).expect("convert");
    assert_eq!(summary.model, ModelKind::OwsmV4Medium1b);
    assert_eq!(summary.tensor_count, 1, "one float tensor written");
    assert!(
        summary
            .notes
            .iter()
            .any(|n| n.starts_with("owsm-v4-medium-1b:") && n.contains("BF16 passthrough")),
        "notes must surface the OWSM pass-through counter, got {:?}",
        summary.notes
    );

    let file = GgufFile::open(&output).expect("load output gguf");
    let info = file
        .tensor_info("encoder.embed.weight")
        .expect("BF16 tensor present");
    assert_eq!(
        info.dtype,
        GgmlType::BF16,
        "BF16 must not be widened at convert time (GGUF type 30 verbatim)"
    );
    assert_eq!(file.tensor_bytes(info), bf16.as_slice());

    // Provenance defaults are the CC-BY 4.0 / AttributionRequired ESPnet
    // posture — the whole reason OWSM gets its own arm instead of the
    // Permissive fleet arm.
    assert_eq!(
        file.get("vokra.model.arch").and_then(|v| v.as_str()),
        Some("owsm-v4-medium-1b")
    );
    assert_eq!(
        file.get("vokra.model.name").and_then(|v| v.as_str()),
        Some("owsm-v4-medium-1b")
    );
    assert_eq!(
        file.get("vokra.model.category").and_then(|v| v.as_str()),
        Some("asr")
    );
    assert_eq!(
        file.get("vokra.provenance.upstream_hf")
            .and_then(|v| v.as_str()),
        Some("espnet/owsm_v4_medium_1B")
    );
    assert_eq!(
        file.get("vokra.provenance.license")
            .and_then(|v| v.as_str()),
        Some("cc-by-4.0")
    );
    assert_eq!(
        file.get("vokra.provenance.weight_license")
            .and_then(|v| v.as_str()),
        Some(LicenseClass::AttributionRequired.as_str())
    );
    let attribution = file
        .get("vokra.provenance.attribution")
        .and_then(|v| v.as_str())
        .expect("attribution string must be stamped for CC-BY 4.0");
    assert!(
        attribution.contains("ESPnet") && attribution.contains("CC-BY 4.0"),
        "attribution must name ESPnet and cite CC-BY 4.0, got {attribution}"
    );

    // The M2-13 gate resolves AttributionRequired and passes the
    // strict (commercial) policy WITHOUT a research flag — CC-BY 4.0
    // is commercial-OK (never confuse with the CC-BY-NC gate).
    let res = vokra_core::resolve_license_class(&file);
    assert_eq!(res.class, LicenseClass::AttributionRequired);
    assert!(!res.is_research_only());
    vokra_core::check_weight_license(&file, &vokra_core::CompliancePolicy::strict())
        .expect("CC-BY 4.0 passes the strict gate");

    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&output);
}

#[test]
fn convert_file_licensed_override_swaps_the_stamped_licence() {
    let (_values, bf16) = synthetic_bf16_payload();
    let input_bytes = safetensors_one_bf16("decoder.output.weight", &[2, 2], &bf16);

    let input = tmp_path("override-in");
    let output = tmp_path("override-out");
    std::fs::write(&input, &input_bytes).expect("write input");

    // Override with a plain MIT SPDX id. The default path stamps
    // cc-by-4.0 + AttributionRequired; the override must re-stamp both.
    let summary = convert_file_licensed(ModelKind::OwsmV4Medium1b, &input, &output, Some("MIT"))
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
fn direct_convert_owsm_v4_medium_1b_file_equivalent_to_dispatch() {
    // Confirms the file-based re-export and the
    // `ModelKind::OwsmV4Medium1b` dispatch arm land the same bytes
    // over the same input — a regression fence against the two entry
    // points drifting apart (they must share
    // `models::owsm_v4_medium_1b::convert_owsm_v4_medium_1b_file`).
    let (_values, bf16) = synthetic_bf16_payload();
    let input_bytes = safetensors_one_bf16("encoder.norm.weight", &[1, 4], &bf16);

    let input_a = tmp_path("direct-in-a");
    let output_a = tmp_path("direct-out-a");
    let input_b = tmp_path("direct-in-b");
    let output_b = tmp_path("direct-out-b");
    std::fs::write(&input_a, &input_bytes).expect("write A");
    std::fs::write(&input_b, &input_bytes).expect("write B");

    let report = convert_owsm_v4_medium_1b_file(&input_a, &output_a, None).expect("direct convert");
    assert_eq!(report.written, 1);
    assert_eq!(report.bf16_passthrough, 1);

    let summary =
        convert_file(ModelKind::OwsmV4Medium1b, &input_b, &output_b).expect("dispatch convert");
    assert_eq!(summary.tensor_count, 1);

    let bytes_a = std::fs::read(&output_a).expect("read A");
    let bytes_b = std::fs::read(&output_b).expect("read B");
    assert_eq!(
        bytes_a, bytes_b,
        "direct convert_owsm_v4_medium_1b_file and ModelKind::OwsmV4Medium1b \
         dispatch must produce byte-identical GGUFs for the same input"
    );

    let _ = std::fs::remove_file(&input_a);
    let _ = std::fs::remove_file(&output_a);
    let _ = std::fs::remove_file(&input_b);
    let _ = std::fs::remove_file(&output_b);
}
