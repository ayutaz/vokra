//! External roundtrip test for the Sortformer-Diar-4spk-v1 converter
//! (coverage-audit 2026-08-03 Wave B ticket).
//!
//! Exercises the [`convert_file`] / [`convert_file_licensed`] dispatch
//! (i.e. the outward `ModelKind::SortformerDiar4spkV1` arm — not the
//! module-internal `convert_sortformer_diar_4spk_v1_file`) with a
//! synthetic BF16 safetensors, so the wire-up between the CLI-facing
//! enum and the file-based converter is held under the same regression
//! watch as the sibling xcodec2 / hibiki / parakeet_unified / neucodec
//! / emotion2vec skeletons.

use std::path::PathBuf;

use vokra_convert::{
    ModelKind, convert_file, convert_file_licensed, convert_sortformer_diar_4spk_v1_file,
};
use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufFile};

/// A unique temp path for this test process. Nanosecond suffix keeps
/// parallel `cargo test` runs from colliding.
fn tmp_path(tag: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "vokra-convert-sortformer-it-{tag}-{}-{}",
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
/// test fixture in `models::sortformer_diar_4spk_v1`, kept private to
/// this file so the external test remains self-contained.
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
fn convert_file_dispatch_lands_sortformer_metadata_and_bf16_passthrough_with_nc_gate() {
    let (_values, bf16) = synthetic_bf16_payload();
    let input_bytes = safetensors_one_bf16("encoder.layers.0.self_attn.qkv.weight", &[2, 2], &bf16);

    let input = tmp_path("dispatch-in");
    let output = tmp_path("dispatch-out");
    std::fs::write(&input, &input_bytes).expect("write input");

    // Route through the outward `convert_file` -> `convert_file_licensed`
    // arm so the ModelKind::SortformerDiar4spkV1 dispatch is exercised
    // end-to-end.
    let summary = convert_file(ModelKind::SortformerDiar4spkV1, &input, &output).expect("convert");
    assert_eq!(summary.model, ModelKind::SortformerDiar4spkV1);
    assert_eq!(summary.tensor_count, 1, "one float tensor written");
    assert!(
        summary
            .notes
            .iter()
            .any(|n| n.starts_with("sortformer-diar-4spk-v1:") && n.contains("BF16 passthrough")),
        "notes must surface the sortformer pass-through counter, got {:?}",
        summary.notes
    );

    let file = GgufFile::open(&output).expect("load output gguf");
    let info = file
        .tensor_info("encoder.layers.0.self_attn.qkv.weight")
        .expect("BF16 tensor present");
    assert_eq!(
        info.dtype,
        GgmlType::BF16,
        "BF16 must not be widened at convert time (GGUF type 30 verbatim)"
    );
    assert_eq!(file.tensor_bytes(info), bf16.as_slice());

    // Provenance defaults are the CC-BY-NC 4.0 / NonCommercial NVIDIA
    // posture — the whole reason Sortformer gets its own arm instead
    // of the Permissive fleet arm (a silent Apache-2.0 default would
    // mis-classify NC weights on load and let commercial-mode callers
    // load them without a research flag).
    assert_eq!(
        file.get("vokra.model.arch").and_then(|v| v.as_str()),
        Some("sortformer")
    );
    assert_eq!(
        file.get("vokra.model.name").and_then(|v| v.as_str()),
        Some("sortformer-diar-4spk-v1")
    );
    assert_eq!(
        file.get("vokra.model.category").and_then(|v| v.as_str()),
        Some("diarize")
    );
    assert_eq!(
        file.get("vokra.provenance.upstream_hf")
            .and_then(|v| v.as_str()),
        Some("nvidia/diar_sortformer_4spk-v1")
    );
    assert_eq!(
        file.get("vokra.provenance.license")
            .and_then(|v| v.as_str()),
        Some("cc-by-nc-4.0")
    );
    assert_eq!(
        file.get("vokra.provenance.weight_license")
            .and_then(|v| v.as_str()),
        Some(LicenseClass::NonCommercial.as_str())
    );

    // The M2-13 runtime gate refuses to load the resulting GGUF in
    // commercial mode (`LicenseClass::NonCommercial::requires_research_flag
    // = true`) — an operator who never touched the license flag cannot
    // silently bring up an NC weight in production. Assert this end-
    // to-end so a regression that reclassified NonCommercial as
    // Permissive (silently letting the strict gate pass) fails loudly
    // here.
    let res = vokra_core::resolve_license_class(&file);
    assert_eq!(res.class, LicenseClass::NonCommercial);
    assert!(res.is_research_only());
    let strict_err =
        vokra_core::check_weight_license(&file, &vokra_core::CompliancePolicy::strict())
            .expect_err("strict policy MUST refuse cc-by-nc-4.0 without a research flag");
    // The error surfaces the class so a downstream operator can act on
    // it. A regression that dropped the class from the message would
    // still refuse loading (good) but leave the operator without a
    // clear signal (bad); pin the substring.
    let msg = format!("{strict_err}");
    assert!(
        msg.contains("NonCommercial") || msg.contains("non-commercial") || msg.contains("nc"),
        "strict-policy refusal must mention the NC class in its error, got {msg:?}"
    );

    // A research-scoped policy MUST accept the same artifact — this is
    // the whole point of the fail-closed gate (opt-in, not blanket
    // refusal). A regression that made NonCommercial unloadable
    // outright would fail here.
    vokra_core::check_weight_license(
        &file,
        &vokra_core::CompliancePolicy::strict().with_research_license(true),
    )
    .expect("research-scoped policy MUST accept cc-by-nc-4.0");

    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&output);
}

#[test]
fn convert_file_licensed_override_swaps_the_stamped_licence() {
    let (_values, bf16) = synthetic_bf16_payload();
    let input_bytes = safetensors_one_bf16("encoder.layers.0.linear.weight", &[2, 2], &bf16);

    let input = tmp_path("override-in");
    let output = tmp_path("override-out");
    std::fs::write(&input, &input_bytes).expect("write input");

    // Override with Apache-2.0 — e.g. a caller who retrained on a
    // permissive corpus. The default path stamps cc-by-nc-4.0 +
    // NonCommercial; the override must re-stamp both (SPDX + class).
    // A regression that dropped the class rederivation would leave
    // NonCommercial stamped alongside Apache-2.0 — an inconsistent
    // artifact the model-zoo gate would refuse to publish.
    let summary = convert_file_licensed(
        ModelKind::SortformerDiar4spkV1,
        &input,
        &output,
        Some("apache-2.0"),
    )
    .expect("convert_file_licensed with SPDX override");
    assert_eq!(summary.tensor_count, 1);

    let file = GgufFile::open(&output).expect("load output gguf");
    assert_eq!(
        file.get("vokra.provenance.license")
            .and_then(|v| v.as_str()),
        Some("apache-2.0"),
        "override SPDX must land in `vokra.provenance.license`"
    );
    assert_eq!(
        file.get("vokra.provenance.weight_license")
            .and_then(|v| v.as_str()),
        Some(LicenseClass::Permissive.as_str()),
        "override must reclassify the weight-class alongside the SPDX"
    );

    // With Permissive the strict gate MUST accept — otherwise the
    // override path silently kept the fail-closed default.
    vokra_core::check_weight_license(&file, &vokra_core::CompliancePolicy::strict())
        .expect("Permissive override MUST clear the strict policy");

    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&output);
}

#[test]
fn direct_convert_sortformer_file_equivalent_to_dispatch() {
    // Confirms the file-based re-export and the
    // `ModelKind::SortformerDiar4spkV1` dispatch arm land the same
    // bytes over the same input — a regression fence against the two
    // entry points drifting apart (they must share
    // `models::sortformer_diar_4spk_v1::convert_sortformer_diar_4spk_v1_file`).
    let (_values, bf16) = synthetic_bf16_payload();
    let input_bytes = safetensors_one_bf16("out_head.sigmoid_proj.weight", &[1, 4], &bf16);

    let input_a = tmp_path("direct-in-a");
    let output_a = tmp_path("direct-out-a");
    let input_b = tmp_path("direct-in-b");
    let output_b = tmp_path("direct-out-b");
    std::fs::write(&input_a, &input_bytes).expect("write A");
    std::fs::write(&input_b, &input_bytes).expect("write B");

    let report =
        convert_sortformer_diar_4spk_v1_file(&input_a, &output_a, None).expect("direct convert");
    assert_eq!(report.written, 1);
    assert_eq!(report.bf16_passthrough, 1);

    let summary = convert_file(ModelKind::SortformerDiar4spkV1, &input_b, &output_b)
        .expect("dispatch convert");
    assert_eq!(summary.tensor_count, 1);

    let bytes_a = std::fs::read(&output_a).expect("read A");
    let bytes_b = std::fs::read(&output_b).expect("read B");
    assert_eq!(
        bytes_a, bytes_b,
        "direct convert_sortformer_diar_4spk_v1_file and \
         ModelKind::SortformerDiar4spkV1 dispatch must produce byte-identical \
         GGUFs for the same input"
    );

    let _ = std::fs::remove_file(&input_a);
    let _ = std::fs::remove_file(&output_a);
    let _ = std::fs::remove_file(&input_b);
    let _ = std::fs::remove_file(&output_b);
}
