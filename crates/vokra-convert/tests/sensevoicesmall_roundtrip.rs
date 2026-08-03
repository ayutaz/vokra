//! External roundtrip test for the SenseVoiceSmall converter
//! (coverage-audit 2026-08-03 Wave B ticket).
//!
//! Exercises the [`convert_file`] / [`convert_file_licensed`] dispatch
//! (i.e. the outward `ModelKind::SenseVoiceSmall` arm — not the
//! module-internal `convert_sensevoicesmall_file`) with a synthetic BF16
//! safetensors, so the wire-up between the CLI-facing enum and the
//! file-based converter is held under the same regression watch as the
//! sibling reazonspeech_nemo_v2 / hibiki / sber_gigaam_multilingual
//! skeletons.

use std::path::PathBuf;

use vokra_convert::{ModelKind, convert_file, convert_file_licensed, convert_sensevoicesmall_file};
use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufFile};

/// A unique temp path for this test process. Nanosecond suffix keeps
/// parallel `cargo test` runs from colliding.
fn tmp_path(tag: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "vokra-convert-sensevoicesmall-it-{tag}-{}-{}",
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
/// test fixture in `models::sensevoicesmall`, kept private to this file
/// so the external test remains self-contained.
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

/// End-to-end pin: the `ModelKind::SenseVoiceSmall` dispatch arm in
/// `convert_file_licensed` lands the arch / name / category /
/// upstream_hf metadata + a BF16 pass-through tensor + the fail-closed
/// FunASR_MODEL_LICENSE → LicenseClass::Unknown compliance verdict.
#[test]
fn convert_file_dispatch_lands_sensevoicesmall_metadata_and_bf16_passthrough() {
    let (_values, bf16) = synthetic_bf16_payload();
    // SAN-M enhanced Conformer tensor name — the FunASR state-dict key
    // convention preserved verbatim through the
    // `nemo_pt_to_safetensors.py --allow-strip-any` bridge.
    let input_bytes = safetensors_one_bf16(
        "encoder.encoders.0.self_attn.linear_q.weight",
        &[2, 2],
        &bf16,
    );

    let input = tmp_path("dispatch-in");
    let output = tmp_path("dispatch-out");
    std::fs::write(&input, &input_bytes).expect("write input");

    // Route through the outward `convert_file` -> `convert_file_licensed`
    // arm so the ModelKind::SenseVoiceSmall dispatch is exercised
    // end-to-end.
    let summary = convert_file(ModelKind::SenseVoiceSmall, &input, &output).expect("convert");
    assert_eq!(summary.model, ModelKind::SenseVoiceSmall);
    assert_eq!(summary.tensor_count, 1, "one float tensor written");
    assert!(
        summary
            .notes
            .iter()
            .any(|n| n.starts_with("sensevoicesmall:") && n.contains("BF16 passthrough")),
        "notes must surface the sensevoicesmall pass-through counter, got {:?}",
        summary.notes
    );

    let file = GgufFile::open(&output).expect("load output gguf");
    let info = file
        .tensor_info("encoder.encoders.0.self_attn.linear_q.weight")
        .expect("BF16 tensor present");
    assert_eq!(
        info.dtype,
        GgmlType::BF16,
        "BF16 must not be widened at convert time (GGUF type 30 verbatim)"
    );
    assert_eq!(file.tensor_bytes(info), bf16.as_slice());

    // Provenance stamps — arch / name / category / upstream_hf pinned
    // on the artifact itself.
    assert_eq!(
        file.get("vokra.model.arch").and_then(|v| v.as_str()),
        Some("sensevoicesmall")
    );
    assert_eq!(
        file.get("vokra.model.name").and_then(|v| v.as_str()),
        Some("sensevoicesmall")
    );
    assert_eq!(
        file.get("vokra.model.category").and_then(|v| v.as_str()),
        Some("asr")
    );
    assert_eq!(
        file.get("vokra.provenance.upstream_hf")
            .and_then(|v| v.as_str()),
        Some("FunAudioLLM/SenseVoiceSmall")
    );
    assert_eq!(
        file.get("vokra.provenance.license")
            .and_then(|v| v.as_str()),
        Some("FunASR_MODEL_LICENSE")
    );
    // Fail-closed default: the custom FunASR MODEL_LICENSE string is
    // not in `LicenseClass::from_license_str`'s SPDX matcher, so it
    // resolves to `Unknown` — the correct default per
    // `[[feedback-license-signoff-primary-source]]`.
    assert_eq!(
        file.get("vokra.provenance.weight_license")
            .and_then(|v| v.as_str()),
        Some(LicenseClass::Unknown.as_str()),
        "FunASR MODEL_LICENSE must resolve to Unknown (fail-closed) — \
         owner sign-off required for commercial redistribution"
    );

    // The runtime research-flag gate resolves `Unknown` and refuses
    // under a strict policy. If this assertion ever fires it means a
    // downstream re-classifier silently promoted the custom licence to
    // permissive without owner sign-off.
    let res = vokra_core::resolve_license_class(&file);
    assert_eq!(res.class, LicenseClass::Unknown);

    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&output);
}

/// Pins the `--license <spdx>` boundary at the outward
/// `convert_file_licensed` entry point (the CLI dispatch, not the
/// module-internal `convert_sensevoicesmall_file`): overriding with a
/// canonical SPDX id reclassifies the artifact away from the
/// fail-closed `Unknown` default and rewrites the raw SPDX string in
/// the GGUF metadata.
#[test]
fn convert_file_licensed_override_swaps_the_stamped_licence() {
    let (_values, bf16) = synthetic_bf16_payload();
    let input_bytes = safetensors_one_bf16("encoder.embed.weight", &[2, 2], &bf16);

    let input = tmp_path("override-in");
    let output = tmp_path("override-out");
    std::fs::write(&input, &input_bytes).expect("write input");

    // A caller who has read the primary-source FunASR MODEL_LICENSE
    // and made a compliance judgement OR is redistributing under
    // stricter terms may reclassify at this boundary.
    let summary = convert_file_licensed(
        ModelKind::SenseVoiceSmall,
        &input,
        &output,
        Some("apache-2.0"),
    )
    .expect("convert with override");
    assert_eq!(summary.model, ModelKind::SenseVoiceSmall);
    assert_eq!(summary.tensor_count, 1);

    let file = GgufFile::open(&output).expect("load output gguf");
    assert_eq!(
        file.get("vokra.provenance.license")
            .and_then(|v| v.as_str()),
        Some("apache-2.0"),
        "override replaces the raw SPDX string"
    );
    assert_eq!(
        file.get("vokra.provenance.weight_license")
            .and_then(|v| v.as_str()),
        Some(LicenseClass::Permissive.as_str()),
        "apache-2.0 reclassifies away from the fail-closed Unknown default"
    );

    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&output);
}

/// Regression fence against the two entry points drifting apart: a
/// direct call to `convert_sensevoicesmall_file` and a routed call
/// through `ModelKind::SenseVoiceSmall` must produce byte-identical
/// GGUFs. This catches accidental divergence between the module
/// internal path and the CLI dispatch (e.g. a metadata stamp that
/// exists on one path but not the other).
#[test]
fn direct_and_dispatched_paths_produce_byte_identical_gguf() {
    let (_values, bf16) = synthetic_bf16_payload();
    let input_bytes = safetensors_one_bf16("encoder.ln.weight", &[2, 2], &bf16);

    let input = tmp_path("byteid-in");
    let output_direct = tmp_path("byteid-direct");
    let output_dispatch = tmp_path("byteid-dispatch");
    std::fs::write(&input, &input_bytes).expect("write input");

    // (a) direct file-based entry.
    let _report = convert_sensevoicesmall_file(&input, &output_direct, None).expect("direct");
    // (b) CLI dispatch entry (goes through `convert_file_licensed`,
    // ultimately reaches the same file-based function).
    let _summary =
        convert_file(ModelKind::SenseVoiceSmall, &input, &output_dispatch).expect("dispatch");

    let direct_bytes = std::fs::read(&output_direct).expect("read direct output");
    let dispatch_bytes = std::fs::read(&output_dispatch).expect("read dispatch output");
    assert_eq!(
        direct_bytes, dispatch_bytes,
        "direct and dispatched entry points must produce byte-identical GGUFs — \
         a divergence means a metadata stamp exists on one path but not the other"
    );

    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&output_direct);
    let _ = std::fs::remove_file(&output_dispatch);
}
