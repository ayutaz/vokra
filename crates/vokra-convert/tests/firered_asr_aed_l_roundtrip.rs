//! FireRedTeam/FireRedASR-AED-L converter — external integration test
//! (coverage-audit wave-b, 2026-08-03).
//!
//! CI-resident round-trip: a synthetic mixed-dtype safetensors buffer is
//! written to disk, run through both the public [`convert_file`] entry
//! point (dispatched by [`ModelKind::FireredAsrAedL`]) and the direct
//! [`convert_firered_asr_aed_l_file`] re-export, and the resulting GGUF
//! is parsed back with [`GgufFile`]. No large real checkpoint is
//! committed; the real ~2.2 GB `FireRedTeam/FireRedASR-AED-L` end-to-end
//! is a manual local run of the `vokra-convert` binary, gated on the
//! prep script `tools/parity/firered_asr_aed_l_prepare_checkpoint.py`.
//!
//! Aliased-slug dispatch coverage (`from_arg` for the underscore /
//! upstream HF slug variants) lives here rather than in-module because
//! the whole `ModelKind` round-trip is a crate-root API concern (mirror
//! of the frcrn integration test's dispatch coverage).

use std::path::PathBuf;

use vokra_convert::{
    FireredAsrAedLReport, ModelKind, convert_file, convert_file_licensed,
    convert_firered_asr_aed_l_file,
};
use vokra_core::gguf::{GgmlType, GgufFile};

/// A unique temp path for this test process (mirror of the other
/// integration tests' `tmp_path` — no `tempfile` dep, preserving
/// zero-dep NFR-DS-02).
fn tmp_path(tag: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "vokra-firered-asr-aed-l-it-{tag}-{}-{}.bin",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_default(),
    ));
    p
}

/// Builds a synthetic safetensors buffer with three tensors:
///
/// * one BF16 encoder weight (`[2, 3]`, 12 B) at bytes [0, 12)
/// * one F32 decoder weight (`[2, 2]`, 16 B) at bytes [12, 28)
/// * one F16 decoder bias (`[2]`, 4 B) at bytes [28, 32)
///
/// Non-zero payloads on all three so a silent widen / downcast can't
/// round-trip trivially through F32 / F16 widen.
fn synthetic_three_dtype_safetensors() -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
    // BF16 payload: 6 non-zero half-floats (top 16 bits of f32).
    let bf16_vals: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
    let bf16: Vec<u8> = bf16_vals
        .iter()
        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
        .collect();
    assert_eq!(bf16.len(), 12, "6 elements × 2 bytes BF16 payload");
    // F32 payload: 4 non-zero floats.
    let f32_vals: [f32; 4] = [1.5, -2.0, 3.25, -4.125];
    let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(f32_bytes.len(), 16, "4 elements × 4 bytes F32 payload");
    // F16 payload: 2 non-zero half-floats.
    let f16_patterns: [u16; 2] = [0x3C00, 0xC000]; // 1.0, -2.0
    let f16_bytes: Vec<u8> = f16_patterns.iter().flat_map(|p| p.to_le_bytes()).collect();
    assert_eq!(f16_bytes.len(), 4, "2 elements × 2 bytes F16 payload");

    // Assemble header — offsets are relative to the data-region start.
    let header = r#"{"encoder.blocks.0.attn.qkv_proj.weight":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]},"decoder.blocks.0.self_attn.out_proj.weight":{"dtype":"F32","shape":[2,2],"data_offsets":[12,28]},"decoder.blocks.0.self_attn.out_proj.bias":{"dtype":"F16","shape":[2],"data_offsets":[28,32]}}"#;
    let mut buf = Vec::new();
    buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
    buf.extend_from_slice(header.as_bytes());
    buf.extend_from_slice(&bf16);
    buf.extend_from_slice(&f32_bytes);
    buf.extend_from_slice(&f16_bytes);
    (buf, bf16, f32_bytes, f16_bytes)
}

/// FireRedASR-AED-L: three-dtype round-trip via the public
/// `convert_file` entry point — dispatch by `ModelKind::FireredAsrAedL`
/// must reach the pass-through converter, and all three tensor dtypes
/// (BF16 / F32 / F16) must survive with their upstream names + dtypes +
/// payloads intact.
#[test]
fn three_dtype_roundtrip_through_convert_file() {
    let (input_bytes, bf16_payload, f32_payload, f16_payload) = synthetic_three_dtype_safetensors();
    let input = tmp_path("three-in");
    let output = tmp_path("three-out");
    std::fs::write(&input, &input_bytes).expect("write input");

    let summary = convert_file(ModelKind::FireredAsrAedL, &input, &output).expect("convert");
    assert_eq!(
        summary.tensor_count, 3,
        "all three float tensors must survive the ModelKind dispatch arm"
    );

    // Round-trip through the emitted GGUF.
    let file = GgufFile::open(&output).expect("load output GGUF");
    assert_eq!(file.tensors().len(), 3);

    let bf16_info = file
        .tensor_info("encoder.blocks.0.attn.qkv_proj.weight")
        .expect("BF16 tensor present");
    assert_eq!(bf16_info.dtype, GgmlType::BF16, "BF16 stays BF16 verbatim");
    assert_eq!(bf16_info.dimensions, vec![2, 3]);
    assert_eq!(file.tensor_bytes(bf16_info), bf16_payload.as_slice());

    let f32_info = file
        .tensor_info("decoder.blocks.0.self_attn.out_proj.weight")
        .expect("F32 tensor present");
    assert_eq!(f32_info.dtype, GgmlType::F32);
    assert_eq!(f32_info.dimensions, vec![2, 2]);
    assert_eq!(file.tensor_bytes(f32_info), f32_payload.as_slice());

    let f16_info = file
        .tensor_info("decoder.blocks.0.self_attn.out_proj.bias")
        .expect("F16 tensor present");
    assert_eq!(f16_info.dtype, GgmlType::F16);
    assert_eq!(f16_info.dimensions, vec![2]);
    assert_eq!(file.tensor_bytes(f16_info), f16_payload.as_slice());

    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&output);
}

/// Direct `convert_firered_asr_aed_l_file` re-export must be reachable
/// from external callers and must return the counter-typed
/// [`FireredAsrAedLReport`] rather than the generic `ConvertSummary`.
/// Guards the `pub use` re-export line at the crate root.
#[test]
fn direct_entry_point_returns_typed_report() {
    let (input_bytes, _, _, _) = synthetic_three_dtype_safetensors();
    let input = tmp_path("direct-in");
    let output = tmp_path("direct-out");
    std::fs::write(&input, &input_bytes).expect("write input");

    let report: FireredAsrAedLReport =
        convert_firered_asr_aed_l_file(&input, &output, None).expect("convert");
    assert_eq!(report.read, 3, "three tensors observed on input");
    assert_eq!(report.written, 3, "three tensors written verbatim");
    assert_eq!(report.skipped_non_float, 0);
    assert_eq!(report.bf16_passthrough, 1, "one BF16 tensor in fixture");

    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&output);
}

/// The `--model` argument parser must accept the underscore /
/// hyphenated / upstream HF release id / case-preserving upstream HF
/// slug spellings, and every spelling must dispatch to the same
/// pass-through converter (`ModelKind::FireredAsrAedL`).
#[test]
fn from_arg_aliases_dispatch_to_same_converter() {
    let aliases = [
        "firered-asr-aed-l",
        "firered_asr_aed_l",
        "fireredasr-aed-l",
        "fireredasr_aed_l",
        "firered-asr-aed",
        "firered_asr_aed",
        "fireredteam/firered-asr-aed-l",
        "fireredteam/firered_asr_aed_l",
        "fireredteam/fireredasr-aed-l",
        "FireRedTeam/FireRedASR-AED-L",
    ];
    for alias in aliases {
        assert_eq!(
            ModelKind::from_arg(alias),
            Some(ModelKind::FireredAsrAedL),
            "alias {alias:?} must resolve to FireredAsrAedL"
        );
    }
    // Canonical CLI slug survives the round trip `as_arg -> from_arg`.
    let canonical = ModelKind::FireredAsrAedL.as_arg();
    assert_eq!(canonical, "firered-asr-aed-l");
    assert_eq!(
        ModelKind::from_arg(canonical),
        Some(ModelKind::FireredAsrAedL)
    );
}

/// `convert_file_licensed` must forward the `license` override to the
/// converter — pinning that the `convert_file` -> `convert_file_licensed`
/// -> `convert_firered_asr_aed_l_file` chain honors the override.
#[test]
fn convert_file_licensed_forwards_override() {
    let (input_bytes, _, _, _) = synthetic_three_dtype_safetensors();
    let input = tmp_path("license-in");
    let output = tmp_path("license-out");
    std::fs::write(&input, &input_bytes).expect("write input");

    // Override the default apache-2.0 with mit — both Permissive, so
    // the class stays; only the SPDX flips.
    convert_file_licensed(ModelKind::FireredAsrAedL, &input, &output, Some("mit"))
        .expect("convert with license override");

    let file = GgufFile::open(&output).expect("load output GGUF");
    assert_eq!(
        file.get(vokra_core::gguf::chunks::KEY_PROVENANCE_LICENSE)
            .and_then(|v| v.as_str()),
        Some("mit"),
        "license override must reach the artifact"
    );

    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&output);
}
