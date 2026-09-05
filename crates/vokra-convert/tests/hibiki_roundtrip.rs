//! External roundtrip test for the Hibiki-2B converter (coverage-audit
//! 2026-08-03 Wave B ticket).
//!
//! Exercises the [`convert_file`] / [`convert_file_licensed`] dispatch and
//! the module-internal `convert_hibiki_file` boundary with arbitrary BF16
//! safetensors. Hibiki remains `INSPECTION_ONLY` until its fixed composite,
//! provenance, and license evidence is authenticated, so none of these public
//! surfaces may emit a GGUF from an arbitrary input.

use std::path::PathBuf;

use vokra_convert::{ModelKind, convert_file, convert_file_licensed, convert_hibiki_file};

/// A unique temp path for this test process. Nanosecond suffix keeps
/// parallel `cargo test` runs from colliding.
fn tmp_path(tag: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "vokra-convert-hibiki-it-{tag}-{}-{}",
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
/// test fixture in `models::hibiki`, kept private to this file so the
/// external test remains self-contained.
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

/// Non-zero BF16 payload so the refusal is exercised against a plausible
/// arbitrary tensor rather than an empty or degenerate input.
fn synthetic_bf16_payload() -> ([f32; 4], Vec<u8>) {
    let values: [f32; 4] = [1.0, -2.5, 0.15625, 42.0];
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
        .collect();
    (values, bytes)
}

#[test]
fn convert_file_dispatch_rejects_hibiki_without_authenticated_composite() {
    let (_values, bf16) = synthetic_bf16_payload();
    let input_bytes = safetensors_one_bf16("depformer.linear.weight", &[2, 2], &bf16);

    let input = tmp_path("dispatch-in");
    let output = tmp_path("dispatch-out");
    std::fs::write(&input, &input_bytes).expect("write input");

    // Route through the outward `convert_file` -> `convert_file_licensed`
    // arm. A plausible arbitrary tensor must not bypass the fixed Hibiki
    // composite gate or create an output artifact.
    let error = convert_file(ModelKind::Hibiki, &input, &output)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("INSPECTION_ONLY"),
        "explicit refusal: {error}"
    );
    assert!(error.contains("Mimi"), "composite blocker: {error}");
    assert!(!output.exists(), "rejected conversion must not create GGUF");

    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&output);
}

#[test]
fn convert_file_licensed_override_cannot_bypass_hibiki_gate() {
    let (_values, bf16) = synthetic_bf16_payload();
    let input_bytes = safetensors_one_bf16("depformer.emb.weight", &[2, 2], &bf16);

    let input = tmp_path("override-in");
    let output = tmp_path("override-out");
    std::fs::write(&input, &input_bytes).expect("write input");

    // Even an arbitrary permissive SPDX override must not reclassify the
    // unverified input or bypass the fixed CC-BY-4.0 composite gate.
    let error = convert_file_licensed(ModelKind::Hibiki, &input, &output, Some("MIT"))
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("INSPECTION_ONLY"),
        "explicit refusal: {error}"
    );
    assert!(error.contains("CC-BY-4.0"), "license blocker: {error}");
    assert!(!output.exists(), "rejected conversion must not create GGUF");

    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&output);
}

#[test]
fn direct_convert_hibiki_file_and_dispatch_fail_closed() {
    // Both outward surfaces must retain the same inspection-only boundary;
    // neither direct access nor ModelKind dispatch may emit a GGUF.
    let (_values, bf16) = synthetic_bf16_payload();
    let input_bytes = safetensors_one_bf16("out_norm.alpha", &[1, 4], &bf16);

    let input_a = tmp_path("direct-in-a");
    let output_a = tmp_path("direct-out-a");
    let input_b = tmp_path("direct-in-b");
    let output_b = tmp_path("direct-out-b");
    std::fs::write(&input_a, &input_bytes).expect("write A");
    std::fs::write(&input_b, &input_bytes).expect("write B");

    let direct_error = convert_hibiki_file(&input_a, &output_a, None)
        .unwrap_err()
        .to_string();
    assert!(
        direct_error.contains("INSPECTION_ONLY"),
        "direct refusal: {direct_error}"
    );
    let dispatch_error = convert_file(ModelKind::Hibiki, &input_b, &output_b)
        .unwrap_err()
        .to_string();
    assert!(
        dispatch_error.contains("INSPECTION_ONLY"),
        "dispatch refusal: {dispatch_error}"
    );
    assert!(
        direct_error.contains("SentencePiece") && dispatch_error.contains("SentencePiece"),
        "both paths must name the fixed composite blocker"
    );
    assert!(!output_a.exists(), "direct refusal must not create GGUF");
    assert!(!output_b.exists(), "dispatch refusal must not create GGUF");

    let _ = std::fs::remove_file(&input_a);
    let _ = std::fs::remove_file(&output_a);
    let _ = std::fs::remove_file(&input_b);
    let _ = std::fs::remove_file(&output_b);
}
