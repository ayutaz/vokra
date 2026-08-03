//! Coverage-audit 2026-08-03 Wave A ticket: integration test that a
//! synthetic RNNoise-shaped safetensors checkpoint round-trips through the
//! public [`convert_file`] dispatch (`ModelKind::Rnnoise`) into a
//! well-formed GGUF the runtime loader can parse. The narrower unit tests
//! live inline in `crates/vokra-convert/src/models/rnnoise.rs`; this test
//! guards the outer `ModelKind` dispatch + `ConvertSummary` reporting +
//! `vokra.provenance.upstream_url` stamp so a future refactor that dropped
//! the RNNoise arm from `convert_file_licensed` would fail loudly here.
//!
//! Mirror of `crates/vokra-convert/tests/roundtrip.rs` (M0-03-T17) — no
//! large real checkpoint is committed. Full real-weight parity against
//! Xiph's reference C forward is the owner deliverable
//! (`docs/license-audit.md` §3.1 sign-off queue for RNNoise v0.2).

use std::path::PathBuf;

use vokra_convert::{ConvertSummary, ModelKind, convert_file};
use vokra_core::LicenseClass;
use vokra_core::gguf::{GgufFile, chunks};

/// A unique temp path for this test process (moshi / emotion2vec pattern
/// — no external `tempfile` dep, preserving zero-dep NFR-DS-02).
fn tmp_path(tag: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "vokra-rnnoise-it-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_default(),
    ));
    p
}

/// Builds a synthetic RNNoise-shaped safetensors buffer: a small F32
/// `input_dense.kernel` and a smaller F32 `vad_output.bias`. Tensor names
/// track the RNNoise topology documented in
/// `github.com/xiph/rnnoise/blob/main/src/denoise.c` so a downstream
/// reader can eyeball the round-trip as "recognisably RNNoise-shaped";
/// the actual per-layer axes / dtype (int8 quantized in the real Xiph
/// release) are the owner deliverable.
fn synthetic_rnnoise_safetensors() -> Vec<u8> {
    // Non-zero payloads so a silent widen / drop regression falls out
    // rather than trivially round-tripping a zero buffer. Six + two
    // values keeps the total payload at 32 bytes.
    let kernel: Vec<u8> = [1.0f32, -2.0, 3.5, -0.25, 100.0, 0.001]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    assert_eq!(kernel.len(), 24, "6 elements × 4 bytes F32 payload");
    let bias: Vec<u8> = [-42.0f32, 7.5]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    assert_eq!(bias.len(), 8, "2 elements × 4 bytes F32 payload");
    let header = r#"{"input_dense.kernel":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]},"vad_output.bias":{"dtype":"F32","shape":[2],"data_offsets":[24,32]}}"#;
    let mut buf = Vec::new();
    buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
    buf.extend_from_slice(header.as_bytes());
    buf.extend_from_slice(&kernel);
    buf.extend_from_slice(&bias);
    buf
}

#[test]
fn rnnoise_safetensors_roundtrips_through_convert_file_dispatch() {
    let input = tmp_path("in");
    let output = tmp_path("out");
    std::fs::write(&input, synthetic_rnnoise_safetensors()).expect("write input");

    let summary: ConvertSummary =
        convert_file(ModelKind::Rnnoise, &input, &output).expect("convert");
    assert_eq!(summary.model, ModelKind::Rnnoise);
    assert_eq!(summary.tensor_count, 2, "both float tensors must land");
    assert!(
        summary.output_bytes > 0,
        "output GGUF must be non-empty on disk"
    );
    // Note: `ConvertSummary::metadata_count` on the file-based early-return
    // path is stamped as `0` for the Phase 5 fleet + coverage-audit
    // pattern (the fleet dispatch does not read the builder's metadata
    // counter back before returning — see `convert_file_licensed` in
    // `lib.rs`). The real metadata count is asserted below by GGUF parse.

    let file = GgufFile::open(&output).expect("load output gguf");
    assert_eq!(
        file.tensors().len(),
        2,
        "both synthetic tensors survive the round-trip"
    );
    assert!(file.tensor_info("input_dense.kernel").is_some());
    assert!(file.tensor_info("vad_output.bias").is_some());

    // Arch + name + category — the three axes downstream tooling
    // (publish-one.sh / make_model_card.py / zoo manifest) reads.
    assert_eq!(
        file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
        Some("rnnoise"),
        "vokra.model.arch pins the short arch tag (distinct from DFN3's `denoise`)"
    );
    assert_eq!(
        file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
        Some("rnnoise-v0.2"),
        "vokra.model.name matches the publish repo slug"
    );
    assert_eq!(
        file.get("vokra.model.category").and_then(|v| v.as_str()),
        Some("denoise"),
        "category is the shared denoise family (sibling of DFN3)"
    );

    // Provenance — RNNoise ships from GitHub Release, not HF, so the
    // URL slot must be stamped (and `upstream_hf` must be absent to
    // avoid misleading the model-card generator).
    assert_eq!(
        file.get("vokra.provenance.upstream_url")
            .and_then(|v| v.as_str()),
        Some("https://github.com/xiph/rnnoise/releases/tag/v0.2"),
        "upstream_url pins the GitHub Release the blob ships from"
    );
    assert!(
        file.get("vokra.provenance.upstream_hf").is_none(),
        "upstream_hf must NOT be stamped for a non-HF release (would misrepresent \
         the serving location)"
    );

    // License stamp — BSD-3-Clause = Permissive per PERMISSIVE_TOKENS.
    assert_eq!(
        file.get(chunks::KEY_PROVENANCE_LICENSE)
            .and_then(|v| v.as_str()),
        Some("bsd-3-clause")
    );
    assert_eq!(
        file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
            .and_then(|v| v.as_str()),
        Some(LicenseClass::Permissive.as_str())
    );

    // Schema stamp — the writer emits this unconditionally.
    assert!(
        file.get(chunks::KEY_SCHEMA_VERSION).is_some(),
        "vokra.schema.version must be stamped"
    );
    assert!(
        file.get(chunks::KEY_SCHEMA_PRODUCER).is_some(),
        "vokra.schema.producer must be stamped"
    );

    // Byte-identical payload survival — a silent widen / re-quantize
    // would flip these payloads.
    let expected_kernel: Vec<u8> = [1.0f32, -2.0, 3.5, -0.25, 100.0, 0.001]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    let expected_bias: Vec<u8> = [-42.0f32, 7.5]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    assert_eq!(
        file.tensor_data("input_dense.kernel").unwrap(),
        expected_kernel.as_slice(),
        "F32 kernel payload must be byte-identical to input"
    );
    assert_eq!(
        file.tensor_data("vad_output.bias").unwrap(),
        expected_bias.as_slice(),
        "F32 bias payload must be byte-identical to input"
    );

    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&output);
}
